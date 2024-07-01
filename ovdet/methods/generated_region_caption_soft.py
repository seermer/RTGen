import torch
import torch.nn as nn
import torch.nn.functional as F
from ovdet.models.vlms.clip import clip as CLIP
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_overlaps
from ovdet.methods.builder import build_queue, OVD
from ovdet.methods.detic.utils import multi_apply
import random
import copy
import numpy as np


@OVD.register_module()
class GeneratedRegionCaptionSoft(nn.Module):
    def __init__(self,
                 base_batch_size=32,
                 bce_bias=0.0, norm_temp=50.0, gen_caption_weight=1.0,
                 queue_cfg=dict(lengths=[256], id_length=0, ),
                 neg_weight=0.125,
                 name='gen_region_caption',
                 soft_anchor_neg_weight=0.125,
                 soft_anchor_weight=0.5):
        super(GeneratedRegionCaptionSoft, self).__init__()
        self.base_batch_size = base_batch_size
        self.norm_temp = norm_temp
        self.queues = build_queue(queue_cfg)
        self.gen_caption_weight = gen_caption_weight
        self.queue_name = queue_cfg.get('names') or queue_cfg.get('name')
        if isinstance(self.queue_name, list):
            self.queue_name = self.queue_name[0]

        if bce_bias is None:
            self.bce_bias = 0.0
        else:
            self.bce_bias = nn.Parameter(torch.ones(1) * bce_bias)
        self.neg_weight = neg_weight
        self.name = f'loss_{name}'
        self.soft_anchor_neg_weight = soft_anchor_neg_weight
        self.soft_anchor_weight = soft_anchor_weight

    @torch.no_grad()
    def get_caption_features(self, captions, device, clip_model):
        tokens = CLIP.tokenize_dynamic(captions, truncate=True).to(device)
        caption_features = clip_model.text_encoder.encode_text(tokens, normalize=True).float()
        return caption_features

    @staticmethod
    @torch.no_grad()
    def filter_bbox_by_iou(gts, proposals):
        ious = bbox_overlaps(proposals, gts)
        loss_weights = torch.zeros_like(ious)

        # positives
        positive_vals, positive_inds = ious.max(dim=1)
        loss_weights.scatter_(1, positive_inds.unsqueeze(1), positive_vals.unsqueeze(1))
        positive_mask = ious > 0.8
        loss_weights[~positive_mask] = 0.  # positive requires to be max iou of gts and iou>0.5
        positives = positive_mask.any(dim=1)
        num_positives = positives.sum()

        # negatives
        negative_mask = (ious <= 0.6) & (ious >= 0.1)
        negatives = negative_mask.any(dim=1)
        num_negatives = negatives.sum()
        num_remove = num_negatives - num_positives * 3
        if num_remove > 0:
            removable = negatives & ~positives
            removed_inds = removable.argwhere()
            removed_inds = removed_inds[torch.randperm(len(removed_inds))[:num_remove]]
            negative_mask[removed_inds] = False
        # negatives = negatives.argwhere()
        # num_negatives = len(negatives)
        # if num_negatives > num_positives * 3:
        #     negatives = negatives[torch.randperm(num_negatives)]
        #     negative_mask[negatives[num_positives * 3:]] = False
        loss_weights[negative_mask] = -1.

        # filter proposals without any negative/positive target
        valid_mask = loss_weights.any(dim=1)
        proposals = proposals[valid_mask].detach().clone()
        loss_weights = loss_weights[valid_mask]

        return proposals, loss_weights

    @staticmethod  # The caption version of detic use image boxes
    def sample(rpn_results, batch_data_sample, **kwargs):
        captions = batch_data_sample.metainfo['captions']
        boxes = batch_data_sample.gt_instances.bboxes
        label = batch_data_sample.gt_instances.labels[0]
        if label == 0:
            valid_mask = []
            for caption in captions:
                caption_valid = [True] * (len(caption) - 1)
                caption_valid[0] = False
                valid_mask += caption_valid
            boxes = boxes[valid_mask]
        num_gts = len(boxes)
        labels = [label] * num_gts
        assert num_gts > 0, f'no valid gt boxes found'

        soft_anchors, loss_weights = GeneratedRegionCaptionSoft.filter_bbox_by_iou(boxes, rpn_results.bboxes)
        if len(soft_anchors) > 0:
            boxes = torch.vstack([boxes, soft_anchors])
            loss_weights = torch.vstack([torch.ones((num_gts, num_gts),
                                                    device=boxes.device, dtype=loss_weights.dtype),
                                         loss_weights])
            labels += [-1] * len(soft_anchors)
        else:
            loss_weights = torch.ones((num_gts, num_gts),
                                      device=boxes.device, dtype=loss_weights.dtype)

        return InstanceData(
            bboxes=boxes,
            labels=torch.tensor(labels, device=boxes.device, dtype=torch.int64),
            scores=torch.tensor([1.0] * len(boxes), device=boxes.device),
            soft_anchor_weights=loss_weights,
            metainfo=copy.deepcopy(batch_data_sample.metainfo)
        )

    def get_losses(self, region_embeddings, sampling_results, clip_model,
                   *args, update_queue=True, **kwargs):
        region_embeddings = F.normalize(region_embeddings, dim=-1)
        device = region_embeddings.device
        label = sampling_results[0].labels[0]
        all_cap_embeddings = []
        all_masks = []
        for i, sampling_result in enumerate(sampling_results):
            base_idx = 0
            current_mask = sampling_result.get('valid_mask', None)
            if label == 1:  # generated images
                captions = sampling_result['captions']
                while not isinstance(captions[0], str):
                    captions = [cap for b in captions for cap in b]
                all_cap_embeddings += captions

                if current_mask is None:
                    current_mask = np.ones((len(captions),), dtype=np.bool_)
                assert len(captions) == len(current_mask), f'{len(captions)} != {len(current_mask)}'
                all_masks.append(current_mask)
                continue
            for caption in sampling_result['captions']:
                if current_mask is None:
                    all_cap_embeddings.append(caption[2:])
                else:
                    all_cap_embeddings.append(caption[2:][current_mask[base_idx + 1:base_idx + len(caption) - 1]])
                base_idx += len(caption) - 1
        if label == 1:
            all_cap_embeddings = self.get_caption_features(all_cap_embeddings, device, clip_model)
            all_masks = np.concatenate(all_masks)
            all_cap_embeddings = all_cap_embeddings[all_masks]
        else:
            all_cap_embeddings = torch.cat(all_cap_embeddings, dim=0).to(device)
        all_cap_embeddings = F.normalize(all_cap_embeddings, dim=-1)
        # assert len(all_cap_embeddings) == len(region_embeddings), \
        #     f'{all_cap_embeddings.shape},{region_embeddings.shape}'

        global_clip_caption_features = self.queues.get_queue(self.queue_name)
        contrast_clip_caption_features = torch.cat([all_cap_embeddings,
                                                    global_clip_caption_features], dim=0)

        label_vector = torch.cat([sampling_result.labels for sampling_result in sampling_results])
        is_soft_anchor = label_vector < 0
        is_gt = ~is_soft_anchor
        num_gts = len(all_cap_embeddings)
        assert is_gt.sum() == num_gts, f'{is_gt.sum()} != {num_gts}'

        loss_weights = torch.ones((len(region_embeddings), len(contrast_clip_caption_features)),
                                  dtype=region_embeddings.dtype, device=device)
        soft_anchor_loss_weights = loss_weights[num_gts:, :num_gts]  # without queue

        label_matrix = torch.zeros_like(loss_weights)
        all_gt_labels = label_matrix[:num_gts]
        all_soft_anchor_labels = label_matrix[num_gts:]
        gt_queue_labels = all_gt_labels[:, num_gts:]
        soft_anchor_queue_labels = all_soft_anchor_labels[:, num_gts:]
        soft_anchor_labels = all_soft_anchor_labels[:, :num_gts]  # without queue

        with torch.no_grad():
            positives = all_cap_embeddings @ contrast_clip_caption_features.T
            positives = positives > 0.99
            all_gt_labels[positives] = 1.

            loss_weights[:num_gts, :num_gts] = torch.eye(num_gts,  # ground truths without queue
                                                         device=device, dtype=loss_weights.dtype)
            loss_weights[num_gts:, :num_gts] = 0.  # soft anchors without queue
            loss_weights[num_gts:, num_gts:] = self.neg_weight  # 0.125 for soft anchors vs queue

            row_idx, col_idx = 0, 0
            for sampling_result in sampling_results:
                weights = sampling_result.soft_anchor_weights[sampling_result.labels < 0]  # label < 0 is soft anchor
                positive_mask = weights > 0
                h, w = weights.shape
                soft_anchor_loss_weights[row_idx:row_idx + h, col_idx:col_idx + w] = weights.abs()
                soft_anchor_loss_weights[row_idx:row_idx + h, col_idx:col_idx + w][weights < 0] *= \
                    self.soft_anchor_neg_weight

                contains_positive = positive_mask.any(1)
                positive_mask = positive_mask.to(soft_anchor_labels.dtype)
                positive_ids = positive_mask.argmax(1)[contains_positive]
                soft_anchor_queue_labels[row_idx:row_idx + h][contains_positive] = \
                    gt_queue_labels[col_idx:][positive_ids]
                soft_anchor_labels[row_idx:row_idx + h, col_idx:col_idx + w] = positive_mask

                row_idx += h
                col_idx += w

            loss_weights[num_gts:, num_gts:][soft_anchor_queue_labels > 0.5] = 1

        # reorder so that all gt are before soft anchors
        region_embeddings = torch.vstack([region_embeddings[is_gt],
                                          region_embeddings[is_soft_anchor]])

        similarity_matrix = region_embeddings @ contrast_clip_caption_features.T * self.norm_temp
        similarity_matrix += self.bce_bias
        loss = F.binary_cross_entropy_with_logits(similarity_matrix, label_matrix,
                                                  reduction='none')
        loss *= loss_weights

        # loss_mask = loss.new_ones((loss.shape))
        # loss_mask[0:len(loss), 0:len(loss)] = 0
        # for indices in range(len(loss_mask)):
        #     loss_mask[indices, indices] = 1
        # loss *= loss_mask

        # adjust positive weights in the case of multi-label target
        gt_label_matrix = label_matrix[:num_gts]
        num_pos_labels = gt_label_matrix.sum(-1, keepdim=True)
        # assert num_pos_labels.all()
        gt_loss = loss[:num_gts]
        pos_weights = torch.ones_like(gt_loss) / num_pos_labels
        neg_weights = torch.ones_like(gt_loss) * self.neg_weight
        gt_loss *= torch.where(gt_label_matrix > 0.0, pos_weights, neg_weights)
        soft_anchor_loss = loss[num_gts:]
        num_pos_labels = label_matrix[num_gts:].sum(-1, keepdim=True)
        pos_weights = torch.ones_like(soft_anchor_loss) / torch.clip_(num_pos_labels, 1)
        soft_anchor_loss *= torch.where(label_matrix[num_gts:] > 0.0, pos_weights, 1)

        loss1 = loss[:num_gts].sum(-1).mean()
        loss2 = loss[num_gts:].sum(-1).mean() * self.soft_anchor_weight
        loss_val = torch.tensor(0, device=loss1.device, dtype=loss1.dtype)

        if torch.isfinite(loss1):
            loss_val += loss1
        else:
            print('loss1 nan!!')
        if torch.isfinite(loss2):
            loss_val += loss2
        else:
            print('loss2 nan!!')

        if update_queue:
            queue_update = {self.queue_name: all_cap_embeddings}
            self.queues.dequeue_and_enqueue(queue_update)
        if self.base_batch_size is not None:
            loss_val *= (num_gts / self.base_batch_size)
        losses = {self.name: loss_val * self.gen_caption_weight}
        return losses
