import torch
import torch.nn as nn
import torch.nn.functional as F
from ovdet.models.vlms.clip import clip as CLIP
from mmengine.structures import InstanceData
from ovdet.methods.builder import build_queue, OVD
from ovdet.methods.detic.utils import multi_apply
import random
import copy
import numpy as np


@OVD.register_module()
class GeneratedRegionCaption(nn.Module):
    def __init__(self,
                 base_batch_size=32,
                 bce_bias=0.0, norm_temp=50.0, gen_caption_weight=1.0,
                 queue_cfg=dict(lengths=[256], id_length=0, ),
                 neg_weight=0.125,
                 name='gen_region_caption'):
        super(GeneratedRegionCaption, self).__init__()
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

    @torch.no_grad()
    def get_caption_features(self, captions, device, clip_model):
        tokens = CLIP.tokenize_dynamic(captions, truncate=True).to(device)
        caption_features = clip_model.text_encoder.encode_text(tokens, normalize=True).float()
        return caption_features

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
        return InstanceData(
            bboxes=boxes,
            labels=torch.tensor([label] * len(boxes), device=boxes.device, dtype=torch.int64),
            scores=torch.tensor([1.0] * len(boxes), device=boxes.device),
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
        assert len(all_cap_embeddings) == len(region_embeddings), \
            f'{all_cap_embeddings.shape},{region_embeddings.shape}'

        global_clip_caption_features = self.queues.get_queue(self.queue_name)
        contrast_clip_caption_features = torch.cat([all_cap_embeddings,
                                                    global_clip_caption_features], dim=0)

        with torch.no_grad():
            positives = all_cap_embeddings @ contrast_clip_caption_features.T
            positives = positives > 0.99
        label_matrix = torch.eye(n=len(region_embeddings), m=len(contrast_clip_caption_features),
                                 dtype=region_embeddings.dtype, device=device)
        label_matrix[positives] = 1.
        similarity_matrix = region_embeddings @ contrast_clip_caption_features.T * self.norm_temp
        similarity_matrix += self.bce_bias
        loss = F.binary_cross_entropy_with_logits(similarity_matrix, label_matrix,
                                                  reduction='none')

        loss_mask = loss.new_ones((loss.shape))
        loss_mask[0:len(loss), 0:len(loss)] = 0
        for indices in range(len(loss_mask)):
            loss_mask[indices, indices] = 1
        loss *= loss_mask

        # adjust positive weights in the case of multi-label target
        num_pos_labels = label_matrix.sum(-1, keepdim=True)
        assert (num_pos_labels > 0).all()
        pos_weights = torch.ones_like(loss) / num_pos_labels
        neg_weights = torch.ones_like(loss) * self.neg_weight
        loss *= torch.where(label_matrix > 0.0, pos_weights, neg_weights)
        loss = loss.sum(-1).mean()

        if update_queue:
            queue_update = {self.queue_name: all_cap_embeddings}
            self.queues.dequeue_and_enqueue(queue_update)
        if self.base_batch_size is not None:
            loss *= (label_matrix.shape[0] / self.base_batch_size)
        losses = {self.name: loss * self.gen_caption_weight}
        return losses
