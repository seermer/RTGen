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
class GeneratedCaption(nn.Module):
    def __init__(self,
                 base_batch_size=32,
                 bce_bias=0.0, norm_temp=50.0, gen_caption_weight=1.0,
                 queue_cfg=dict(lengths=[256], id_length=0, ),
                 neg_weight=0.125,
                 sampling_method='half',
                 name='gen_caption'):
        super(GeneratedCaption, self).__init__()
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
        self.sampling_method = sampling_method
        self.name = f'loss_{name}'

    @staticmethod
    def sample(rpn_results, batch_data_sample, **kwargs):
        captions = batch_data_sample.metainfo['captions']
        boxes = batch_data_sample.gt_instances.bboxes
        valid_mask = []
        for caption in captions:
            caption_valid = [False] * (len(caption) - 1)
            caption_valid[0] = True
            valid_mask += caption_valid
        boxes = boxes[valid_mask]
        return InstanceData(
            bboxes=boxes,
            labels=torch.tensor([0] * len(boxes), device=boxes.device, dtype=torch.int64),
            scores=torch.tensor([1.0] * len(boxes), device=boxes.device),
            caption_gts=[[i] for i in range(len(boxes))],
            metainfo=copy.deepcopy(batch_data_sample.metainfo)
        )

    def get_losses(self, region_embeddings, sampling_results, clip_model,
                   *args, update_queue=True, **kwargs):
        region_embeddings = F.normalize(region_embeddings, dim=-1)
        device = region_embeddings.device
        all_cap_embeddings = []
        caption_gts = []
        base_idx = 0
        for sampling_result in sampling_results:
            for caption in sampling_result.captions:
                if self.sampling_method == 'half':
                    all_cap_embeddings.append(random.choice(caption[:2]))
                elif self.sampling_method == 'gen_only':
                    all_cap_embeddings.append(caption[1])
                elif self.sampling_method == 'orig_only':
                    all_cap_embeddings.append(caption[0])
                else:
                    raise NotImplementedError(self.sampling_method)
            caption_gts += [[base_idx + idx for idx in caption_gt]
                            for caption_gt in sampling_result.caption_gts]
            base_idx += len(sampling_result.captions)
        assert len(caption_gts) == len(region_embeddings), f'{len(caption_gts)} != {len(region_embeddings)}'
        all_cap_embeddings = torch.stack(all_cap_embeddings).to(device)
        all_cap_embeddings = F.normalize(all_cap_embeddings, dim=-1)

        global_clip_caption_features = self.queues.get_queue(self.queue_name)
        contrast_clip_caption_features = torch.cat([all_cap_embeddings,
                                                    global_clip_caption_features], dim=0)

        # matrix 0
        label_matrix = torch.zeros((len(region_embeddings), len(contrast_clip_caption_features)),
                                   dtype=region_embeddings.dtype, device=device)
        for row, indices in zip(label_matrix, caption_gts):
            row[indices] = 1.

        similarity_matrix = region_embeddings @ contrast_clip_caption_features.T * self.norm_temp
        similarity_matrix += self.bce_bias
        loss = F.binary_cross_entropy_with_logits(similarity_matrix, label_matrix,
                                                  reduction='none')
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
