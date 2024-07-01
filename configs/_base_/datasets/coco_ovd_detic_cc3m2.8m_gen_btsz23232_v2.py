# dataset settings
_base_ = 'mmdet::_base_/datasets/coco_detection.py'
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
file_client_args = dict(backend='disk')
branch_field = ['det_batch', 'generated_caption_batch', 'generated_image_batch'] # remove caption_batch, mosaic_batch, generated_caption_mosaic_batch
det_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PackDetInputs')
    dict(type='MultiBranch',
         branch_field=branch_field,
         det_batch=dict(type='PackDetInputs'))
]

# keep this
generated_caption_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadCaption'),
    dict(type='RandomResize',
         scale=(667, 400),
         ratio_range=(0.5, 1.5),
         keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='MultiBranch',
         branch_field=branch_field,
         generated_caption_batch=dict(type='PackDetInputs',
                                      meta_keys=['img_id', 'img_path', 'ori_shape',
                                                 'img_shape', 'scale_factor', 'valid_mask',
                                                 'flip', 'flip_direction', 'captions', 'image_ids'])
         )
]

generated_image_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadCaption'),
    dict(type='RandomResize',
         scale=(667, 400),
         ratio_range=(0.5, 1.5),
         keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='MultiBranch',
         branch_field=branch_field,
         generated_image_batch=dict(type='PackDetInputs',
                                           meta_keys=['img_id', 'img_path', 'ori_shape',
                                                      'img_shape', 'scale_factor', 'valid_mask',
                                                      'flip', 'flip_direction', 'captions', 'image_ids'])
         )
]


det_dataset = dict(
    type='CocoDataset',
    data_root=data_root,
    ann_file='wusize/instances_train2017_base.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=det_pipeline)

# keep
generated_caption_dataset = dict(
    type='GeneratedCaptionDataset',
    ann_file='data/cc3m/annotations/cc3m_blip2_full_and_region_captions_v2.pkl',
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=generated_caption_pipeline
)

generated_image_dataset = dict(
    type='GeneratedImageDataset',
    ann_file='/media/Bootes/fangyic/ramdisk/cleaned_inpainting_box_text_1M+/media/Bootes/fangyic/ramdisk/generation_samples/inpainting_box_text',
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=generated_image_pipeline
)

batch_split = [2, 32, 32]
train_dataloader = dict(
    batch_size=sum(batch_split),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='CustomGroupMultiSourceSampler',
                 batch_size=sum(batch_split),
                 source_ratio=batch_split),
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[det_dataset, generated_caption_dataset, generated_image_dataset])  
)

val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'wusize/instances_val2017_base.json',
        metric='bbox',
        prefix='Base',
        format_only=False),
    dict(
        type='CocoMetric',
        ann_file=data_root + 'wusize/instances_val2017_novel.json',
        metric='bbox',
        prefix='Novel',
        format_only=False)
]
test_evaluator = val_evaluator
