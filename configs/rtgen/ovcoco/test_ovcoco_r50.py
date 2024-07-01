_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50-caffe-c4.py',
    'mmdet::_base_/datasets/coco_detection.py',
    '../../_base_/iter_based_runtime.py'
]

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
file_client_args = dict(backend='disk')

reg_layer = [
    dict(type='Linear', in_features=2048, out_features=2048),
    dict(type='ReLU', inplace=True),
    dict(type='Linear', in_features=2048, out_features=4)
]

clip_cfg = dict(  # ViT-B/32
    type='CLIP',
    image_encoder=None,
    text_encoder=dict(
        type='CLIPTextEncoder',
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,  # also the word embedding dim
        transformer_heads=8,
        transformer_layers=12,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/clip_vitb32.pth')
    )
)

model = dict(
    type='OVDTwoStageDetector',
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        _delete_=True,
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[103.530, 116.280, 123.675],
            std=[1.0, 1.0, 1.0],
            bgr_to_rgb=False,
            pad_size_divisor=32
        ),
    ),
    rpn_head=dict(
        type='CustomRPNHead',
        anchor_generator=dict(
            scale_major=False,  # align with detectron2
        )
    ),
    backbone=dict(init_cfg=None),
    roi_head=dict(
        type='OVDStandardRoIHead',
        shared_head=dict(init_cfg=None),
        clip_cfg=clip_cfg,
        bbox_head=dict(
            type='DeticBBoxHead',
            reg_predictor_cfg=reg_layer,
            reg_class_agnostic=True,
            cls_bias=-20.0,
            cls_temp=25.0,
            cls_embeddings_path='data/metadata/coco_clip_hand_craft.npy',
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=True),
        ),
    ),
    test_cfg=dict(
        rcnn=dict(
            nms=dict(iou_threshold=0.4, type='nms'),
        )
    ),
)

train_dataloader = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
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
