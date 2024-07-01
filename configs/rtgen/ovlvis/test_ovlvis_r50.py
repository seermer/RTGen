_base_ = 'mmdet::_base_/default_runtime.py'
dataset_type = 'LVISV1Dataset'

num_classes = 1203

# 'data/metadata/lvis_v1_clip_a+cname.npy' is pre-computed
# CLIP embeddings for each category
cls_layer = dict(
    type='ZeroShotClassifier',
    zs_weight_path='data/metadata/lvis_v1_clip_a+cname.npy',
    zs_weight_dim=512,
    use_bias=0.0,
    norm_weight=True,
    norm_temperature=50.0)
reg_layer = [
    dict(type='Linear', in_features=1024, out_features=1024),
    dict(type='ReLU', inplace=True),
    dict(type='Linear', in_features=1024, out_features=4)
]

find_unused_parameters = True

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
    type='Detic',
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32),
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    rpn_head=dict(
        type='CenterNetRPNHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        conv_bias=True,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_cls=dict(
            type='HeatmapFocalLoss',
            alpha=0.25,
            beta=4.0,
            gamma=2.0,
            pos_weight=0.5,
            neg_weight=0.5,
            loss_weight=1.0,
            ignore_high_fp=0.85,
        ),
        loss_bbox=dict(type='GIoULoss', eps=1e-6, loss_weight=1.0),
    ),
    roi_head=dict(
        type='DeticRoIHead',
        clip_cfg=clip_cfg,
        num_stages=3,
        stage_loss_weights=[1.0, 1.0, 1.0],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=0,
                use_torchvision=True),
            out_channels=256,
            featmap_strides=[8, 16, 32],
            # approximately equal to
            # canonical_box_size=224, canonical_level=4 in D2
            finest_scale=112),
        bbox_head=[
            dict(
                type='OriginalDeticBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                cls_predictor_cfg=cls_layer,
                reg_predictor_cfg=reg_layer,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=0.1,
                               loss_weight=1.0)),
            dict(
                type='OriginalDeticBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                cls_predictor_cfg=cls_layer,
                reg_predictor_cfg=reg_layer,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=0.1,
                               loss_weight=1.0)),
            dict(
                type='OriginalDeticBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                cls_predictor_cfg=cls_layer,
                reg_predictor_cfg=reg_layer,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=0.1, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32],
            # approximately equal to
            # canonical_box_size=224, canonical_level=4 in D2
            finest_scale=112),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            class_agnostic=True,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    test_cfg=dict(
        rpn=dict(
            score_thr=0.0001,
            nms_pre=1000,
            max_per_img=256,
            nms=dict(type='nms', iou_threshold=0.9),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.02,
            nms=dict(type='nms', iou_threshold=0.4),
            max_per_img=300,
            mask_thr_binary=0.5)))

val_pipeline = test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args,
        imdecode_backend=backend_args),
    dict(
        type='Resize',
        scale=(1333, 800),
        keep_ratio=True,
        backend=backend_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

test_dataloader = val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LVISV1Dataset',
        data_root='data/lvis_v1/',
        ann_file='annotations/lvis_v1_val.json',
        data_prefix=dict(img=''),
        pipeline=val_pipeline,
        return_classes=False))

test_evaluator = val_evaluator = dict(
    type='LVISMetric',
    ann_file='data/lvis_v1/annotations/lvis_v1_val.json',
    metric=['bbox', 'segm'])

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


