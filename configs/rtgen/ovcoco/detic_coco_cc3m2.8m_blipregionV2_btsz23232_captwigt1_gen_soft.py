_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50-caffe-c4.py',
    '../../_base_/datasets/coco_ovd_detic_cc3m2.8m_gen_btsz23232_v2.py',
    '../../_base_/schedules/schedule_45k.py',
    '../../_base_/iter_based_runtime.py'
]
class_weight = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 0, 1, 1, 0, 1, 0, 0, 1,
                0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                1, 0, 1, 1, 1, 1, 0, 0, 0, 1] + [0]

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
    batch2ovd=dict(generated_caption_batch=['generated_caption', 'generated_caption_region'],
                   generated_image_batch=['generated_image_region']),
    roi_head=dict(
        type='OVDStandardRoIHead',
        shared_head=dict(init_cfg=None),
        clip_cfg=clip_cfg,
        ovd_cfg=dict(generated_caption=dict(type='GeneratedCaption',
                                            base_batch_size=None, # None
                                            bce_bias=-20.0, norm_temp=25.0, gen_caption_weight=1, # None, 50.0, 0.1
                                            queue_cfg=dict(type='GeneratedQueue',
                                                           length=256,
                                                           name='generated_caption'),
                                            neg_weight=0.125,       
                                            sampling_method='orig_only',
                                            name='generated_caption'),  # add sampling_method='orig_only',
                     generated_caption_region=dict(type='GeneratedRegionCaptionSoft',
                                                   base_batch_size=None,
                                                   bce_bias=-20.0, norm_temp=25.0, gen_caption_weight=1,
                                                   queue_cfg=dict(type='GeneratedQueue',
                                                                  length=768,
                                                                  name='generated_caption_region'),
                                                   neg_weight=0.125,
                                                   name='generated_caption_region_soft'),
                     generated_image_region=dict(type='GeneratedRegionCaptionSoft',
                                                   base_batch_size=None,
                                                   bce_bias=-20.0, norm_temp=25.0, gen_caption_weight=1,
                                                   queue_cfg=dict(type='GeneratedQueue',
                                                                  length=256,
                                                                  name='generated_image_region'),
                                                   neg_weight=0.125,
                                                   name='generated_image_region_soft'),
                     ),
        bbox_head=dict(
            type='DeticBBoxHead',
            reg_predictor_cfg=reg_layer,
            reg_class_agnostic=True,
            cls_bias=-20.0,
            cls_temp=25.0,
            cls_embeddings_path='data/metadata/coco_clip_hand_craft.npy',
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=True,
                class_weight=class_weight),
        ),
    ),
    test_cfg=dict(
        rcnn=dict(
            nms=dict(iou_threshold=0.4, type='nms'),
        )
    ),
)

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',  # amp training
    clip_grad=dict(max_norm=35, norm_type=2),
)
# load_from = 'work_dirs/detic_base/iter_90000.pth'
load_from = 'checkpoints/detic_coco_base.pth'
