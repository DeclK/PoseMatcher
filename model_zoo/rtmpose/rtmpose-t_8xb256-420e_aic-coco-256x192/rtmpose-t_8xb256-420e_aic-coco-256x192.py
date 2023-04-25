default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False))
custom_hooks = [
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=390,
        switch_pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomHalfBody'),
            dict(
                type='RandomBBoxTransform',
                shift_factor=0.0,
                scale_factor=[0.75, 1.25],
                rotate_factor=60),
            dict(type='TopdownAffine', input_size=(192, 256)),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                type='Albumentation',
                transforms=[
                    dict(type='Blur', p=0.1),
                    dict(type='MedianBlur', p=0.1),
                    dict(
                        type='CoarseDropout',
                        max_holes=1,
                        max_height=0.4,
                        max_width=0.4,
                        min_holes=1,
                        min_height=0.2,
                        min_width=0.2,
                        p=0.5)
                ]),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='SimCCLabel',
                    input_size=(192, 256),
                    sigma=(4.9, 5.66),
                    simcc_split_ratio=2.0,
                    normalize=False,
                    use_dark=False)),
            dict(type='PackPoseInputs')
        ])
]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False
backend_args = dict(backend='local')
train_cfg = dict(by_epoch=True, max_epochs=420, val_interval=10)
val_cfg = dict()
test_cfg = dict()
max_epochs = 420
stage2_num_epochs = 30
base_lr = 0.004
randomness = dict(seed=21)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.0),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=210,
        end=420,
        T_max=210,
        by_epoch=True,
        convert_to_iter_based=True)
]
auto_scale_lr = dict(base_batch_size=1024)
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=
            'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-tiny_udp-aic-coco_210e-256x192-cbed682d_20230130.pth'
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=384,
        out_channels=17,
        input_size=(192, 256),
        in_featuremap_size=(6, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.0,
            label_softmax=True),
        decoder=dict(
            type='SimCCLabel',
            input_size=(192, 256),
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    test_cfg=dict(flip_test=True))
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/'
train_pipeline = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0)
        ]),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='SimCCLabel',
            input_size=(192, 256),
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='PackPoseInputs')
]
train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.0,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5)
        ]),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='SimCCLabel',
            input_size=(192, 256),
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    dict(type='PackPoseInputs')
]
dataset_coco = dict(
    type='RepeatDataset',
    dataset=dict(
        type='CocoDataset',
        data_root='data/',
        data_mode='topdown',
        ann_file='coco/annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='detection/coco/train2017/'),
        pipeline=[]),
    times=3)
dataset_aic = dict(
    type='AicDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='aic/annotations/aic_train.json',
    data_prefix=dict(
        img=
        'pose/ai_challenge/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'
    ),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(0, 6), (1, 8), (2, 10), (3, 5), (4, 7), (5, 9), (6, 12),
                     (7, 14), (8, 16), (9, 11), (10, 13), (11, 15)])
    ])
train_dataloader = dict(
    batch_size=256,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[
            dict(
                type='RepeatDataset',
                dataset=dict(
                    type='CocoDataset',
                    data_root='data/',
                    data_mode='topdown',
                    ann_file='coco/annotations/person_keypoints_train2017.json',
                    data_prefix=dict(img='detection/coco/train2017/'),
                    pipeline=[]),
                times=3),
            dict(
                type='AicDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='aic/annotations/aic_train.json',
                data_prefix=dict(
                    img=
                    'pose/ai_challenge/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'
                ),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(0, 6), (1, 8), (2, 10), (3, 5), (4, 7),
                                 (5, 9), (6, 12), (7, 14), (8, 16), (9, 11),
                                 (10, 13), (11, 15)])
                ])
        ],
        pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomHalfBody'),
            dict(
                type='RandomBBoxTransform',
                scale_factor=[0.6, 1.4],
                rotate_factor=80),
            dict(type='TopdownAffine', input_size=(192, 256)),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                type='Albumentation',
                transforms=[
                    dict(type='Blur', p=0.1),
                    dict(type='MedianBlur', p=0.1),
                    dict(
                        type='CoarseDropout',
                        max_holes=1,
                        max_height=0.4,
                        max_width=0.4,
                        min_holes=1,
                        min_height=0.2,
                        min_width=0.2,
                        p=1.0)
                ]),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='SimCCLabel',
                    input_size=(192, 256),
                    sigma=(4.9, 5.66),
                    simcc_split_ratio=2.0,
                    normalize=False,
                    use_dark=False)),
            dict(type='PackPoseInputs')
        ],
        test_mode=False))
val_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/',
        data_mode='topdown',
        ann_file='coco/annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='detection/coco/val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(192, 256)),
            dict(type='PackPoseInputs')
        ]))
test_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/',
        data_mode='topdown',
        ann_file='coco/annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='detection/coco/val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(192, 256)),
            dict(type='PackPoseInputs')
        ]))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/person_keypoints_val2017.json')
test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/person_keypoints_val2017.json')
