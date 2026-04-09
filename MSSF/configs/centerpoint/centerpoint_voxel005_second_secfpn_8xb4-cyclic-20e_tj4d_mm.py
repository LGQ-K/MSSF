_base_ = ['../_base_/default_runtime.py']

# ================= 1. 全局基础参数 =================
voxel_size = [0.05, 0.05, 0.125]
point_cloud_range = [0, -40, -4, 70.4, 40, 2]
class_names = ['Pedestrian', 'Cyclist', 'Car', 'Truck']
dataset_type = 'TJ4DDataset'
data_root = 'data/tj4d/'

# ================= 2. 模型结构定义 =================
model = dict(
    type='CenterPointFuse',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            max_voxels=(16000, 40000),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size),
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/htc_r50_fpn_coco-20e_20e_nuim_20201008_211415-d6c60a2c.pth',
            prefix='backbone.')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/htc_r50_fpn_coco-20e_20e_nuim_20201008_211415-d6c60a2c.pth',
            prefix='neck.')),
    pts_middle_encoder=dict(
        type='SparseEncoderFuse',
        in_channels=5,
        sparse_shape=[49, 1600, 1408],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((32, 32), (64, 64, 64), (128, 128, 128), (128, 128, 128)),
        encoder_paddings=((0, 0), (1, 0, 0), (1, 0, 0), ([0, 1, 1], 0, 0)),
        block_type='basicblock',
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        legacy=True),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=512,
        # 类别不均衡补偿：强化弱类训练梯度
        task_loss_weights=[1.35, 1.25, 1.0, 1.15],
        # 按任务设置后处理阈值，降低行人/骑行漏检
        task_score_thresholds=[0.004, 0.005, 0.01, 0.006],
        task_nms_thr=[0.15, 0.15, 0.2, 0.18],
        # 仅增强小目标（Ped/Cyc）分数，Car/Truck 保持 1.0 以稳住大目标
        task_score_gamma=[0.85, 0.9, 1.0, 1.0],
        # 尺寸守护：仅当预测框 max(dx,dy) <= 阈值时才进行分数增强
        task_small_object_max_dim=[1.3, 2.2, 100.0, 100.0],
        # 小目标二阶段NMS补偿，仅对前两类生效
        task_small_nms_thr=[0.12, 0.12, 0.2, 0.18],
        task_small_post_max_size=[180, 180, 83, 83],
        # 阈值过滤后的最小候选保留数，仅强化小目标任务
        task_min_keep_candidates=[220, 220, 0, 0],
        task_pre_max_size=[1400, 1400, 1000, 1000],
        task_post_max_size=[120, 120, 83, 83],
        # 按任务设置目标分配半径与高斯重叠，提升小目标热图学习质量
        task_min_radius=[4, 3, 2, 2],
        task_gaussian_overlap=[0.08, 0.09, 0.1, 0.1],
        # ================= 核心架构升级：任务头解耦 =================
        # 拆解为 4 个独立头，彻底防止大目标的梯度“吞噬”小目标特征
        tasks=[
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Truck'])
        ],
        # ============================================================
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=256,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=point_cloud_range,
            max_num=500,
            task_max_num=[700, 700, 500, 600],
            score_threshold=0.01,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            grid_size=[1408, 1600, 48],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=point_cloud_range,
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.005,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2))
)

# ================= 3. 数据处理流水线 (无 GT-Paste) =================
use_dim = [0, 1, 2, 7, 5]

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=use_dim, backend_args=None),
    dict(type='LoadImageFromFile', color_type='unchanged', to_float32=False, backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'img', 'gt_labels_3d', 'gt_bboxes_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=use_dim, backend_args=None),
    dict(type='LoadImageFromFile', color_type='unchanged', to_float32=False, backend_args=None),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='GlobalRotScaleTrans', rot_range=[0, 0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

# ================= 4. 数据集配置 (引入 CBGS 进行全生命周期护航) =================
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='tj4d_infos_train.pkl',
            modality=dict(use_camera=True, use_lidar=True),
            metainfo=dict(classes=class_names),
            data_prefix=dict(
                img='training/image_2',
                pts='training/velodyne_reduced'),
            pipeline=train_pipeline)
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='tj4d_infos_val.pkl',
        modality=dict(use_camera=True, use_lidar=True),
        metainfo=dict(classes=class_names),
        data_prefix=dict(
            img='training/image_2',
            pts='training/velodyne_reduced'),
        pipeline=test_pipeline,
        test_mode=True)
)
test_dataloader = val_dataloader
val_evaluator = dict(type='TJ4DMetric', ann_file=data_root + 'tj4d_infos_val.pkl', metric='bbox', backend_args=None)
test_evaluator = val_evaluator

# ================= 5. 优化器与学习率策略 (执行完整的 20 轮全新训练) =================
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0008, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = [
    dict(
        type='OneCycleLR',
        total_steps=40,  # 与 max_epochs 保持一致
        by_epoch=True,
        eta_max=0.0008,
        pct_start=0.25,
        div_factor=10.0,
        final_div_factor=10000.0)
]

# ================= 6. 杂项与日志配置 =================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='Kitti metric/pred_instances_3d/KITTI/Overall_3D_moderate',
        rule='greater'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# 初始化网络参数，抛弃之前的所有权重
load_from = None
resume = False