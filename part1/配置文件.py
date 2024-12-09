# mmdet3d and other OpenMMLab projects use MMEngine's configration system.

# MMDetection3D 采用模块化设计，所有的模块都可以通过配置文件来配置。

# 这里以 PointPillars 算法为例，展示如何配置模型。

# point pillars 是一种基于点云的 3D 目标检测算法，其主要思路是将 3D 点云划分为多个体素，然后在每个体素中进行 3D 目标检测。


# 使用 model 字段来配置检测算法的组件
# 需要 voxel_encoder, backbone 等神经网络组件
# 还需要 data_preprocessor, train_cfg, test_cfg 等配置



# 模型配置

model = dict(
    type='VoxelNet',  # 使用 VoxelNet 算法，这里还有很多模型可供选择

    # 数据预处理器
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            voxel_size=[0.16, 0.16, 4],
            max_voxels=(16000, 40000))),

    # 体素编码器
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),

    # 中间编码器
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),

    # 特征金字塔网络
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),

    # 边界框头
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=0.1111111111111111,
            loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),

    # 训练配置: IoU, 交并比
    train_cfg=dict(
        assigner=[
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),

            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),

            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1)
        ],

        allowed_border=0,
        pos_weight=-1,
        debug=False),

    # 测试配置
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

# 数据集和评测器的配置
# 在使用执行器（Runner）进行训练、测试和验证时，我们需要配置数据加载器。
# 构建数据加载器需要设置数据集和数据处理流程。由于这部分的配置较为复杂，我们使用中间变量来简化数据加载器配置的编写。


# 数据集基本信息
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'  # 数据集的根目录
class_names = ['Pedestrian', 'Cyclist', 'Car']

point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]  # 定义点云的空间范围（x、y、z的最小和最大值），用于过滤点云。

input_modality = dict(use_lidar=True, use_camera=False)  # 输入的模态，这里只使用了点云数据。
metainfo = dict(classes=class_names)


# 数据库采样器配置
# 数据库采样器用于从数据集中采样出一部分样本用于训练。
# 这里我们使用数据库采样器来实现类均衡采样。
db_sampler = dict(
    data_root=data_root,  # 即 '/data/kitti/'
    info_path=data_root + 'kitti_dbinfos_train.pkl',  # 指向包含增强信息的文件路径（如目标的位置信息）
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4))

# 数据处理流程配置
# 数据处理流程用于对输入数据进行预处理，包括数据增强、数据转换等。
# 这里我们使用 mmdet3d 中的数据处理流程。
train_pipeline = [
    # 首先，我们使用 LoadPointsFromFile 类来加载点云数据。
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),

    # 然后，我们使用 LoadAnnotations3D 类来加载标注数据。（边界框标注）
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),

    # 采样
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),

    # 水平翻转
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),

    # 随机旋转、缩放和平移
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),

    # 保留在点云范围内的点和目标
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),

    # 打乱点云顺序
    dict(type='PointShuffle'),

    # 数据打包
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]

# 测试和评估数据处理流程配置

# 加载 -> 数据增强 -> 点云范围过滤 -> 数据打包
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points'])
]



# 数据加载器
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR')))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))

# 评估器配置
# 评估器用于对模型的预测结果进行评估。
# 这里我们使用 mmdet3d 中的评估器。

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator

# 训练和测试配置

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器配置
optim_wrapper = dict(  # 优化器封装配置
    type='OptimWrapper',  # 优化器封装类型，切换到 AmpOptimWrapper 启动混合精度训练
    optimizer=dict(  # 优化器配置。支持 PyTorch 的各种优化器，
        # 请参考 https://pytorch.org/docs/stable/optim.html#algorithms
        type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))  # 梯度裁剪选项。设置为 None 禁用梯度裁剪。
# 使用方法请见 https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html

# param_scheduler 是配置调整优化器超参数（例如学习率和动量）的字段。
# 用户可以组合多个调度器来创建所需要的参数调整策略。

param_scheduler = [
    # cosine 学习率衰减
    dict(
        type='CosineAnnealingLR',
        T_max=32,
        eta_min=0.01,
        begin=0,
        end=32,
        by_epoch=True,
        convert_to_iter_based=True),

    # cosine 动量衰减
    dict(
        type='CosineAnnealingLR',
        T_max=48,
        eta_min=1.0000000000000001e-07,
        begin=32,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True),

    # 余弦退火学习率衰减
    dict(
        type='CosineAnnealingMomentum',
        T_max=32,
        eta_min=0.8947368421052632,
        begin=0,
        end=32,
        by_epoch=True,
        convert_to_iter_based=True),

    # 余弦退火动量衰减
    dict(
        type='CosineAnnealingMomentum',
        T_max=48,
        eta_min=1,
        begin=32,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hook 配置
# hook 是一些函数，在特定事件发生时被调用。
# 这里我们使用 mmdet3d 中的 hook。

"""
用户可以在训练、验证和测试循环上添加钩子，从而在运行期间插入一些操作。
有两种不同的钩子字段，一种是 default_hooks，另一种是 custom_hooks。
default_hooks 是一个钩子配置字典，并且这些钩子是运行时所需要的。
它们具有默认优先级，是不需要修改的。如果未设置，执行器将使用默认值。
如果要禁用默认钩子，用户可以将其配置设置为 None。
"""

default_hooks = dict(
    # how much time the training process takes
    timer=dict(type='IterTimerHook'),
    # print the log information every 50 iterations
    logger=dict(type='LoggerHook', interval=50),
    # schedule the learning rate and momentum
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save the checkpoints (when given the interval as -1, we disable it)
    # saving the checkpoints means saving the model and optimizer states
    checkpoint=dict(type='CheckpointHook', interval=-1),
    # when there are multiple gpus, we need to synchronize the data
    # this makes sure that in each one of the gpus, the sampling of the data are independent and stable.
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualize the results, like the predicted and ground truth 3D bounding boxes and point clouds
    visualization=dict(type='Det3DVisualizationHook'))

custom_hooks = []  # 自定义钩子配置，可以自己开发。






# 运行器配置

default_scope = 'mmdet3d'  # 寻找模块的默认注册器域。
# 请参考 https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/registry.html

env_cfg = dict(
    cudnn_benchmark=False,  # 是否启用 cudnn benchmark

    mp_cfg=dict(  # 多进程配置
        mp_start_method='fork',  # 使用 fork 来启动多进程。'fork' 通常比 'spawn' 更快，但可能不安全。请参考 https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0),  # 关闭 opencv 的多进程以避免系统超负荷
    dist_cfg=dict(backend='nccl'))  # 分布式配置

vis_backends = [dict(type='LocalVisBackend')]  # 可视化后端。
# 请参考 https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html

visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(
    type='LogProcessor',  # 日志处理器用于处理运行时日志
    window_size=50,  # 日志数值的平滑窗口
    by_epoch=True)  # 是否使用 epoch 格式的日志。需要与训练循环的类型保持一致

log_level = 'INFO'  # 日志等级
load_from = None  # 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
resume = False  # 是否从 `load_from` 中定义的检查点恢复。如果 `load_from` 为 None，它将恢复 `work_dir` 中的最近检查点。



