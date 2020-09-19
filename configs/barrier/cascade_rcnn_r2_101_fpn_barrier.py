_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    pretrained=None,
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        frozen_stages=1),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[0.1, 0.5, 1, 2])),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0))
        ])
    )

data_root = 'data/obst/'
classes = ['barrier', 'barricade']
data = dict(
    samples_per_gpu=2,
    train=dict(
        classes=classes,
        ann_file=data_root + 'annotations/barrier_barricade_coco_train.json',
        img_prefix=data_root + 'images/'),
    val=dict(
        classes=classes,
        ann_file=data_root + 'annotations/barrier_barricade_coco_val.json',
        img_prefix=data_root + 'images/'),
    test=dict(
        classes=classes,
        ann_file=data_root + 'annotations/barrier_barricade_coco_val.json',
        img_prefix=data_root + 'images/'))

optimizer = dict(type='SGD', lr=0.02 / 8 * 0.1, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[40, 44])
total_epochs = 46
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_rcnn_r2_101_fpn_20e_coco/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth'
work_dir = 'work_dirs/cascade_rcnn_r2_101_fpn_46e_barrier'
