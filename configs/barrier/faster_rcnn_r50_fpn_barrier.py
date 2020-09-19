# tailored anchor
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        frozen_stages=1),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[0.1, 0.3, 0.5, 1])),
    roi_head=dict(
        bbox_head=dict(
            num_classes=2)))
'''
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='Expand',
         mean=img_norm_cfg['mean'],
         to_rgb=img_norm_cfg['to_rgb'],
         ratio_range=(1, 2)),
    dict(type='RandomCrop', crop_size=(720, 1280)),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
'''
data_root = 'data/obst/'
classes=('barrier', 'barricade')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
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

optimizer = dict(type='SGD', lr=0.02 / 8 * 3 * 0.1, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[40, 44])
total_epochs = 46
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
work_dir = 'work_dirs/faster_rcnn_r50_fpn_ta_46e_barrier_barricade'
