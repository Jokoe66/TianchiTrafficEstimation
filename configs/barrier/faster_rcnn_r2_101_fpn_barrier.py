_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    pretrained=None,#'open-mmlab://res2net101_v1d_26w_4s',
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
        bbox_head=dict(
            num_classes=2)))

data_root = 'data/obst/'
classes=('barrier', 'barricade')
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
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/res2net/faster_rcnn_r2_101_fpn_2x_coco/faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth'
work_dir = 'work_dirs/faster_rcnn_r2_101_fpn_46e_barrier_barricade'
