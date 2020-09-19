_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        _delete_=True,
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    neck=dict(
        _delete_=True,
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256),
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
    step=[56])
total_epochs = 64
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_2x_coco/faster_rcnn_hrnetv2p_w32_2x_coco_20200529_015927-976a9c15.pth'
work_dir = 'work_dirs/faster_rcnn_hrnetv2p_w32p_64e_barrier_barricade'
