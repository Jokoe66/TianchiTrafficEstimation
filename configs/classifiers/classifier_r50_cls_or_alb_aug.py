model = dict(
    type='Classifier',
    pretrained='torchvision://resnet50',
    bb_style='mmcls', # build backbone with mmcls or mmdet
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=2,
        style='pytorch'),
    neck=dict(
        type='PFFSeqNeck',
        pooling=dict(
            in_channel=2048,
            pool_h=9,
            pool_w=16,
            pool_c=256,
            flatten=False),
        feature_fusion=dict(
            feat_mask_dim=0,
            feat_vec_dim=0),
        lstm=dict(
            in_channel=9*16*256,
            hidden_size=128)),
    head=dict(
        type='ClsORHead',
        cls_head=dict(
            type='DPClsHead',
            in_channel=128,
            dropout=0,
            num_classes=2,
            loss=dict(type='CrossEntropyLoss'),
            acc=dict(type='Accuracy', topk=1)),
        or_head=dict(
            type='DPORHead',
            in_channel=128,
            dropout=0,
            num_classes=3,
            loss=dict(
                type='BCEWithLogitsLoss',
                reduction='none'),
            acc=dict(type='BAccuracy')),
        cls_labels=[3],
        or_labels=[0, 1, 2]))

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    #dict(type='LoadImagesFromFile'),
    dict(type='SeqNormalize',
         mean=[0,0,0], std=[255.,255.,255.], to_rgb=True),# scale to [0, 1] rgb
    dict(type='UnpackSequence', keys=['imgs', 'feat_mask'], num_fields=5),
    dict(type='Albumentation', # spatial transformation
         keymap={'imgs0': 'image'},
         additional_targets={
             'imgs1': 'image',
             'imgs2': 'image',
             'imgs3': 'image',
             'imgs4': 'image',
             'feat_mask0': 'image',
             'feat_mask1': 'image',
             'feat_mask2': 'image',
             'feat_mask3': 'image',
             'feat_mask4': 'image'
         },
         transforms=[
             dict(
                 type='Resize',
                 width=640,
                 height=360,
                 p=1),
             #dict(
             #    type='RandomResizedCrop',
             #    width=640,
             #    height=360,
             #    scale=(0.5, 1),
             #    ratio=(1.5, 2),
             #    p=1),
             dict(
                 type='HorizontalFlip',
                 p=0.5),
             dict(
                 type='ShiftScaleRotate',
                 shift_limit=0.0625,
                 scale_limit=0.0,
                 rotate_limit=0,
                 interpolation=1,
                 p=0.5)]),
    dict(type='Albumentation', # image transformation
         keymap={'imgs0': 'image'},
         additional_targets={
             'imgs1': 'image',
             'imgs2': 'image',
             'imgs3': 'image',
             'imgs4': 'image',
         },
         transforms=[
             dict(
                 type='RandomSunFlare',
                 src_radius=80,
                 p=0.5),
             dict(
                 type='RandomBrightnessContrast',
                 brightness_limit=[-0.1, 0.1],
                 contrast_limit=[-0.1, 0.1],
                 p=0.3),
             dict(type='ChannelShuffle', p=0.2),
             dict(
                 type='OneOf',
                 transforms=[
                     dict(type='Blur', blur_limit=3, p=1.0),
                     dict(type='MedianBlur', blur_limit=3, p=1.0)
                 ],
                 p=0.5),
             dict(type='Normalize', max_pixel_value=1, p=1), # normalize
         ]),
    dict(type='PackSequence', keys=['imgs', 'feat_mask']),
    dict(type='PadSeq', seq_len_max=5, pad_value=0,
         keys=['imgs', 'feat_vector', 'feat_mask']),
    dict(type='ImagesToTensor', keys=['imgs']),
    dict(type='StackSeq', keys=['imgs', 'feat_vector', 'feat_mask']),
    dict(type='Collect',
         keys=['imgs', 'labels', 'seq_len'])
    ]
test_pipeline = [
    #dict(type='LoadImagesFromFile'),
    #dict(type='AssignImgFields', keys=['imgs', 'feat_mask']),
    dict(type='SeqResize', size=(360, 640)),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='PadSeq', seq_len_max=5, pad_value=0,
         keys=['imgs', 'feat_vector', 'feat_mask']),
    dict(type='ImagesToTensor', keys=['imgs']),
    dict(type='StackSeq', keys=['imgs', 'feat_vector', 'feat_mask']),
    dict(type='Collect',
         keys=['imgs', 'labels', 'seq_len'])
    ]
load_from = None
