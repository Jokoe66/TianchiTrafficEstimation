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
            feat_mask_dim=2,
            feat_vec_dim=13),
        lstm=dict(
            in_channel=9*16*258+13,
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
    dict(type='AssignImgFields', keys=['imgs', 'feat_mask']),
    dict(type='SeqRandomResizedCrop',
        size=(360, 640), scale=(0.6, 1), ratio=(1.5, 2)),
    #dict(type='SeqResize', size=(360, 640)),
    dict(type='SeqRandomFlip', flip_prob=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='PadSeq', seq_len_max=5, pad_value=0,
         keys=['imgs', 'feat_vector', 'feat_mask']),
    dict(type='ImagesToTensor', keys=['imgs']),
    dict(type='StackSeq', keys=['imgs', 'feat_vector', 'feat_mask']),
    dict(type='Collect',
         keys=['imgs', 'feat_vector', 'feat_mask', 'labels', 'seq_len'])
    ]
test_pipeline = [
    #dict(type='LoadImagesFromFile'),
    dict(type='AssignImgFields', keys=['imgs', 'feat_mask'],
        extra_aug_fields=['feat_mask']),
    #dict(type='SeqResize', size=(360, 640)),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqRandomFlip', flip_prob=1., extend=True), # flip and store in imgs_flipped field
    dict(type='PadSeq', seq_len_max=5, pad_value=0,
         keys=['imgs', 'imgs_flipped', 'feat_vector', 'feat_mask',
               'feat_mask_flipped']),
    dict(type='MultiScaleCrop', size=(360, 640), scales=(0.8, ),
         num_fixed_crops=5, crop_all=True), # crop 5 regions (left upper to right bottom)
    dict(type='ImagesToTensor', keys=[f'imgs{i}' for i in range(5)]), # 5 cropped sequences
    dict(type='ImagesToTensor', keys=[f'imgs_flipped{i}' for i in range(5)]),
    dict(type='StackSeq', keys=['feat_vector']), # stack list of tensors to a tensor
    dict(type='StackSeq', keys=[f'imgs{i}' for i in range(5)]),
    dict(type='StackSeq', keys=[f'imgs_flipped{i}' for i in range(5)]),
    dict(type='StackSeq', keys=[f'feat_mask{i}' for i in range(5)]),
    dict(type='StackSeq', keys=[f'feat_mask_flipped{i}' for i in range(5)]),
    dict(type='PackSequence', keys=['imgs', 'feat_mask']), # Pack sequence named with same prefix
    dict(type='Collect',
         keys=['imgs', 'feat_vector', 'feat_mask', 'labels', 'seq_len',
               'extra_aug_fields'])
    ]
load_from = None
