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
        type='DPORHead',
        in_channel=128,
        dropout=0,
        num_classes=4,
        loss=dict(type='BCEWithLogitsLoss'),
        acc=dict(type='BAccuracy')))

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    #dict(type='LoadImagesFromFile'),
    #dict(type='SeqRandomResizedCrop',
    #    size=(360, 640), scale=(0.5, 1), ratio=(1.5, 2)),
    # TODO: Crop feat_mask
    dict(type='SeqResize', size=(360, 640)),
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
    dict(type='SeqResize', size=(360, 640)),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='PadSeq', seq_len_max=5, pad_value=0,
         keys=['imgs', 'feat_vector', 'feat_mask']),
    dict(type='ImagesToTensor', keys=['imgs']),
    dict(type='StackSeq', keys=['imgs', 'feat_vector', 'feat_mask']),
    dict(type='Collect',
         keys=['imgs', 'feat_vector', 'feat_mask', 'labels', 'seq_len'])
    ]
load_from = None
