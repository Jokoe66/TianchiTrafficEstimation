model = dict(
    type='Classifier',
    pretrained=None,
    bb_style='mmcls', # build backbone with mmcls or mmdet
    backbone=dict(
        type='EfficientNet',
        arch='b4',
        out_indices=(6,),
        frozen_stages=4,
        conv_cfg=dict(type='Conv2dAdaptivePadding'),
        norm_cfg=dict(type='BN', eps=1e-3)),
    neck=dict(
        type='PFFSeqNeck',
        pooling=dict(
            in_channel=1792,
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
        type='BBHead',
        head=dict(
            type='DPClsHead',
            in_channel=128,
            dropout=0,
            num_classes=4),
        loss=dict(
            type='BBNLoss',
            #np.ceil(len(training_set) / ngpu) / 4 * max_epochs
            #np.ceil(3431 / 4) / 4 * 8 = 1716
            max_steps=1716,
            criterion=dict(type='CrossEntropyLoss')),
        acc=dict(
            type='BBNAccuracy',
            max_steps=1716,
            acc=dict(type='Accuracy', tok=1))))

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

load_from = ('https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmclassification'
             '/v0/imagenet/efficientnet_b4_20200902-6e724d3d.pth')
