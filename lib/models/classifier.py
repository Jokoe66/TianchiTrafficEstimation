import torch
import torch.nn.functional as F
import mmcv
from mmcv.runner import get_dist_info
from mmcls.models.builder import CLASSIFIERS
from mmcls.models import (build_backbone as build_backbone_mmcls, build_neck,
                          build_head)
from mmdet.models import build_backbone as build_backbone_mmdet

_BACKBONE_BUILDER = {
    'mmdet': build_backbone_mmdet,
    'mmcls': build_backbone_mmcls,
    }

@CLASSIFIERS.register_module()
class Classifier(torch.nn.Module):

    def __init__(self,
                 backbone,
                 pretrained=None,
                 neck=None,
                 head=None,
                 **kwargs):
        super(Classifier, self).__init__()
        bb_style = kwargs.get('bb_style', 'mmcls')
        self.backbone = _BACKBONE_BUILDER[bb_style](backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained)

    def extract_feat(self, input, seq_len=5, **kwargs):
        feat = self.backbone(input)
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        feat = self.neck(feat, **kwargs)
        return feat

    def forward(self, input, **kwargs):
        feat = self.extract_feat(input, **kwargs)
        logit = self.head(feat)
        return logit 
