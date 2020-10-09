from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.distributed as dist
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

    def forward(self, *args, **kwargs):
        if self.training:
            outputs = self.forward_train(*args, **kwargs)
        else:
            outputs = self.forward_test(*args, **kwargs)
        return outputs

    def forward_train(self, input, **kwargs):
        feat = self.extract_feat(input, **kwargs)
        logit = self.head(feat, **kwargs)
        losses = dict()
        loss_head = self.head.loss(logit, **kwargs)
        losses.update(loss_head)
        return losses

    def forward_test(self, input, **kwargs):
        if isinstance(input, torch.Tensor):
            outputs = self.simple_test(input, **kwargs)
        elif isinstance(input, list) and isinstance(input[0], torch.Tensor):
            outputs = self.aug_test(input, **kwargs)
        else:
            raise TypeError(
                "Type of input must be torch.Tensor or list of Tensor)")
        return outputs

    def simple_test(self, input, **kwargs):
        feat = self.extract_feat(input, **kwargs)
        logit = self.head(feat, **kwargs)
        return logit

    def aug_test(self, inputs, **kwargs):
        preds = []
        single_kwargs = kwargs.copy()
        for i in range(len(inputs)):
            extra_fields = kwargs.get('extra_fields', [])
            for field in extra_fields:
                single_kwargs[field] = kwargs[field][i]
            pred = self.simple_test(inputs[i], **single_kwargs)
            preds.append(pred)
        merged_pred = torch.stack(preds).mean(0)
        return merged_pred

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


@CLASSIFIERS.register_module()
class MixupClassifier(Classifier):

    def forward(self, input, **kwargs):
        if self.training:
            alpha = self.head.criterion.scheduler.alpha
            input = alpha * input[0::2] + (1 - alpha) * input[1::2]
            feat = self.extract_feat(input, **kwargs)
            logit = self.head(feat, **kwargs)
            losses = dict()
            loss_head = self.head.loss(logit, **kwargs)
            losses.update(loss_head)
            return losses
        else:
            feat = self.extract_feat(input, **kwargs)
            logit = self.head(feat, **kwargs)
            return logit
