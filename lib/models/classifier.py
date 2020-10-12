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

    def forward(self, imgs, **kwargs):
        # format sequence (N, C, H, W, T) to (NxT, C, H, W)
        if not isinstance(imgs, list):
            imgs = [imgs]
        if len(imgs[0].shape) > 4: # frames
            seq_len_max = imgs[0].shape[-1]
            for i, img in enumerate(imgs):
                imgs[i] = (img.permute(4, 0, 1, 2, 3).contiguous()
                    .view(-1, *img.shape[1:4])
                    .to(next(self.parameters()).device))
        else:
            imgs[0] = imgs[0].to(next(self.parameters()).device)
            seq_len_max = 1
        kwargs['seq_len_max'] = seq_len_max
        if self.training:
            outputs = self.forward_train(imgs[0], **kwargs)
        else:
            outputs = self.forward_test(imgs, **kwargs)
        return outputs

    def forward_train(self, input, **kwargs):
        feat = self.extract_feat(input, **kwargs)
        logit = self.head(feat, **kwargs)
        losses = dict()
        loss_head = self.head.loss(logit, **kwargs)
        losses.update(loss_head)
        return losses

    def simple_test(self, input, **kwargs):
        feat = self.extract_feat(input, **kwargs)
        if kwargs.get('return_features', False):
            return feat
        logit = self.head(feat, **kwargs)
        return logit

    def forward_test(self, inputs, **kwargs):
        preds = []
        single_kwargs = kwargs.copy()
        extra_fields = kwargs.get('extra_aug_fields', [])
        for i in range(len(inputs)):
            for field in extra_fields:
                field = field[0] # squeeze batch dim
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
