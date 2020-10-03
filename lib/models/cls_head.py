import torch
import torch.nn as nn
from mmcls.models import build_head
from mmcls.models.builder import HEADS

from .necks import Seq

@HEADS.register_module()
class DPClsHead(nn.Module):

    def __init__(self,
                 in_channel,
                 dropout=0,
                 num_classes=1000
                 ):
        super(DPClsHead, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.00),
            torch.nn.Linear(in_channel, num_classes))

    def forward(self, feat, **kwargs):
        logit = self.fc(feat)
        return logit


@HEADS.register_module()
class LSTMDPClsHead(nn.Module):

    def __init__(self,
                 lstm=None,
                 cls_head=None,
                 ):
        super(LSTMDPClsHead, self).__init__()
        assert lstm is not None and cls_head is not None, \
            "lstm and cls_head must not be None."
        self.lstm = Seq(**lstm)
        self.cls_head = build_head(cls_head)

    def forward(self, feat, seq_len=5, **kwargs):
        feat = self.lstm(feat, seq_len)
        logit = self.cls_head(feat, **kwargs)
        return logit


@HEADS.register_module()
class BBHead(nn.Module):

    def __init__(self, head):
        super(BBHead, self).__init__()
        self.branch1 = build_head(head)
        self.branch2 = build_head(head)
        self.register_buffer('alpha', torch.tensor(1.))

    def forward(self, feat, **kwargs):
        n, c = feat.shape
        if self.training:
            assert n % 2 == 0, "Batchsize must be divisible by 2."
            logit1 = self.branch1(feat[0::2], **kwargs)
            logit2 = self.branch2(feat[1::2], **kwargs)
            #logit = torch.stack([logit1, logit2], 1).view(n, -1)
            logit = self.alpha * logit1 + (1 - self.alpha) * logit2
        else:
            logit1 = self.branch1(feat, **kwargs)
            logit2 = self.branch2(feat, **kwargs)
            #logit = torch.stack([logit1, logit2], 1).view(2 * n, -1)
            logit = 0.5 * logit1 + 0.5 * logit2
        return logit
