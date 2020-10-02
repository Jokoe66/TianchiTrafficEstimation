import torch
import torch.nn as nn
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

    def forward(self, feat):
        logit = self.fc(feat)
        return logit


@HEADS.register_module()
class BBDPClsHead(nn.Module):

    def __init__(self,
                 in_channel,
                 dropout=0,
                 num_classes=1000
                 ):
        super(BBDPClsHead, self).__init__()
        self.branch1 = torch.nn.Sequential(
            torch.nn.Dropout(0.00),
            torch.nn.Linear(in_channel, num_classes))
        self.branch2 = torch.nn.Sequential(
            torch.nn.Dropout(0.00),
            torch.nn.Linear(in_channel, num_classes))
        self.register_buffer('alpha', torch.tensor(1.))

    def forward(self, feat):
        n, c = feat.shape
        if self.training:
            assert n % 2 == 0, "Batchsize must be divisible by 2."
            logit1 = self.branch1(feat[0::2])
            logit2 = self.branch2(feat[1::2])
            #logit = torch.stack([logit1, logit2], 1).view(n, -1)
        else:
            logit1 = self.branch1(feat)
            logit2 = self.branch2(feat)
            #logit = torch.stack([logit1, logit2], 1).view(2 * n, -1)
        logit = self.alpha * logit1 + (1 - self.alpha) * logit2
        return logit


@HEADS.register_module()
class BBLSTMDPClsHead(nn.Module):

    def __init__(self,
                 in_channel,
                 dropout=0,
                 lstm=None,
                 num_classes=1000
                 ):
        super(BBLSTMDPClsHead, self).__init__()
        self.branch1 = torch.nn.Sequential(
            Seq(in_channel, **lstm),
            torch.nn.Dropout(0.00),
            torch.nn.Linear(lstm['hidden_size'], num_classes))
        self.branch2 = torch.nn.Sequential(
            Seq(in_channel, **lstm),
            torch.nn.Dropout(0.00),
            torch.nn.Linear(lstm['hidden_size'], num_classes))
        self.register_buffer('alpha', torch.tensor(1.))

    def forward(self, feat):
        n, c = feat.shape
        if self.training:
            assert n % 2 == 0, "Batchsize must be divisible by 2."
            logit1 = self.branch1(feat[0::2])
            logit2 = self.branch2(feat[1::2])
            #logit = torch.stack([logit1, logit2], 1).view(n, -1)
        else:
            logit1 = self.branch1(feat)
            logit2 = self.branch2(feat)
            #logit = torch.stack([logit1, logit2], 1).view(2 * n, -1)
        logit = self.alpha * logit1 + (1 - self.alpha) * logit2
        return logit
