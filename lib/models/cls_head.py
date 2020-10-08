import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models import build_head, build_loss
from mmcls.models.builder import HEADS

from .necks import Seq
from .utils import AlphaScheduler

@HEADS.register_module()
class DPClsHead(nn.Module):

    def __init__(self,
                 in_channel,
                 dropout=0,
                 num_classes=1000,
                 loss=dict(type='CrossEntropyLoss'),
                 acc=dict(type='Accuracy', topk=(1,))
                 ):
        super(DPClsHead, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.00),
            torch.nn.Linear(in_channel, num_classes))
        self.criterion = build_loss(loss)
        self.acc = build_loss(acc)
        self.num_classes = num_classes

    def forward(self, feat, **kwargs):
        logit = self.fc(feat)
        return logit

    def loss(self, preds, labels, **kwargs):
        loss = self.criterion(preds, labels)
        acc = self.acc(preds, labels)
        return {'loss_cls': loss,
                'acc_cls': acc}


@HEADS.register_module()
class DPORHead(nn.Module):

    def __init__(self,
                 in_channel,
                 dropout=0,
                 num_classes=1000,
                 loss=dict(type='BCEWithLogitsLoss'),
                 acc=dict(type='BAccuracy')
                 ):
        super(DPORHead, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.00),
            torch.nn.Linear(in_channel, num_classes - 1))
        self.criterion = build_loss(loss)
        self.acc = build_loss(acc)
        self.num_classes = num_classes

    def forward(self, feat, **kwargs):
        logit = self.fc(feat)
        if not self.training:
            logit = torch.sigmoid(logit)
            logit = torch.cat([
                logit.new_ones((len(logit), 1)),
                logit,
                logit.new_zeros((len(logit), 1))], 1)
            logit = logit[:, :-1] - logit[:, 1:]
        return logit

    def loss(self, preds, labels, **kwargs):
        targets = torch.zeros_like(preds)
        for b in range(len(labels)):
            targets[b, :labels[b]] = 1
        loss = self.criterion(preds, targets)
        acc = self.acc(preds, targets)
        return {'loss_or': loss,
                'acc_or': acc}


@HEADS.register_module()
class ClsORHead(nn.Module):

    def __init__(self,
                 cls_head,
                 or_head,
                 cls_labels,
                 or_labels,
                 **kwargs
                 ):
        super(ClsORHead, self).__init__()
        self.cls_head = build_head(cls_head)
        self.or_head = build_head(or_head)
        cls_labels.sort() # ensure the ascent order of labels
        or_labels.sort()
        self.cls_labels = cls_labels
        self.or_labels = or_labels
        self.num_classes = (
            self.cls_head.num_classes + self.or_head.num_classes - 1)
        self.label2cls_label = dict()
        self.label2or_label = dict()
        cls_label = 1
        or_label = 0
        for label in range(self.num_classes):
            if label in cls_labels:
                self.label2cls_label[label] = cls_label
                cls_label += 1
                self.label2or_label[label] = -1
            elif label in or_labels:
                self.label2or_label[label] = or_label
                or_label += 1
                self.label2cls_label[label] = 0
            else:
                raise RuntimeError(
                    f"Class {label} is lost in both cls_labels and or_labels")

    def forward(self, feat, **kwargs):
        logit = self.cls_head(feat)
        rank = self.or_head(feat)
        if not self.training:
            prob = F.softmax(logit, 1)
            rank = rank * prob[:, 0:1]
            pred = torch.zeros((len(rank), self.num_classes)).to(rank.device)
            pred[:, self.cls_labels] = prob[:, 1:]
            pred[:, self.or_labels] = rank
        else:
            pred = rank, logit
        return pred

    def loss(self, preds, labels, **kwargs):
        losses = dict()
        logit = preds[1]
        cls_labels = labels.new_tensor(
            [self.label2cls_label[label] for label in labels.tolist()])
        cls_losses = self.cls_head.loss(logit, cls_labels)
        for k, v in cls_losses.items():
            losses[f'cls_head.{k}'] = v
        rank = preds[0]
        or_labels = labels.new_tensor(
            [self.label2or_label[label] for label in labels.tolist()])
        or_mask = or_labels != -1
        if or_mask.any():
            or_losses = self.or_head.loss(rank[or_mask], or_labels[or_mask])
            for k, v in or_losses.items():
                losses[f'or_head.{k}'] = v
        return losses


@HEADS.register_module()
class LSTMDPClsHead(nn.Module):

    def __init__(self,
                 lstm=None,
                 cls_head=None,
                 **kwargs
                 ):
        super(LSTMDPClsHead, self).__init__()
        assert lstm is not None and cls_head is not None, \
            "lstm and cls_head must not be None."
        self.lstm = Seq(**lstm)
        self.cls_head = build_head(cls_head)

    def forward(self, feat, seq_len_max=5, **kwargs):
        feat = self.lstm(feat, seq_len_max)
        logit = self.cls_head(feat, **kwargs)
        return logit

    def loss(self, preds, labels, **kwargs):
        losses = self.cls_head.loss(preds, labels)
        return losses


@HEADS.register_module()
class BBHead(nn.Module):

    def __init__(self,
                 head,
                 alpha_scheduler=dict(max_steps=1716),
                 **kwargs
                 ):
        super(BBHead, self).__init__()
        self.branch1 = build_head(head)
        self.branch2 = build_head(head)
        self.scheduler = AlphaScheduler(**alpha_scheduler)

    def forward(self, feat, **kwargs):
        n, c = feat.shape
        if self.training:
            assert n % 2 == 0, "Batchsize must be divisible by 2."
            logit1 = self.branch1(feat[0::2], **kwargs)
            logit2 = self.branch2(feat[1::2], **kwargs)
            alpha = self.scheduler.alpha
            logit = alpha * logit1 + (1 - alpha) * logit2
        else:
            logit1 = self.branch1(feat, **kwargs)
            logit2 = self.branch2(feat, **kwargs)
            logit = 0.5 * logit1 + 0.5 * logit2
        return logit

    def loss(self, preds, labels, **kwargs):
        alpha = self.scheduler.alpha
        losses = dict()
        losses1 = self.branch1.loss(preds, labels[0::2])
        losses2 = self.branch2.loss(preds, labels[1::2])
        for k, v in losses1.items():
            losses[f'branch1.{k}'] = alpha * v
        for k, v in losses2.items():
            losses[f'branch2.{k}'] = (1 - alpha) * v
        self.scheduler.step()
        return losses
