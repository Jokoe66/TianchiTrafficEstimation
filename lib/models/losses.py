import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import LOSSES
from mmcls.models import build_loss
from mmcls.models.losses import Accuracy

from .utils import AlphaScheduler

LOSSES.register_module()(Accuracy)
LOSSES.register_module()(nn.BCEWithLogitsLoss)


@LOSSES.register_module()
class BAccuracy(nn.Module):

    def forward(self, preds, labels):
        preds = torch.sigmoid(preds)
        acc = (((preds >= 0.5) == labels) * 1.).mean()
        return acc


@LOSSES.register_module()
class ORLoss(nn.Module):
    """ Ordinal regression loss.
    """

    def __init__(self):
        super(ORLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, labels):
        targets = torch.zeros_like(preds)
        for b in range(len(labels)):
            targets[b, :labels[b]] = 1
        return self.bce(preds, targets)


@LOSSES.register_module()
class ClsORLoss(nn.Module):
    """ Classification and Ordinal regression loss.
    """

    def __init__(self):
        super(ClsORLoss, self).__init__()
        self.bce = F.binary_cross_entropy_with_logits

    def forward(self, preds, labels):
        rank = preds[:, :-1]
        logit = preds[:, -1]
        cls_labels = (labels != preds.shape[1]).type(torch.float32)
        cls_loss = self.bce(logit, cls_labels)
        rank_targets = torch.zeros_like(rank)
        for b in range(len(labels)):
            rank_targets[b, :labels[b]] = 1
        rank_loss = self.bce(rank, rank_targets, reduction='none')
        rank_loss = (rank_loss * cls_labels.unsqueeze(1).type(torch.float32))
        rank_loss = rank_loss.mean()
        loss = cls_loss + rank_loss
        return loss


@LOSSES.register_module()
class SORDLoss(nn.Module):
    """ Soft Ordinal Label loss.
    """

    def __init__(self):
        super(SORDLoss, self).__init__()
        self.kl_div = nn.KLDivLoss()

    def forward(self, preds, labels):
        soft_labels = torch.zeros_like(preds)
        for cls in range(preds.shape[1]):
            soft_labels[:, cls] = torch.exp(
                -1.8 * (labels.type_as(preds) - cls) ** 2)
        soft_labels /= soft_labels.sum(1, keepdim=True)
        return self.kl_div(preds, soft_labels)


@LOSSES.register_module()
class BBNLoss(nn.Module):

    def __init__(self, alpha_scheduler, criterion, **kwargs):
        super(BBNLoss, self).__init__()
        self.scheduler = AlphaScheduler(**alpha_scheduler)
        self.criterion = build_loss(criterion)

    def forward(self, preds, labels):
        labels1 = labels[0::2]
        labels2 = labels[1::2]
        alpha = self.scheduler.alpha
        loss = (alpha * self.criterion(preds, labels1)
                + (1 - alpha) * self.criterion(preds, labels2))
        self.scheduler.step()
        return loss


@LOSSES.register_module()
class BBNAccuracy(nn.Module):

    def __init__(self, alpha_scheduler, acc, **kwargs):
        super(BBNAccuracy, self).__init__()
        self.scheduler = AlphaScheduler(**alpha_scheduler)
        self.acc = build_loss(acc)

    def forward(self, preds, labels):
        labels1 = labels[0::2]
        labels2 = labels[1::2]
        alpha = self.scheduler.alpha
        acc = (alpha * self.acc(preds, labels1)
                + (1 - alpha) * self.acc(preds, labels2))
        self.scheduler.step()
        return acc
