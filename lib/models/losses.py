import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import LOSSES
from mmcls.models import build_loss
from mmcls.models.losses import Accuracy, weight_reduce_loss

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

    def __init__(self, beta, weight=None, reduction='mean', avg_factor=None):
        super(SORDLoss, self).__init__()
        self.beta = beta
        if weight is not None:
            weight = weight.float()
        self.weight = weight
        self.reduction = reduction
        self.avg_factor = avg_factor

    def forward(self, preds, labels):
        soft_labels = -self.beta * (torch.arange(preds.shape[1])
                                   .expand_as(preds)
                                   .to(labels.device)
                              - labels.unsqueeze(1)
                                      .type_as(preds)
                                      .expand_as(preds)) ** 2
        soft_labels = F.softmax(soft_labels, dim=1)
        loss = -F.log_softmax(preds, dim=1) * soft_labels
        loss = weight_reduce_loss(
            loss,
            weight=self.weight,
            reduction=self.reduction,
            avg_factor=self.avg_factor)
        return loss


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


@LOSSES.register_module()
class BinaryLabelSmoothLoss(nn.Module):

    def __init__(self,
                 label_smooth_val,
                 reduction='mean',
                 loss_weight=1.0):
        super(BinaryLabelSmoothLoss, self).__init__()
        self.label_smooth_val = label_smooth_val
        self.avg_smooth_val = self.label_smooth_val / 2
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        prob = torch.sigmoid(cls_score).view(-1, 1)
        prob = torch.cat([1 - prob, prob], 1)

        # # element-wise losses
        one_hot = torch.zeros_like(prob)
        one_hot.fill_(self.avg_smooth_val)
        label = label.view(-1, 1).type(torch.long)
        one_hot.scatter_(
            1, label, 1 - self.label_smooth_val + self.avg_smooth_val)
        loss = -torch.log(prob) * one_hot.detach()

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        loss = self.loss_weight * loss
        return loss
