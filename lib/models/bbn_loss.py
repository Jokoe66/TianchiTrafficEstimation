import torch
import torch.nn as nn
from mmcls.models.builder import LOSSES
from mmcls.models import build_loss

@LOSSES.register_module()
class BBNLoss(nn.Module):

    def __init__(self, max_steps, criterian, **kwargs):
        super(BBNLoss, self).__init__()
        self.step = 0
        self.max_steps = max_steps
        self.criterian = build_loss(criterian)

    def get_alpha(self):
        # parabolic-decay
        alpha = 1 - (self.step * 1. / self.max_steps) ** 2
        return alpha

    def update_step(self):
        self.step += 1

    def reset_step(self):
        self.step = 0

    def forward(self, preds, labels1, labels2):
        alpha = self.get_alpha()
        loss = (alpha * self.criterian(preds, labels1)
                + (1 - alpha) * self.criterian(preds, labels2))
        self.update_step()
        return loss
