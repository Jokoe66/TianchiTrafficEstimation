import torch
import torch.nn as nn

class BBNLoss(nn.Module):

    def __init__(self, criterian, max_steps, **kwargs):
        super(BNNLoss, self).__init__()
        self.step = 0
        self.max_steps = max_steps
        self.criterian = criterian

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
        self.update_step()
        loss = (alpha * self.criterian(preds, labels1)
                + (1 - alpha) * self.criterian(preds, labels2))
        return loss
