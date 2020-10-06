import torch
import torch.nn as nn

class AlphaScheduler(nn.Module):

    def __init__(self, max_steps):
        super(AlphaScheduler, self).__init__()
        self.max_steps = max_steps
        self.register_buffer('cur_step', torch.tensor(0.))

    def reset(self):
        self.cur_step.fill_(0.)

    def step(self):
        self.cur_step += 1

    @property
    def alpha(self):
        alpha = 1 - (self.cur_step / self.max_steps) ** 2
        return alpha.item()

    def __repr__(self):
        return self.__class__.__name__ + f'(max_steps={self.max_steps})'
