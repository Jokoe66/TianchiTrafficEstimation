from mmdet.models.backbones import HRNet
from mmdet.models.builder import BACKBONES

@BACKBONES.register_module()
class HRNetm(HRNet):

    def __init__(self,
                 *args,
                 frozen_stages=-1,
                 **kwargs):
        super(HRNetm, self).__init__(*args, **kwargs)
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def train(self, mode=True):
        super(HRNetm, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.norm1, self.conv2, self.norm2]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        if self.frozen_stages >= 1:
            m = self.layer1
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        for i in range(2, self.frozen_stages + 1):
            m = getattr(self, f'transition{i - 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            m = getattr(self, f'stage{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
