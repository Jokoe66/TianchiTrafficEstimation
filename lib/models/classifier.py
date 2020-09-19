import torch
import mmcv
from mmdet.models import build_backbone

class Classifier(torch.nn.Module):

    def __init__(self, backbone, pretrained=None, num_classes=4, **kwargs):
        super(Classifier, self).__init__()
        self.num_classes = num_classes

        self.backbone = build_backbone(backbone)
        self.backbone.init_weights(pretrained)

        h, w, c = 9, 16, 256 # make feat_size indpendent on input_size
        self.pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((h, w)),
            torch.nn.Conv2d(2048, c, 1, 1, 0),
            torch.nn.ReLU(inplace=True)
        )

        hidden_size = kwargs.get('lstm')
        self.lstm = torch.nn.GRU(h * w * c , hidden_size) if hidden_size else None
        hidden_size = hidden_size or (h * w * c)

        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.00),
            torch.nn.Linear(hidden_size, num_classes))


    def forward(self, input, seq_len=5):
        feat = self.backbone(input)[0]
        
        feat = self.pool(feat)
        n, c, h, w = feat.shape
        feat = feat.view(n, -1)
        if self.lstm:
            feat = feat.view(seq_len, len(feat)//seq_len, -1)
            feat, _ = self.lstm(feat)
            feat = feat.mean(0)
        
        logit = self.fc(feat)
        return logit 

