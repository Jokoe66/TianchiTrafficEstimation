import torch
import torch.nn.functional as F
import mmcv
from mmcv.runner import get_dist_info
from mmcls.models import build_backbone as build_backbone_mmcls
from mmdet.models import build_backbone as build_backbone_mmdet

_BACKBONE_BUILDER = {
    'mmdet': build_backbone_mmdet,
    'mmcls': build_backbone_mmcls}

class Classifier(torch.nn.Module):

    def __init__(self, backbone, pretrained=None, num_classes=4, **kwargs):
        super(Classifier, self).__init__()
        self.num_classes = num_classes

        bb_style = kwargs.get('bb_style', 'mmcls')
        self.backbone = _BACKBONE_BUILDER[bb_style](backbone)
        self.backbone.init_weights(pretrained)

        h, w, c = 9, 16, 256 # make feat_size indpendent on input_size
        self.pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((h, w)),
            torch.nn.Conv2d(kwargs.get('bb_feat_dim', 2048), c, 1, 1, 0),
            torch.nn.ReLU(inplace=True)
        )
        self.feat_mask_dim = kwargs.get('feat_mask_dim', 0)
        c += self.feat_mask_dim

        use_bilinear_pooling = kwargs.get('bilinear_pooling', False)
        self.use_bilinear_pooling = use_bilinear_pooling
        bilinear_dim = c * c

        self.feat_vec_dim = kwargs.get('feat_vec_dim', 0)

        in_channels = h * w * c + self.feat_vec_dim \
            if not use_bilinear_pooling  else bilinear_dim + self.feat_vec_dim
        hidden_size = kwargs.get('lstm')
        self.lstm = torch.nn.GRU(
            in_channels, hidden_size) if hidden_size else None

        hidden_size = hidden_size or in_channels
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.00),
            torch.nn.Linear(hidden_size, num_classes))


    def forward(self, input, seq_len=5, **kwargs):
        feat = self.backbone(input)
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        feat = self.pool(feat)
        n, c, h, w = feat.shape
        # feature mask fusion
        assert self.feat_mask_dim == 0 or \
            (self.feat_mask_dim > 0) ^ ('feat_mask' not in kwargs), \
            "feat_mask should be in kwargs when feat_mask_dim > 0"
        if self.feat_mask_dim:
            # n, h, w, c,( t)
            feat_mask = kwargs['feat_mask'].to(next(self.parameters()))
            if len(feat_mask.shape) == 4: # n, h, w, c
                feat_mask = feat_mask.unsqueeze(-1) # n, h, w, 1
            assert feat_mask.shape[3] == self.feat_mask_dim, \
                (f"feat_vector.shape[1]({feat_vector.shape[1]}) != "
                 f"self.feat_vec_dim({self.feat_vec_dim}).")
            feat_mask = feat_mask.permute(4,0,3,1,2).contiguous() # t, n, c, h, w
            # txn, c, h, w
            feat_mask = feat_mask.view(-1, *feat_mask.shape[2:])
            feat_mask = F.interpolate(feat_mask, (h, w), mode='nearest')
            feat = torch.cat([feat, feat_mask], 1)

        # bilinear pooling models interactions between features
        if self.use_bilinear_pooling:
            n, c, h, w = feat.shape
            feat = feat.view(n, c, h * w)
            feat = torch.bmm(feat, feat.transpose(1, 2)).view(n, -1) / (h * w)
            feat = F.normalize(torch.sqrt(feat + 1e-10))

        feat = feat.view(n, -1)
        # feature vector fusion
        assert self.feat_vec_dim == 0 or \
            (self.feat_vec_dim > 0) ^ ('feat_vector' not in kwargs), \
            "feat_vector should be in kwargs when feat_vec_dim > 0"
        if self.feat_vec_dim:
            feat_vector = kwargs['feat_vector'].to(next(self.parameters())) # n, c, t
            if len(feat_vector.shape) == 2: # n, c
                feat_vector = feat_vector.unsqueeze(-1) # n, c, 1
            assert feat_vector.shape[1] == self.feat_vec_dim, \
                (f"feat_vector.shape[1]({feat_vector.shape[1]}) != "
                 f"self.feat_vec_dim({self.feat_vec_dim}).")
            feat_vector = feat_vector.permute(2,0,1).contiguous() # t, n, c
            feat_vector = feat_vector.view(-1, *feat_vector.shape[2:]) # txn, c
            feat = torch.cat([feat, feat_vector], 1)
        if self.lstm:
            feat = feat.view(seq_len, len(feat)//seq_len, -1)
            feat, _ = self.lstm(feat)
            feat = feat.mean(0)
        
        logit = self.fc(feat)
        return logit 

