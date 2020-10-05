import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import NECKS


@NECKS.register_module()
class Pool(nn.Module):
    """ Pooling Module.

    """
    def __init__(self,
                 in_channel,
                 pool_h=9,
                 pool_w=16,
                 pool_c=256,
                 flatten=False
                 ):
        super(Pool, self).__init__()
        h, w, c = pool_h, pool_w, pool_c # make feat_size indpendent on input_size
        self.pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((h, w)),
            torch.nn.Conv2d(in_channel, c, 1, 1, 0),
            torch.nn.ReLU(inplace=True)
        )
        self.flatten = flatten

    def forward(self, feat, **kwargs):
        feat = self.pool(feat)
        if self.flatten:
            feat = feat.view(len(feat), -1)
        return feat


class FF(nn.Module):
    """ Feature Fusion Module.

    """
    def __init__(self,
                 feat_mask_dim=0,
                 feat_vec_dim=0,
                 ):
        super(FF, self).__init__()
        self.feat_mask_dim = feat_mask_dim
        self.feat_vec_dim = feat_vec_dim 

    def feat_mask_fusion(self, feat, feat_mask):
        """
        Args:
            feat(torch.Tensor): conv feature maps(N, C, H, W)
            feat_mask(torch.Tensor): hand-crafted features(N', H', W', C', T)
        Return:
            torch.Tensor: fused feature maps(N, C+C', H, W)
        """
        n, c, h, w = feat.shape
        assert self.feat_mask_dim == 0 or \
            (self.feat_mask_dim > 0) ^ (feat_mask is None), \
            "feat_mask should be not none when feat_mask_dim > 0"
        if self.feat_mask_dim:
            # n, h, w, c,( t)
            feat_mask = feat_mask.to(next(self.parameters()))
            if len(feat_mask.shape) == 4: # n, h, w, c
                feat_mask = feat_mask.unsqueeze(-1) # n, h, w, 1
            assert feat_mask.shape[3] == self.feat_mask_dim, \
                (f"feat_maskk.shape[3]({feat_mask.shape[3]}) != "
                 f"self.feat_mask_dim({self.feat_mask_dim}).")
            feat_mask = feat_mask.permute(4,0,3,1,2).contiguous() # t, n, c, h, w
            # txn, c, h, w
            feat_mask = feat_mask.view(-1, *feat_mask.shape[2:])
            feat_mask = F.interpolate(feat_mask, (h, w), mode='nearest')
            feat = torch.cat([feat, feat_mask], 1)
        return feat

    def feat_vector_fusion(self, feat, feat_vector):
        """
        Args:
            feat(torch.Tensor): DNN feature vector(N, C)
            feat_vector(torch.Tensor): hand-crafted feature vector(N', C', T)
        Return:
            torch.Tensor: fused feature vector(N, C+C')
        """
        feat = feat.view(len(feat), -1)
        assert self.feat_vec_dim == 0 or \
            (self.feat_vec_dim > 0) ^ ('feat_vector' is None), \
            "feat_vector should not be None when feat_vec_dim > 0"
        if self.feat_vec_dim:
            feat_vector = feat_vector.to(next(self.parameters())) # n, c, t
            if len(feat_vector.shape) == 2: # n, c
                feat_vector = feat_vector.unsqueeze(-1) # n, c, 1
            assert feat_vector.shape[1] == self.feat_vec_dim, \
                (f"feat_vector.shape[1]({feat_vector.shape[1]}) != "
                 f"self.feat_vec_dim({self.feat_vec_dim}).")
            feat_vector = feat_vector.permute(2,0,1).contiguous() # t, n, c
            feat_vector = feat_vector.view(-1, *feat_vector.shape[2:]) # txn, c
            feat = torch.cat([feat, feat_vector], 1)
        return feat

    def forward(self, feat, **kwargs):
        feat = self.feat_mask_fusion(feat, kwargs.get('feat_mask'))
        feat = self.feat_vector_fusion(feat, kwargs.get('feat_vector'))
        return feat

    def extra_repr(self):
        return (f'feat_mask_dim={self.feat_mask_dim}, '
                f'feat_vec_dim={self.feat_vec_dim}')


class Seq(nn.Module):

    def __init__(self, in_channel, hidden_size):
        super(Seq, self).__init__()
        self.lstm = torch.nn.GRU(
            in_channel, hidden_size) if hidden_size else None

    def forward(self, feat, seq_len_max, seq_len=None, key=None, **kwargs):
        feat = feat.view(seq_len_max, len(feat)//seq_len_max, -1)
        feat, _ = self.lstm(feat)
        if key is not None:
            # emphasize key frame
            weights = feat.new_ones(feat.shape) * 0.8
            for b in range(len(key)):
                weights[key[b], b] *= 1.5
            feat = feat * weights
        if seq_len is not None:
            feat = torch.stack(
                [feat[:seq_len[b], b].mean(0) for b in range(len(seq_len))], 0) #b, c
        else:
            feat = feat.mean(0)
        return feat


class BilinearPooling(nn.Module):

    def forward(self, feat):
        # bilinear pooling models interactions between features
        n, c, h, w = feat.shape
        feat = feat.view(n, c, h * w)
        feat = torch.bmm(feat, feat.transpose(1, 2)).view(n, -1) / (h * w)
        feat = F.normalize(torch.sqrt(feat + 1e-10))
        return feat


@NECKS.register_module()
class PFFSeqNeck(nn.Module):
    """ Neck with Pooling, Feature Fusion and SeqNeck.

    Example config:
        neck=dict(type='PFFSeqNeck', 
                  pooling=dict(in_channel=2048,
                               pool_h=9,
                               pool_w=16,
                               pool_c=256),
                  feature_fusion=dict(feat_mask_dim=0,
                                      feat_vec_dim=0),
                  lstm=dict(hidden_size=128))
    """
    def __init__(self,
                 pooling=None,
                 feature_fusion=None,
                 lstm=None,
                 ):
        super(PFFSeqNeck, self).__init__()
            #in_channel, pool_h, pool_w, pool_c)
        self.pooling = Pool(**pooling)
        self.ff = FF(**feature_fusion)
        self.lstm = Seq(**lstm)

    def forward(self, feat, **kwargs):
        # pooling
        feat = self.pooling(feat)
        # feature fusion
        feat = self.ff(feat, **kwargs)
        # combine seqence features
        feat = self.lstm(feat, **kwargs)
        return feat


@NECKS.register_module()
class PFFBPSeqNeck(PFFSeqNeck):
    """ Neck with Pooling, Feature Fusion, Bilinear Pooling  and SeqNeck.

    Example config:
        neck=dict(type='PFFBPSeqNeck', 
                  pooling=dict(in_channel=2048,
                               pool_h=9,
                               pool_w=16,
                               pool_c=256),
                  bilinear=dict(
                  feature_fusion=dict(feat_mask_dim=0,
                                      feat_vec_dim=0),
                  lstm=dict(hidden_size=128))
    """
    def __init__(self, *args, **kwargs):
        super(PFFBPSeqNeck, self).__init__(*args, **kwargs)
        self.bp = BilinearPooling()

    def forward(self, feat, **kwargs):
        # pooling
        feat = self.pooling(feat)
        # feature mask fusion
        feat = self.ff.feat_mask_fusion(feat, kwargs.get('feat_mask'))
        # bilinear pooling
        feat = self.bp(feat)
        # feature vector fusion
        feat = self.ff.feat_vector_fusion(feat, kwargs.get('feat_vector'))
        # combine seqence features
        feat = self.lstm(feat, **kwargs)
        return feat
