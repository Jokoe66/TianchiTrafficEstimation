import os
from collections import defaultdict

import pycocotools._mask as mask_utils
from PIL import Image
import numpy as np
import torch
import mmcv
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class ImageSequenceDataset(Dataset):

    img_root = '../data/amap_traffic_%s_0712/'
    ann_file = '../data/amap_traffic_annotations_%s.json'
    img_root = '../data/amap_traffic_b_%s_0828'
    ann_file = '../data/amap_traffic_annotations_b_%s_0828.json'

    def __init__(self,
                 img_root=None,
                 ann_file=None,
                 split='train',
                 transform=None,
                 **kwargs):
        self.img_root = img_root if img_root else self.img_root % split
        self.ann_file = ann_file if ann_file else self.ann_file % split
        self._load_anns()
        self.img_norm = dict(mean=[123.675, 116.28, 103.53],
                             std=[58.395, 57.12, 57.375])
        if transform:
            self.transform = transform
        else:
            input_size = kwargs.get('input_size', (1280, 720))
            self.transform = Compose([
                lambda x: mmcv.imresize(x, input_size),
                ToTensor(),
                Normalize(**self.img_norm)
            ])
        self.seq_max_len = kwargs.get('seq_max_len', 5)
        self.key_frame_only = kwargs.get('key_frame_only', False)
        
    def _load_anns(self):
        self.anns = mmcv.load(self.ann_file)
        if isinstance(self.anns, dict):
            self.anns = self.anns['annotations']
        for ann in self.anns:
            ann['frames'].sort(key=lambda x:x['frame_name'])

    def get_cat_ids(self, idx):
        return self.anns[idx]['status']

    def gen_feat_mask(self, feats, h, w, keys=None):
        """ Generate masks from boxes, e.g. vehicles, obstacles

        Args:
            feats (dict): dictionary of str: ndarray
            h, w (int): height and width of generated mask
            keys (list[str]): keys of features to generate masks
        Returns:
            ndarray: mask of shape (h, w, c), c denotes number of features
        """
        mask = np.empty((h, w, 0))
        if not feats:
            return mask
        for key, feat in feats.items():
            if keys and key not in keys: continue
            rles = mask_utils.frBbox(feat.astype('float64'), h, w)
            if len(rles):
                _mask = mask_utils.decode([mask_utils.merge(rles)]).astype(float)
            else:
                _mask = np.zeros((h, w, 1)).astype('float')
            for box in feat:
                l, t, r, b, c = box
                l, t, r, b = list(map(int, [l, t, r, b]))
                _mask[t:b, l:r] = c
            mask = np.concatenate([mask, _mask], 2)
        return mask

    def gen_feat_vector(self, feats, seq_len, keys=None, check=False):
        """ Load specific sequence features from the annotation.

        Args:
            feats (dict): str: list, sequence features, e.g. num_vehicles
            seq_len (int): length of sequence
            keys (list[str]): names of feature to load
            check (bool): whether to check if the element of each feature 
                          is a single number.
        Return:
            list[ndarray]: list of feature vector
        """
        feat = np.empty((seq_len, 0))
        if not feats:
            return list(feat)
        for k, v in feats.items():
            if keys and k not in keys: continue
            if (check
                and any(not isinstance(e, (int, float)) for e in v)
                ):
                continue
            if seq_len != len(v): continue
            feat = np.hstack([feat, np.array(v)[:,None]])
        return list(feat)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        if self.key_frame_only:
            img = Image.open(os.path.join(self.img_root,
                    ann['id'], ann['key_frame']))
            img = np.array(img)
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(img)
            key_idx = [_['frame_name'] for _ in ann['frames']].index(
                ann['key_frame'])
            feats = dict()
            feats['feat_mask'] = self.gen_feat_mask(
                ann['frames'][key_idx].get('feats'),
                *img.shape[:2], keys=['vehicles', 'obstacles'])
            seq_feats = self.gen_feat_vector(ann.get('feats'),
                seq_len=len(ann['frames']), check=True)
            feats['feat_vector'] = seq_feats[key_idx]

            return dict(imgs=img,
                        key=0,
                        len_seq=1,
                        seq_len=1,
                        label=ann['status'],
                        **feats)

        imgs = []
        feats = defaultdict(list)

        for i in range(self.seq_max_len):
            if i < len(ann['frames']):
                frame = ann['frames'][i]
                assert int(frame['frame_name'][0]) == i + 1, \
                    f'{frame["frame_name"]} is not {i+1}th frame'
                if frame['frame_name'] == ann['key_frame']:
                    ann['key'] = i
                img = Image.open(os.path.join(self.img_root, 
                    ann['id'], frame['frame_name']))
                img = np.array(img)
                feats['feat_mask'].append(
                    self.gen_feat_mask(frame.get('feats'), *img.shape[:2],
                                       keys=['vehicles', 'obstacles']))
            else:
                img = np.zeros(imgs[0].shape[-2:] + (3,))
                img = img + self.img_norm['mean']
                img = img.astype('uint8')
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(img)
            imgs.append(img)
        imgs = torch.stack(imgs, -1)
        seq_feats = self.gen_feat_vector(ann.get('feats'),
            seq_len=len(ann['frames']), check=True)
        feats['feat_vector'] = seq_feats

        # pad and stack ndarrays with same shape (e.g. masks)
        for k, v in feats.items():
            if isinstance(v[0], np.ndarray) and len({b.shape for b in v}) == 1:
                # pad ndarray features, e.g. masks
                v.extend((self.seq_max_len - len(v)) * [np.zeros_like(v[0])])
                feats[k] = np.stack(v, -1)
        return dict(imgs=imgs,
                    key=ann['key'],
                    len_seq=len(ann['frames']),
                    seq_len=len(ann['frames']),
                    label=ann['status'],
                    **feats)

    def __len__(self):
        return len(self.anns)
