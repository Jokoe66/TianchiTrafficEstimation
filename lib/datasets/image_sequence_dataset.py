import os
from collections import defaultdict

import pycocotools._mask as mask_utils
from PIL import Image
import numpy as np
import torch
import mmcv
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from mmcv.utils import build_from_cfg
from mmcls.datasets import PIPELINES


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
        if transform:
            self.transform = Compose([
                build_from_cfg(t, PIPELINES) for t in transform])
        else:
            self.transform = None
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
        data = self.parse_ann(ann)
        if self.transform:
            data = self.transform(data)
        return data

    def parse_ann(self, ann):
        results = defaultdict(list)
        results['img_info'] = dict(
            filenames=[_['frame_name'] for _ in ann['frames']]
        )
        results['img_prefix'] = os.path.join(self.img_root, ann['id'])
        results['labels'] = ann['status']
        if self.key_frame_only:
            img = mmcv.imread(
                os.path.join(results['img_prefix'],
                ann['key_frame'])) # bgr mode
            results['imgs'] = img
            key_idx = results['img_info']['filenames'].index(ann['key_frame'])
            results['feat_mask'] = self.gen_feat_mask(
                ann['frames'][key_idx].get('feats'),
                *img.shape[:2], keys=['vehicles', 'obstacles'])
            seq_feats = self.gen_feat_vector(ann.get('feats'),
                seq_len=len(ann['frames']), check=True)
            results['feat_vector'] = seq_feats[key_idx]
            results['key'] = 0
            return dict(**results)

        for i, frame in enumerate(ann['frames']):
            assert int(frame['frame_name'][0]) == i + 1, \
                f'{frame["frame_name"]} is not {i+1}th frame'
            if frame['frame_name'] == ann['key_frame']:
                results['key'] = i
            img = mmcv.imread(
                os.path.join(results['img_prefix'], frame['frame_name']))
            results['imgs'].append(img)
            results['feat_mask'].append(
                self.gen_feat_mask(frame.get('feats'), *img.shape[:2],
                                   keys=['vehicles', 'obstacles']))
        seq_feats = self.gen_feat_vector(ann.get('feats'),
            seq_len=len(ann['frames']), check=True)
        results['feat_vector'] = seq_feats
        return dict(**results)

    def __len__(self):
        return len(self.anns)
