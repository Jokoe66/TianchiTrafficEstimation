import argparse
import re
import glob
import os
from collections import defaultdict, Counter

import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import mmcv
from mmcv.utils import Config
from torch.utils.data import DataLoader
from mmcls.models.builder import build_classifier
from sklearn.model_selection import KFold

from lib.datasets import ImageSequenceDataset
from lib.models import Classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', type=str, default='')
    parser.add_argument('--feat_file', type=str, default='')
    args = parser.parse_args()

    anns = mmcv.load(args.ann_file)
    feats = mmcv.load(args.feat_file)
    results = defaultdict(
        lambda :defaultdict(list)) # seq_id: dict(feat1:[], feat2:[])
    img_ids = list(feats.keys())
    img_ids.sort()
    print('Transform features')
    for img_id in tqdm.tqdm(img_ids):
        seq_id = re.findall('(\d+)/', img_id)[0]
        counts = Counter(feats[img_id].flatten())
        h, w = feats[img_id].shape
        for i in range(194):
            results[seq_id][f'seg{i}'].append(
                counts.get(i, 0) * 1. / h / w)
    for i, ann in enumerate(anns):
        ann['feats'].update(dict(results[ann['id']]))
    mmcv.dump(anns, args.ann_file[:-4] + '_seg.pkl')
