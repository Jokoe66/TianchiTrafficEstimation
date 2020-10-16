import argparse
import glob
import os
from collections import defaultdict

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
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--ann_file', type=str, default='')
    parser.add_argument('--config_file', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--key_frame_only', action='store_true',
                        default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    test_set = ImageSequenceDataset(
        args.img_root,
        args.ann_file,
        'test',
        key_frame_only=args.key_frame_only,
        transform=cfg.test_pipeline) # consistent with training time's input_size
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    k = 5
    kf = KFold(k, shuffle=True, random_state=666)
    models = [build_classifier(cfg.model).to(args.device) for _ in range(k)]

    # construct mapping between data index and validation set idx
    ind2fold = defaultdict(int) # default to 0
    indices = range(len(test_loader))
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        for ind in val_idx:
            ind2fold[ind] = fold
        models[fold].load_state_dict(
            torch.load(os.path.join(args.model_dir, f'best{fold+1}.pth'),
               map_location='cpu'))
        models[fold].eval()

    results = dict() # id: pred
    for ind, data in enumerate(tqdm.tqdm(test_loader)):
        fold = ind2fold[ind]
        self = models[fold]
        imgs = data.pop('imgs')
        with torch.no_grad():
            preds = self(imgs, return_features=True, **data)
        id = test_set.anns[ind]['id']
        results[id] = preds.cpu().numpy()[0]
    enriched_anns = mmcv.load(args.ann_file)
    for ann in enriched_anns:
        ann['feats']['dnn_feats'] = results[ann['id']]
    mmcv.dump(enriched_anns, args.ann_file[:-4] + '_dnn.pkl')
