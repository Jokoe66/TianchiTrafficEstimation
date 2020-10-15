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
from sklearn.metrics import f1_score, confusion_matrix

from lib.datasets import ImageSequenceDataset, DistributedTestSubsetSampler
from lib.models import Classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--ann_file', type=str, default='')
    parser.add_argument('--config_file', type=str, default='')
    parser.add_argument('--test_file', type=str,
                        default='/tcdata/amap_traffic_final_test_0906.json')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--ensemble', type=int, default=0)
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
        #sampler=DistributedTestSubsetSampler(test_set, range(len(test_set))))

    k = 5
    kf = KFold(k, shuffle=True, random_state=666)
    models = [build_classifier(cfg.model).to(args.device) for _ in range(k)]

    # construct mapping between data index and validation set idx
    ind2fold = defaultdict(int) # default to 0
    indices = range(len(test_set))
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        for ind in val_idx:
            ind2fold[ind] = fold
        models[fold].load_state_dict(
            torch.load(f'../user_data/res50/best{fold+1}.pth',
            #torch.load(f'work_dirs/classification/fold{fold+1}/classifier_epoch6.pth',
               map_location='cpu'))
        models[fold].eval()

    preds = []
    labels = []
    sampled_indices = list(iter(test_loader.sampler)) #deterministic sampler
    # when evaluating scene classification, the forward method of MultiClsHead
    # should output the outputs of scene classification head.
    eval_type = 'labels' # or 'scenes'
    for ind, data in enumerate(tqdm.tqdm(test_loader)):
        ind = sampled_indices[ind]
        fold = ind2fold[ind]
        self = models[fold]
        imgs = data.pop('imgs')
        with torch.no_grad():
            pred = self(imgs, **data)
        preds.append(pred.argmax(1).cpu().numpy()[0])
        #labels.append(test_set.anns[ind]['status'])
        labels.append(data[eval_type].cpu().item())
    result_file = '/tmp/tmp_result.pkl'
    mmcv.dump([labels, preds], result_file)
    if eval_type == 'scenes':
        labels = np.array(labels)
        preds = np.array(preds)
        mask = np.where(labels != -1)
        labels = labels[mask]
        preds = preds[mask]
    f1s = f1_score(labels, preds, average=None)
    cm = confusion_matrix(labels, preds)
    print(cm)
    print(f1s)
    #print((np.array([0.1, 0.2, 0.3, 0.4]) * np.array(f1s)).sum())
