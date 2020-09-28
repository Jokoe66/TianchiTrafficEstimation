import argparse
import glob

import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import mmcv
from torch.utils.data import DataLoader

from lib.datasets import ImageSequenceDataset
from lib.models import Classifier

class Ensemble(torch.nn.Module):

    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        preds = []
        for model in self.models:
            pred = model(*args, **kwargs)
            pred = F.softmax(pred, 1)
            preds.append(pred)
        preds = torch.stack(preds, 0)
        preds = preds.mean(0)
        return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--ann_file', type=str, default='')
    parser.add_argument('--test_file', type=str,
                        default='/tcdata/amap_traffic_final_test_0906.json')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--ensemble', type=int, default=0)
    parser.add_argument('--key_frame_only', action='store_true',
                        default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    test_set = ImageSequenceDataset(
        args.img_root,
        args.ann_file,
        'test',
        key_frame_only=args.key_frame_only,
        input_size=(640, 360)) # consistent with training time's input_size
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    lstm = None if args.key_frame_only else 128
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=4,
        style='pytorch')
    params= dict(
        backbone=backbone,
        pretrained=None,
        bb_style='mmcls', # build backbone with mmcls or mmdet
        bb_feat_dim=2048,
        num_classes=4,
        lstm=lstm,
        bilinear_pooling=False,
        feat_mask_dim=2,
        feat_vec_dim=10,
        )
    if args.ensemble:
        self = Ensemble([Classifier(**params) for _ in range(args.ensemble)])
        self = self.to(args.device)
        for idx, model in enumerate(self.models):
            model.load_state_dict(
                torch.load(glob.glob(args.model_path)[idx],
                   map_location='cpu'))
    else:
        self = Classifier(**params).to(args.device)
        self.load_state_dict(torch.load(glob.glob(args.model_path)[0],
            map_location='cpu'))
    self.eval()
    results = dict() # id: pred
    for ind, data in enumerate(tqdm.tqdm(test_loader)):
        imgs = data['imgs']
        if len(imgs.shape) > 4: # frames
            seq_len = imgs.shape[-1]
            imgs = (imgs.permute(4, 0, 1, 2, 3).contiguous()
                .view(-1, *imgs.shape[1:4])) # t, c, h, w
        else:
            seq_len = 1
        imgs = imgs.to(next(self.parameters()).device)

        data.pop('seq_len')
        with torch.no_grad():
            preds = self(imgs, seq_len, **data)
        pred = preds.argmax(1).detach().cpu().numpy()[0]
        id = test_set.anns[ind]['id']
        results[id] = pred
    result_json = mmcv.load(args.test_file)
    for ann in result_json['annotations']:
        ann['status'] = results[ann['id']]
    mmcv.dump(result_json, './result.json')
