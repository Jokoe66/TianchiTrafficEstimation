import argparse

import tqdm
import pandas as pd
import numpy as np
import torch
import mmcv
from torch.utils.data import DataLoader

from lib.datasets import ImageSequenceDataset
from lib.models import Classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--ann_file', type=str, default='')
    parser.add_argument('--test_file', type=str,
                        default='/tcdata/amap_traffic_final_test_0906.json')
    parser.add_argument('--model_path', type=str, default='')
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
        type='ResNeSt',
        depth=101, #50, 101, 152, 200, 269
        num_stages=4,
        stem_channels=128,
        out_indices=(3, ),
        frozen_stages=4,
        style='pytorch')
    self = Classifier(
        backbone,
        num_classes=4,
        lstm=lstm,
        feat_mask_dim=2,
        feat_vec_dim=10).to(args.device)
    self.load_state_dict(torch.load(args.model_path, map_location='cpu'))
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
