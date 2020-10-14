import argparse
import glob

import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import mmcv
from mmcv.utils import Config
from torch.utils.data import DataLoader
from mmcls.models.builder import build_classifier

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
            if (torch.abs(pred.sum(1) - 1) > 1e-4).any():
                pred = F.softmax(pred, 1)
            preds.append(pred)
        preds = torch.stack(preds, 0)
        preds = preds.mean(0)
        return preds

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

    if args.ensemble:
        self = Ensemble(
            [build_classifier(cfg.model) for _ in range(args.ensemble)])
        self = self.to(args.device)
        for idx, model in enumerate(self.models):
            model.load_state_dict(
                torch.load(glob.glob(args.model_path)[idx],
                   map_location='cpu'))
    else:
        self = build_classifier(cfg.model).to(args.device)
        self.load_state_dict(torch.load(glob.glob(args.model_path)[0],
            map_location='cpu'))
    self.eval()
    results = dict() # id: pred
    for ind, data in enumerate(tqdm.tqdm(test_loader)):
        imgs = data.pop('imgs')
        with torch.no_grad():
            preds = self(imgs, **data)
        pred = preds.argmax(1).detach().cpu().numpy()[0]
        id = test_set.anns[ind]['id']
        results[id] = pred
    result_json = mmcv.load(args.test_file)
    for ann in result_json['annotations']:
        ann['status'] = results[ann['id']]
    mmcv.dump(result_json, './result.json')
