import argparse

import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from lib import ImageSequenceDataset
from lib.classification import Classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--ann_file', type=str, default='')
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
    test_loader = DataLoader(test_set, batch_size=4, num_workers=0)

    lstm = None if args.key_frame_only else 128
    self = Classifier(num_classes=4, lstm=lstm).to(args.device)
    self.load_state_dict(torch.load(args.model_path))

    self.eval()
    all_preds = np.empty((0,))
    all_labels = np.empty((0,))
    for ind, data in enumerate(tqdm.tqdm(test_loader)):
        imgs = data['imgs']
        if len(imgs.shape) > 4: # frames
            seq_len = imgs.shape[-1]
            imgs = (imgs.permute(4, 0, 1, 2, 3).contiguous()
                .view(-1, *imgs.shape[1:4]))
        else:
            seq_len = 1
        labels = data['label']

        imgs = imgs.to(next(self.parameters()).device)
        labels = labels.to(next(self.parameters()).device)

        with torch.no_grad():
            preds = self(imgs, seq_len)

        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
        all_preds = np.hstack([all_preds,
            preds.argmax(1).detach().cpu().numpy()])
    f1_scores = f1_score(all_labels, all_preds, average=None)
    f1 = (np.array([0.1, 0.2, 0.3, 0.4]) * f1_scores).sum()
    print(f'f1_scores: {f1_scores.tolist()}\n'
          f'f1_score: {f1}')
