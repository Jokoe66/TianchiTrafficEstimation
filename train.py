import argparse

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from lib import ImageSequenceDataset, ClassBalancedSubsetSampler
from lib.models import Classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--ann_file', type=str, default='')
    parser.add_argument('--key_frame_only', action='store_true',
                        default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--milestones', nargs='+', type=int,
                        default=[8, ])
    args = parser.parse_args()

    lstm = None if args.key_frame_only else 128

    training_set = ImageSequenceDataset(
        args.img_root,
        args.ann_file,
        'train',
        key_frame_only=args.key_frame_only,
        input_size=(640, 360)) # 0.5x input_size for better efficiency

    bs = args.batch_size
    outputs = []

    indices = np.arange(len(training_set))
    k = 5
    kf = KFold(k, shuffle=True, random_state=666)
    for idx, (train_inds, val_inds) in enumerate(kf.split(indices)):
        train_loader = DataLoader(training_set, batch_size=bs, num_workers=4,
            sampler=ClassBalancedSubsetSampler(training_set, train_inds))
        val_loader = DataLoader(training_set, batch_size=bs, num_workers=4,
            sampler=torch.utils.data.SubsetRandomSampler(val_inds))

        model = Classifier(num_classes=4, lstm=lstm).to(args.device)
        output = model.fit(train_loader,
            val_dataloader=val_loader,
            log_iters=10,
            max_epoch=args.max_epoch,
            milestones=args.milestones,
            lr=args.lr,
            class_weights=[0.1, 0.2, 0.3, 0.4],
            save_dir=f'work_dirs/classification/fold{idx + 1}')
        outputs.append(output)
        print(f'{k}-fold cross validation: {idx + 1}.')
        print(pd.DataFrame(outputs)[-1:].transpose())
    print(pd.DataFrame(outputs)
            .describe()
            .transpose()
            [['mean','std','min','max']])
