import argparse

import torch
from torch.utils.data import DataLoader

from lib import Classifier, ImageSequenceDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key_frame_only', action='store_true',
                        default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_epoch', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--milestones', nargs='+', type=int,
                        default=[8, ])
    args = parser.parse_args()

    lstm = None if args.key_frame_only else 128
    model = Classifier(lstm=lstm)
    model = model.to(args.device)

    training_set = ImageSequenceDataset('train',
        key_frame_only=args.key_frame_only)

    bs = args.batch_size
    split = len(training_set) // 5
    indices = torch.randperm(len(training_set))

    train_loader = DataLoader(training_set, batch_size=bs, num_workers=0,
        sampler=torch.utils.data.SubsetRandomSampler(indices[split:]))
    val_loader = DataLoader(training_set, batch_size=bs, num_workers=0,
        sampler=torch.utils.data.SubsetRandomSampler(indices[:split]))

    model.fit(train_loader,
              val_dataloader=val_loader,
              log_iters=10,
              max_epoch=args.max_epoch,
              milestones=args.milestones,
              lr=args.lr)
