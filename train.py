import argparse

import torch
from torch.utils.data import DataLoader

from lib import Classifier, ImageSequenceDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    model = Classifier()
    model = model.to(args.device)

    training_set = ImageSequenceDataset('train')

    bs = args.batch_size
    split = len(training_set) // 5
    indices = torch.randperm(len(training_set))

    train_loader = DataLoader(training_set, batch_size=bs, num_workers=0,
        sampler=torch.utils.data.SubsetRandomSampler(indices[split:]))
    val_loader = DataLoader(training_set, batch_size=bs, num_workers=0,
        sampler=torch.utils.data.SubsetRandomSampler(indices[:split]))

    model.fit(train_loader, val_dataloader=val_loader, log_iters=10, )
