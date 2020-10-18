import os
import random
from collections import defaultdict
import argparse
import sys
sys.path.insert(0, 'lib/mmdetection')

import pandas as pd
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
from mmcv.utils import Config
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from mmcls.models import build_classifier
from mmcls.datasets import build_dataset

from lib.datasets import (ImageSequenceDataset, ClassBalancedSubsetSampler,
                          DistributedClassBalancedSubsetSampler,
                          DistributedSubsetSampler, CombinedSampler,
                          DistributedReversedSubsetSampler, DASampler)
from lib.models import Classifier, BBNLoss
from lib.utils.dist_utils import collect_results_cpu


def eval(model, dataloader, **kwargs):
    rank = dist.get_rank()
    self = model
    self.eval()
    all_preds = np.empty((0,))
    all_labels = np.empty((0,))
    if rank == 0:
        progress_bar = mmcv.ProgressBar(len(dataloader))
    for data in dataloader:
        imgs = data.pop('imgs')
        labels = data['labels']

        labels = labels.to(next(self.parameters()).device)

        with torch.no_grad():
            preds = self(imgs, **data)

        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
        all_preds = np.hstack([all_preds,
            preds.argmax(1).detach().cpu().numpy()])
        if rank == 0:
            progress_bar.update()
    
    all_preds = collect_results_cpu(
        all_preds, len(val_loader.dataset))
    all_labels = collect_results_cpu(
        all_labels, len(val_loader.dataset))
    if rank == 0:
        cm = confusion_matrix(all_labels, all_preds)
        f1_scores = f1_score(all_labels, all_preds, average=None)
        class_weights = kwargs.get('class_weights', [0.25] * 4)
        f1 = (np.array(class_weights) * f1_scores).sum()
        return dict(f1_scores=f1_scores,
                    f1=f1,
                    cm=cm)
    else:
        return None


def train(self, dataloader, **kwargs):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    optimizer = torch.optim.SGD(
        list(self.parameters()), lr=kwargs.get('lr', 1e-3), 
        momentum=kwargs.get('momentum', 0.9),
        weight_decay=kwargs.get('weight_decay', 0.0005))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, kwargs.get('milestones', [3, ]),
        kwargs.get('gamma', 0.1))
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #    optimizer, kwargs.get('max_epoch', 5) // 3)
    max_epoch = kwargs.get('max_epoch', 5)

    save_dir = kwargs.get('save_dir', 'checkpoints')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logs = defaultdict(list)
    for epoch in range(max_epoch):
        cur_lr = optimizer.param_groups[0]['lr']
        self.train()
        #all_preds = np.empty((0,))
        #all_labels = np.empty((0,))
        dataloader.sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader):
            imgs = data.pop('imgs')
            labels = data['labels']

            labels = labels.to(next(self.parameters()).device)

            losses = self(imgs, **data)
            loss, log_vars = self.module._parse_losses(losses)
            '''
            #acc = (preds.argmax(1) == labels).sum().item() / len(labels)
            all_labels = np.hstack([all_labels, labels.cpu().numpy()])
            all_preds = np.hstack([all_preds,
                preds.argmax(1).detach().cpu().numpy()])
            '''
            if rank == 0 and i % kwargs.get('log_iters', 5) == 0:
                log_str = f' '.join(f'{k}: {v:.4f}'
                                    for k, v in log_vars.items())
                print(f'Epoch {epoch+1}/{max_epoch} '
                      f'Iter: {i+1}/{len(dataloader)} lr: {cur_lr} {log_str}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        '''
        all_preds = collect_results_cpu(all_preds, 10000) # collect all
        all_labels = collect_results_cpu(all_labels, 10000)
        if rank == 0:
            f1_scores = f1_score(all_labels, all_preds, average=None)
            class_weights = kwargs.get('class_weights', [0.25] * 4)
            f1 = (np.array(class_weights) * f1_scores).sum()
            for cat_id in range(len(f1_scores)):
                logs['score_train_%d'%cat_id] += [f1_scores[cat_id]]
            logs['score_train'] += [f1]
        '''

        if kwargs.get('val_dataloader'):
            result = eval(self, kwargs.get('val_dataloader'), **kwargs)
            if rank == 0:
                logs['score'] += [result['f1']]
                logs['cm'] = result['cm']
                f1_scores = result['f1_scores']
                for cat_id in range(len(f1_scores)):
                    logs['score_%d'%cat_id] += [f1_scores[cat_id]]
                torch.save(self.module.state_dict(), os.path.join(
                    save_dir, f'classifier_epoch{epoch+1}.pth'))
        if kwargs.get('test_dataloader'):
            result = eval(self, kwargs.get('test_dataloader'), **kwargs)
            if rank == 0:
                logs['score_test'] += [result['f1']]
                f1_scores = result['f1_scores']
                for cat_id in range(len(f1_scores)):
                    logs['score_%d_test'%cat_id] += [f1_scores[cat_id]]
        lr_scheduler.step()
        if rank == 0:
            print(f'\nEpoch {epoch+1}/{max_epoch} lr: {cur_lr}')
            print('confusion matrix:')
            print(logs.pop('cm'))
            print(pd.DataFrame(logs).transpose())
    if rank == 0:
        best_epoch = np.argmax(logs['score'])
        outputs = {key: value[best_epoch] for key, value in logs.items()}
        outputs['model_name'] = f'classifier_epoch{best_epoch+1}.pth'
        return outputs
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--ann_file', type=str, default='')
    parser.add_argument('--key_frame_only', action='store_true',
                        default=False)
    parser.add_argument('--samples_per_gpu', type=int, default=32)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--milestones', nargs='+', type=int,
                        default=[8, ])
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.local_rank == 0:
        print(cfg.model)

    torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)

    training_set = build_dataset(cfg.data.train)
    val_set = build_dataset(cfg.data.val)

    outputs = []

    if cfg.get('domain_adaption', False):
        indices = np.arange(len(training_set.datasets[0]))
    else:
        indices = np.arange(len(training_set))
        labels = np.array([training_set.get_cat_ids(_) for _ in indices])
        # split train/test
        train_indices, test_indices = next(StratifiedKFold(
            6, shuffle=True, random_state=666).split(indices, labels))
    k = 5
    # cross validation
    kf = StratifiedKFold(k, shuffle=True, random_state=666)
    for idx, (train_inds, val_inds) in enumerate(
        kf.split(train_indices, labels[train_indices])
        ):
        train_inds = train_indices[train_inds] # map to sample index
        val_inds = train_indices[val_inds] # map to sample index
        torch.manual_seed(666)
        torch.cuda.manual_seed_all(666)
        np.random.seed(666)
        random.seed(666)
        bs = args.samples_per_gpu
        samplers = [
            DistributedSubsetSampler(train_inds),
            DistributedReversedSubsetSampler(training_set, train_inds),
            DistributedClassBalancedSubsetSampler(training_set, train_inds),
            #DistributedSubsetSampler(
            #    np.arange(len(training_set.datasets[1]))
            #    + len(training_set.datasets[0])),
            ]
        # specify sampler here to use different long-tail distribution handling
        train_loader = DataLoader(training_set, batch_size=bs, num_workers=4,
            sampler=samplers[2])
        #DASampler(samplers[2:4]))
        # samplers[2])
        #CombinedSampler(samplers[:2])) # for BBN
        val_loader = DataLoader(val_set, batch_size=bs, num_workers=4,
            sampler=DistributedSubsetSampler(val_inds, shuffle=False))
        test_loader = DataLoader(val_set, batch_size=bs, num_workers=4,
            sampler=DistributedSubsetSampler(test_indices, shuffle=False))
        model = build_classifier(cfg.model).to(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True)
        if cfg.load_from:
            load_checkpoint(model, cfg.load_from, map_location='cpu')
        output = train(model, train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            log_iters=10,
            max_epoch=args.max_epoch,
            milestones=args.milestones,
            lr=args.lr,
            class_weights=[0.1, 0.2, 0.3, 0.4],
            save_dir=f'work_dirs/classification/fold{idx + 1}')
        outputs.append(output)
        if dist.get_rank() == 0:
            print(f'{k}-fold cross validation: {idx + 1}.')
            #print output of the last fold
            print(pd.DataFrame(outputs)[-1:].transpose())
    if dist.get_rank() == 0:
        print(pd.DataFrame(outputs)
                .describe()
                .transpose()
                [['mean','std','min','max']])
