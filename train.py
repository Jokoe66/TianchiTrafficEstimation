import os
from collections import defaultdict
import argparse
import sys
sys.path.insert(0, 'lib/mmdetection')

import pandas as pd
import numpy as np
import mmcv
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from mmdet.apis.test import collect_results_cpu

from lib.datasets import (ImageSequenceDataset, ClassBalancedSubsetSampler,
    DistributedClassBalancedSubsetSampler, DistributedSubsetSampler)
from lib.models import Classifier


def eval(model, dataloader, **kwargs):
    rank = dist.get_rank()
    self = model
    self.eval()
    all_preds = np.empty((0,))
    all_labels = np.empty((0,))
    if rank == 0:
        progress_bar = mmcv.ProgressBar(len(dataloader))
    for data in dataloader:
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
        if rank == 0:
            progress_bar.update()
    
    all_preds = collect_results_cpu(
        all_preds, len(val_loader.dataset))
    all_labels = collect_results_cpu(
        all_labels, len(val_loader.dataset))
    if rank == 0:
        f1_scores = f1_score(all_labels, all_preds, average=None)
        class_weights = kwargs.get('class_weights', [0.25] * 4)
        f1 = (np.array(class_weights) * f1_scores).sum()
        return dict(f1_scores=f1_scores,
                    f1=f1)
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
    # re-weighting for balancing class distribution
    #class_weights = torch.tensor(
    #    [0.1, 0.2, 0.3, 0.4], device=next(self.parameters().device)
    criteria = torch.nn.CrossEntropyLoss()

    max_epoch = kwargs.get('max_epoch', 5)
    save_dir = kwargs.get('save_dir', 'checkpoints')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logs = defaultdict(list)
    for epoch in range(max_epoch):
        cur_lr = optimizer.param_groups[0]['lr']
        self.train()
        all_preds = np.empty((0,))
        all_labels = np.empty((0,))
        for i, data in enumerate(dataloader):
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

            preds = self(imgs, seq_len)
            loss = criteria(preds, labels)
            acc = (preds.argmax(1) == labels).sum().item() / len(labels)
            all_labels = np.hstack([all_labels, labels.cpu().numpy()])
            all_preds = np.hstack([all_preds,
                preds.argmax(1).detach().cpu().numpy()])
            if rank == 0 and i % kwargs.get('log_iters', 5) == 0:
                print(f'Epoch {epoch+1}/{max_epoch} '
                      f'Iter: {i+1}/{len(dataloader)} '
                      f'lr: {cur_lr} loss: {loss.item()} acc: {acc}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        all_preds = collect_results_cpu(all_preds, 10000) # collect all
        all_labels = collect_results_cpu(all_labels, 10000)
        if rank == 0:
            f1_scores = f1_score(all_labels, all_preds, average=None)
            class_weights = kwargs.get('class_weights', [0.25] * 4)
            f1 = (np.array(class_weights) * f1_scores).sum()
            for cat_id in range(len(f1_scores)):
                logs['score_train_%d'%cat_id] += [f1_scores[cat_id]]
            logs['score_train'] += [f1]

        if kwargs.get('val_dataloader'):
            result = eval(self, kwargs.get('val_dataloader'), **kwargs)
            if rank == 0:
                logs['score'] += [result['f1']]
                f1_scores = result['f1_scores']
                for cat_id in range(len(f1_scores)):
                    logs['score_%d'%cat_id] += [f1_scores[cat_id]]
                torch.save(self.module.state_dict(), os.path.join(
                    save_dir, f'classifier_epoch{epoch+1}.pth'))
        lr_scheduler.step()
        if rank == 0:
            print(f'Epoch {epoch+1}/{max_epoch} lr: {cur_lr}')
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
    parser.add_argument('--img_root', type=str, default='')
    parser.add_argument('--ann_file', type=str, default='')
    parser.add_argument('--key_frame_only', action='store_true',
                        default=False)
    parser.add_argument('--distributed', action='store_true',
                        default=False)
    parser.add_argument('--samples_per_gpu', type=int, default=32)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--milestones', nargs='+', type=int,
                        default=[8, ])
    args = parser.parse_args()
    if args.distributed:
        torch.distributed.init_process_group('nccl')
        torch.cuda.set_device(args.local_rank)

    lstm = None if args.key_frame_only else 128

    training_set = ImageSequenceDataset(
        args.img_root,
        args.ann_file,
        'train',
        key_frame_only=args.key_frame_only,
        input_size=(640, 360)) # 0.5x input_size for better efficiency

    outputs = []

    indices = np.arange(len(training_set))
    k = 5
    kf = KFold(k, shuffle=True, random_state=666)
    for idx, (train_inds, val_inds) in enumerate(kf.split(indices)):
        bs = args.samples_per_gpu
        train_loader = DataLoader(training_set, batch_size=bs, num_workers=4,
            sampler=DistributedClassBalancedSubsetSampler(training_set, train_inds))
        val_loader = DataLoader(training_set, batch_size=bs, num_workers=4,
            sampler=DistributedSubsetSampler(val_inds))

        backbone = dict(
            type='Res2Net',
            depth=101,
            scales=4,
            base_width=26,
            frozen_stages=4,
            out_indices=(3,),
            norm_eval=False)
        backbone = dict(
            type='ResNet',
            depth=101,
            num_stages=4,
            out_indices=(3,),
            frozen_stages=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch')
        model = Classifier(backbone,
            #pretrained='open-mmlab://res2net101_v1d_26w_4s',
            pretrained='torchvision://resnet101',
            num_classes=4, lstm=lstm).to(args.local_rank)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank])
        output = train(model, train_loader,
            val_dataloader=val_loader,
            log_iters=10,
            max_epoch=args.max_epoch,
            milestones=args.milestones,
            lr=args.lr,
            class_weights=[0.1, 0.2, 0.3, 0.4],
            save_dir=f'work_dirs/classification/fold{idx + 1}')
        outputs.append(output)
        if dist.get_rank() == 0:
            print(f'{k}-fold cross validation: {idx + 1}.')
            print(pd.DataFrame(outputs)[-1:].transpose())
    if dist.get_rank() == 0:
        print(pd.DataFrame(outputs)
                .describe()
                .transpose()
                [['mean','std','min','max']])
