import os
from collections import defaultdict

import pandas as pd
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import mmcv
from mmcv.cnn import ResNet
from sklearn.metrics import f1_score

class Classifier(torch.nn.Module):

    def __init__(self, num_classes=3, **kwargs):
        super(Classifier, self).__init__()
        self.num_classes = num_classes

        net = ResNet(101, frozen_stages=4, style='pytorch', out_indices=(3,))
        net.init_weights('torchvision://resnet101')
        self.feat = net

        h, w, c = 9, 16, 256 # make feat_size indpendent on input_size
        self.pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((h, w)),
            torch.nn.Conv2d(2048, c, 1, 1, 0),
            torch.nn.ReLU(inplace=True)
        )

        hidden_size = kwargs.get('lstm')
        self.lstm = torch.nn.GRU(h * w * c , hidden_size) if hidden_size else None
        hidden_size = hidden_size or (h * w * c)

        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.00),
            torch.nn.Linear(hidden_size, num_classes))

    def forward(self, input, seq_len=5):
        #with torch.no_grad():
        feat = self.feat(input)
        
        feat = self.pool(feat)
        n, c, h, w = feat.shape
        feat = feat.view(n, -1)
        if self.lstm:
            feat = feat.view(seq_len, len(feat)//seq_len, -1)
            feat, h = self.lstm(feat)
            feat = feat.mean(0)
        
        logit = self.fc(feat)
        return logit 

    def fit(self, dataloader, **kwargs):
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
        class_weights = kwargs.get('class_weights', [0.25] * 4)
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
                if i % kwargs.get('log_iters', 5) == 0:
                    print(f'Epoch {epoch+1}/{max_epoch} Iter: {i+1}/{len(dataloader)} '
                          f'lr: {cur_lr} loss: {loss.item()} acc: {acc}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            f1_scores = f1_score(all_labels, all_preds, average=None)
            f1 = (np.array(class_weights) * f1_scores).sum()
            for cat_id in range(len(f1_scores)):
                logs['score_train_%d'%cat_id] += [f1_scores[cat_id]]
            logs['score_train'] += [f1]

            if kwargs.get('val_dataloader'):
                self.eval()
                all_preds = np.empty((0,))
                all_labels = np.empty((0,))
                for data in tqdm.tqdm(kwargs.get('val_dataloader')):
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
                f1 = (np.array(class_weights) * f1_scores).sum()
                logs['score'] += [f1]
                for cat_id in range(len(f1_scores)):
                    logs['score_%d'%cat_id] += [f1_scores[cat_id]]
                torch.save(self.state_dict(), os.path.join(
                    save_dir, f'classifier_epoch{epoch+1}.pth'))
            lr_scheduler.step()
            print(f'Epoch {epoch+1}/{max_epoch} lr: {cur_lr}')
            print(pd.DataFrame(logs).transpose())
        best_epoch = np.argmax(logs['score'])
        outputs = {key: value[best_epoch] for key, value in logs.items()}
        outputs['model_name'] = f'classifier_epoch{best_epoch+1}.pth'
        return outputs
