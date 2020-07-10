import os
import tqdm
import torch
import mmcv
from mmcv.cnn import ResNet

class Classifier(torch.nn.Module):

    def __init__(self, num_classes=3, **kwargs):
        super(Classifier, self).__init__()
        self.num_classes = num_classes

        net = ResNet(50, frozen_stages=4, style='pytorch', out_indices=(3,))
        net.init_weights('torchvision://resnet50')
        for p in net.parameters():
            p.requires_grad = False
        net.eval()
        self.feat = net
        self.avg = torch.nn.AdaptiveAvgPool2d((1,1))
        
        hidden_size = kwargs.get('lstm')
        self.lstm = torch.nn.GRU(2048, hidden_size) if hidden_size else None
        hidden_size = hidden_size or 2048
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input, seq_len=5):
        with torch.no_grad():
            feat = self.feat(input)
        
        feat = self.avg(feat).view(len(input), -1)
        if self.lstm:
            feat = feat.view(seq_len, len(feat)//seq_len, -1)
            feat, h = self.lstm(feat)
            feat = feat.mean(0)
        
        logit = self.fc(feat)
        return logit 

    def fit(self, dataloader, **kwargs):
        optimizer = torch.optim.SGD(
            list(self.fc.parameters()), lr=kwargs.get('lr', 1e-3), 
            momentum=kwargs.get('momentum', 0.9))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, kwargs.get('milestones', [3, ]), kwargs.get('gamma', 0.1))
        criteria = torch.nn.CrossEntropyLoss()

        max_epoch = kwargs.get('max_epoch', 4)
        best_score = 0
        save_dir = kwargs.get('save_dir', 'checkpoints')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for epoch in range(max_epoch):
            cur_lr = optimizer.param_groups[0]['lr']
            self.train()
            for i, data in enumerate(dataloader):
                imgs = data['imgs']
                if len(imgs.shape) > 4: # frames
                    seq_len = imgs.shape[-1]
                    imgs = (imgs.permute(4, 0, 1, 2, 3).contiguous()
                            .view(-1, *imgs.shape[1:4]))
                else:
                    seq_len = 1
                labels = data['label']

                imgs = imgs.to(self.fc.weight.device)
                labels = labels.to(self.fc.weight.device)

                preds = self(imgs, seq_len)
                loss = criteria(preds, labels)
                acc = (preds.argmax(1) == labels).sum().item() / len(labels)
                if i % kwargs.get('log_iters', 5) == 0:
                    print(f'Epoch {epoch+1}/{max_epoch} Iter: {i+1}/{len(dataloader)} '
                          f'lr: {cur_lr} loss: {loss.item()} acc: {acc}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if kwargs.get('val_dataloader'):
                self.eval()
                ns = [0] * 3
                tps = [0] * 3
                fps = [0] * 3
                ps = [0] * 3
                rs = [0] * 3
                f1s = [0] * 3
                for data in tqdm.tqdm(kwargs.get('val_dataloader')):
                    imgs = data['imgs']
                    if len(imgs.shape) > 4: # frames
                        seq_len = imgs.shape[-1]
                        imgs = (imgs.permute(4, 0, 1, 2, 3).contiguous()
                                .view(-1, *imgs.shape[1:4]))
                    else:
                        seq_len = 1
                    labels = data['label']

                    imgs = imgs.to(self.fc.weight.device)
                    labels = labels.to(self.fc.weight.device)

                    with torch.no_grad():
                        preds = self(imgs, seq_len)

                    for cls in range(len(tps)):
                        ns[cls] += (labels == cls).sum().item()
                        tps[cls] += ((labels == cls) 
                                     & (preds.argmax(1) == cls)).sum().item()
                        fps[cls] += ((labels != cls) 
                                     & (preds.argmax(1) == cls)).sum().item()
                eps = 1e-5
                for cls in range(len(tps)):
                    ps[cls] = tps[cls] / (tps[cls] + fps[cls] + eps)
                    rs[cls] = tps[cls] / (ns[cls] + eps)
                    f1s[cls] = 2 * ps[cls] * rs[cls] / (ps[cls] + rs[cls] + eps)
                class_weights = kwargs.get('class_weights', [0.2, 0.2, 0.6])
                score = sum(class_weights[cls] * f1s[cls] for cls in range(len(tps)))
                print(f'Epoch {epoch+1}/{max_epoch} lr: {cur_lr} '
                      f'P: {ps} R: {rs} '
                      f'F1_score: {f1s} score: {score}')

                torch.save(self.state_dict(), os.path.join(
                    save_dir, f'classifier_epoch{epoch+1}.pth'))
            lr_scheduler.step()
