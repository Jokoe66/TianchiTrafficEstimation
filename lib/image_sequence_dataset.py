import json
import os

from PIL import Image
import numpy as np
import torch
import mmcv
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

class ImageSequenceDataset(Dataset):

    img_root = 'data/amap_traffic_%s_0712/'
    ann_file = 'data/amap_traffic_annotations_%s.json'

    def __init__(self, split='train', transform=None, **kwargs):
        self.img_root = self.img_root % split
        self.ann_file = self.ann_file % split
        self._load_anns()
        self.img_norm = dict(mean=[123.675, 116.28, 103.53],
                             std=[58.395, 57.12, 57.375])
        if transform:
            self.transform = transform
        else:
            input_size = kwargs.get('input_size', (1280, 720))
            self.transform = Compose([
                lambda x: mmcv.imresize(x, input_size),
                ToTensor(),
                Normalize(**self.img_norm)
            ])
        self.seq_max_len = kwargs.get('seq_max_len', 5)
        self.key_frame_only = kwargs.get('key_frame_only', False)
        
    def _load_anns(self):
        with open(self.ann_file) as f:
            self.anns = json.load(f)['annotations']
    
    def __getitem__(self, idx):
        ann = self.anns[idx]
        if self.key_frame_only:
            img = Image.open(os.path.join(self.img_root,
                    ann['id'], ann['key_frame']))
            img = np.array(img)
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(img)
            return dict(imgs=img,
                        key=0,
                        len_seq=1,
                        seq_len=1,
                        label=ann['status'])

        imgs = []
        for i in range(self.seq_max_len):
            if i < len(ann['frames']):
                frame = ann['frames'][i]
                if frame['frame_name'] == ann['key_frame']:
                    ann['key'] = i
                img = Image.open(os.path.join(self.img_root, 
                    ann['id'], frame['frame_name']))
                img = np.array(img)
            else:
                img = np.zeros_like(np.array(imgs[0]))
                img = img + self.img_norm['mean']
                img = img.astype('uint8')
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(img)
            imgs.append(img)
        imgs = torch.stack(imgs, -1)

        return dict(imgs=imgs,
                    key=ann['key'],
                    len_seq=len(ann['frames']),
                    seq_len=len(ann['frames']),
                    label=ann['status'])

    def __len__(self):
        return len(self.anns)
