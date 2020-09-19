import os

from PIL import Image
import numpy as np
import torch
import mmcv
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class ImageSequenceDataset(Dataset):

    img_root = '../data/amap_traffic_%s_0712/'
    ann_file = '../data/amap_traffic_annotations_%s.json'
    img_root = '../data/amap_traffic_b_%s_0828'
    ann_file = '../data/amap_traffic_annotations_b_%s_0828.json'

    def __init__(self,
                 img_root=None,
                 ann_file=None,
                 split='train',
                 transform=None,
                 **kwargs):
        self.img_root = img_root if img_root else self.img_root % split
        self.ann_file = ann_file if ann_file else self.ann_file % split
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
        self.anns = mmcv.load(self.ann_file)['annotations']
        for ann in self.anns:
            ann['frames'].sort(key=lambda x:x['frame_name'])

    def get_cat_ids(self, idx):
        return self.anns[idx]['status']

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
                assert int(frame['frame_name'][0]) == i + 1, \
                    f'{frame["frame_name"]} is not {i+1}th frame'
                if frame['frame_name'] == ann['key_frame']:
                    ann['key'] = i
                img = Image.open(os.path.join(self.img_root, 
                    ann['id'], frame['frame_name']))
                img = np.array(img)
            else:
                img = np.zeros(imgs[0].shape[-2:] + (3,))
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
