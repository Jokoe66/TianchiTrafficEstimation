import json
import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

class ImageSequenceDataset(Dataset):

    img_root = 'data/amap_traffic_%s/'
    ann_file = 'data/amap_traffic_annotations_%s.json'

    def __init__(self, split='train', seq_max_len=5, transform=None):
        self.img_root = self.img_root % split
        self.ann_file = self.ann_file % split
        self._load_anns()
        if transform:
            self.transform = transform
        else:
            self.transform = Compose([
                Resize(size=(360, 640)),
                ToTensor(),
                Normalize(mean=[123.675, 116.28, 103.53],
                          std=[58.395, 57.12, 57.375]),
            ])
        self.seq_max_len = seq_max_len
        
    def _load_anns(self):
        with open(self.ann_file) as f:
            self.anns = json.load(f)['annotations']
    
    def __getitem__(self, idx):
        ann = self.anns[idx]
        ann['imgs'] = []
        for i in range(self.seq_max_len):
            if i < len(ann['frames']):
                frame = ann['frames'][i]
                if frame['frame_name'] == ann['key_frame']:
                    ann['key'] = i
                img = Image.open(os.path.join(self.img_root, 
                    ann['id'], frame['frame_name']))
                if self.transform:
                    img = self.transform(img)
            else:
                img = torch.zeros(ann['imgs'][0].shape)
            ann['imgs'].append(img)
        ann['imgs'] = torch.stack(ann['imgs'], -1)

        return dict(imgs=ann['imgs'],
                    key=ann['key'],
                    len_seq=len(ann['frames']),
                    label=ann['status'])

    def __len__(self):
        return len(self.anns)
