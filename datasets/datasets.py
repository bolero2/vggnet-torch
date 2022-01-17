import logging
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(self, datapack, datadir='', category_names=[], imgsz=(224, 224, 3)):
        self.th, self.tw, self.tc = imgsz  # target-height, target-width, target-channel

        self.imgs = datapack[0]
        self.annots = datapack[1]
        self.datadir = datadir
        self.category_names = category_names

    def __getitem__(self, index):
        img = self.imgs[index]
        annot = self.annots[index]
        one_hot_encoded = np.zeros(shape=(len(self.category_names),))
        label_index = self.category_names.index(annot)
        one_hot_encoded[label_index, ] = 1
        
        img = cv2.imread(os.path.join(self.datadir, img), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.th, self.tw))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.

        _img = torch.from_numpy(img).permute(2, 0, 1)
        _label = torch.Tensor(one_hot_encoded)
        return _img, _label

    def __len__(self):
        return len(self.imgs)