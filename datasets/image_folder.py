import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import cv2
import torch
import glob
from torch.utils.data import Dataset
from torchvision import transforms

# just for nyu load data in memory
class ImageFolder(Dataset):
    def __init__(self, size, start_idx=0, root_path='./data/nyu_labeled/'):

        filepath_rgb = os.path.join(root_path, 'RGB')
        filepath_depth = os.path.join(root_path, 'Depth')
        if len(glob.glob(os.path.join(filepath_rgb, '*'))) != len(glob.glob(os.path.join(filepath_depth, '*'))):
            raise NotImplementedError(f'The number of datasets does not match!')
        self.file_rgb = []
        self.file_depth = []
        self.size = size
        self.start_idx = start_idx
        for idx in range(self.start_idx, self.start_idx+self.size):
            path_rgb = os.path.join(filepath_rgb, f'{idx}.jpg')
            path_depth = os.path.join(filepath_depth, f'{idx}.npy')

            self.file_rgb.append(cv2.imread(path_rgb))
            self.file_depth.append(np.load(path_depth))
            
        print(len(self.file_rgb), len(self.file_depth))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.file_rgb[idx % self.size]
        y = self.file_depth[idx % self.size]

        return x, y
