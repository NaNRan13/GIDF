import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import glob
import random
from PIL import Image
import tqdm

from utils import make_coord

# reference: icgNoiseLocalvar (https://github.com/griegler/primal-dual-networks/blob/master/common/icgcunn/IcgNoise.cu)
def add_noise(x, k=1, sigma=651, inv=True):                                                                                                                                                                 
    # x: [H, W, 1                                                                                                                                                                                         
    noise = sigma * np.random.randn(*x.shape)                                                                                                                                                               
    if inv:                                                                                                                                                                                                 
        noise = noise / (x + 1e-5)                                                                                                                                                                          
    else:                                                                                                                                                                                                   
        noise = noise * x                                                                                                                                                                                   
    x = x + k * noise                                                                                                                                                                                       
    return x    


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )

def to_pixel_samples(depth):
    """ Convert the image to coord-RGB pairs.
        depth: Tensor, (1, H, W)
    """
    coord = make_coord(depth.shape[-2:], flatten=True).view(depth.shape[-2], depth.shape[-1], 2) # [H，W, 2]
    
    pixel = depth.view(-1, 1) # [H*W, 1]
    
    return coord, pixel

class LIT_MiddleburyDataset(Dataset):
    def __init__(self, root='data/Middlebury/', split='train', 
                 batch_size=4, scale=8, scale_max=16, augment=True, downsample='bicubic', 
                 pre_upsample=False, to_pixel=False, sample_q=None, input_size=None, window_size=8, noisy=False, if_AR = True):
        super().__init__()
        self.root = root
        self.split = split
        self.init_scale = scale
        self.scale = scale
        self.augment = augment
        self.downsample = downsample
        self.pre_upsample = pre_upsample
        self.to_pixel = to_pixel
        self.sample_q = sample_q
        self.input_size = input_size
        self.window_size = window_size
        self.noisy = noisy
        self.batch_size = batch_size
        self.if_AR = if_AR
        self.scale_min = 1.0
        self.scale_max = scale_max

        # Test dataset
        self.repeat = 1

        if self.noisy:
            print("==> noisy <==")

        if self.if_AR:
            print("================Use ARBITRARY dataloader DSF_NYUDataset================")
            print("the sclae factor id from ", self.scale_min, ' to ', self.scale_max)
        else:
            print("================Fixed scaling factor = ", self.scale, "================")
    
        if self.split == 'train':
            raise AttributeError('Middlebury dataset only support test mode.')
        else:
            self.image_files = sorted(glob.glob(os.path.join(root, 'RGB', '*'))) 
            self.depth_files = sorted(glob.glob(os.path.join(root, 'Depth', '*'))) 
            assert len(self.image_files) == len(self.depth_files)
            self.size = len(self.image_files)
            
        print("========Use middlebury dataset=======")
    
    def collate_fn(self, datas):
        image_hr_list = []
        depth_hr_list = []
        depth_lr_list = []
        depth_min_list = []
        depth_max_list = []
        idx_list = []
            
        for idx, data in enumerate(datas):
            idx_list.append(torch.tensor(data['idx']))
            image_hr = data['image']
            depth_hr = data['hr_depth']
            
            # norm
            # In geodsr and jiif dataloader
            # depth_hr suffer norm, crop(impossible lu and middlebury is test dataset), resize
            # Maybe it's because the input is an entire image
            depth_min = depth_hr.min()
            depth_max = depth_hr.max()
            depth_hr = (depth_hr - depth_min) / (depth_max - depth_min)
            
            # crop to make divisible
            h, w = image_hr.shape[:2]
            # h = h - h % self.scale
            # w = w - w % self.scale
            h = int(h // self.scale * self.scale)
            w = int(w // self.scale * self.scale)
            image_hr = image_hr[:h, :w]
            depth_hr = depth_hr[:h, :w]
            
            # crop after rescale
            if self.input_size is not None:
                x0 = random.randint(0, image_hr.shape[0] - self.input_size)
                y0 = random.randint(0, image_hr.shape[1] - self.input_size)
                image_hr = image_hr[x0:x0+self.input_size, y0:y0+self.input_size]
                depth_hr = depth_hr[x0:x0+self.input_size, y0:y0+self.input_size]
        
            h, w = image_hr.shape[:2]
            
            # 是否存在其他下采样可能，比如jiif其他的下采样方法
            depth_lr = np.array(Image.fromarray(depth_hr).resize((int(w//self.scale), int(h//self.scale)), Image.BICUBIC)) 

            if self.noisy:
                # print(depth_lr.min(), depth_lr.max())
                depth_lr = add_noise(depth_lr, sigma=0.04, inv=False)

            image_hr = image_hr.astype(np.float32).transpose(2,0,1) / 255
            image_hr = (image_hr- np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

            # to tensor
            image_hr = torch.from_numpy(image_hr).float()
            depth_hr = torch.from_numpy(depth_hr).unsqueeze(0).float()
            depth_lr = torch.from_numpy(depth_lr).unsqueeze(0).float()
            depth_max = torch.tensor(depth_max)
            depth_min = torch.tensor(depth_min)
               
            image_hr_list.append(image_hr)
            depth_hr_list.append(depth_hr)
            depth_lr_list.append(depth_lr)
            depth_min_list.append(depth_min)
            depth_max_list.append(depth_max) 

        image = torch.stack(image_hr_list, dim=0)
        depth_hr = torch.stack(depth_hr_list, dim=0)
        depth_lr = torch.stack(depth_lr_list, dim=0)
        depth_min = torch.stack(depth_min_list, dim=0)
        depth_max = torch.stack(depth_max_list, dim=0)
        idx_ = torch.stack(idx_list, dim=0)

        cell = torch.ones(2)
        cell[0] *= 2. / depth_hr.shape[-2]
        cell[1] *= 2. / depth_hr.shape[-1]
        cell = cell.unsqueeze(0).repeat(self.batch_size, 1)

        lr_cell = torch.ones(2)
        lr_cell[0] *= 2. / depth_lr.shape[-2]
        lr_cell[1] *= 2. / depth_lr.shape[-1]
        lr_cell = lr_cell.unsqueeze(0).repeat(self.batch_size, 1)

        img_h, img_w = image.shape[-2:]
        lr_h, lr_w = depth_lr.shape[-2:]

        hr_coord = make_coord((img_h, img_w), flatten=False) \
                            .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        lr_coord = make_coord((lr_h, lr_w), flatten=False) \
                            .unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        
        hr_pixel = depth_hr.reshape(self.batch_size, -1, 1)

        if self.sample_q is None:
            sample_coord = hr_coord.reshape(self.batch_size,-1,2)
            sample_pixel = []
        else:
            sample_coord = []
            sample_pixel = []
            for i in range(len(depth_hr_list)):
                flatten_coord = hr_coord[i].reshape(-1, 2)      #   (q,2)
                sample_list = np.random.choice(flatten_coord.shape[0], self.sample_q, replace=False)
                sample_flatten_coord = flatten_coord[sample_list, :]    #  (q_sample,2)
                sample_coord.append(sample_flatten_coord) 
                flatten_hrpixel = hr_pixel[i]   
                sample_flatten_hrpixel = flatten_hrpixel[sample_list, :]
                sample_pixel.append(sample_flatten_hrpixel)   
            sample_coord = torch.stack(sample_coord, dim=0)       #sample_coord (B,q_sample,2)
            sample_pixel = torch.stack(sample_pixel)
            
        # hr_pixel is gt for train, hr_depth is gt for eval and test
        return {
                'hr_image': image,  
                'hr_depth': depth_hr,
                'lr_depth': depth_lr,
                'hr_coord': hr_coord,   
                'lr_coord': lr_coord,
                'sample_coord': sample_coord,
                'hr_pixel': sample_pixel,
                'cell': cell,
                'lr_cell': lr_cell,
                'scale': self.scale,
                'min': depth_min,  
                'max': depth_max,
                'idx': idx_,
            }   


    def __getitem__(self, idx):

        image_file = self.image_files[idx]
        depth_file = self.depth_files[idx]             

        image = cv2.imread(image_file).astype(np.uint8) # [H, W, 3]
        depth_hr = cv2.imread(depth_file)[:,:,0].astype(np.float32) # [H, W]


        return {
            'image': image,
            'hr_depth': depth_hr,
            'idx': idx,
        }   


    def __len__(self):
        return self.size

