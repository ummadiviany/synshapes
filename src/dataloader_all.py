import torch, time, torch, torchvision, monai
import numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
torch.manual_seed(2000)
from monai.transforms import RandSpatialCrop, ToTensor, AddChannel
import torchvision.transforms.functional as TF
from einops import rearrange

# get nearest multiple of given number
def get_nearest_multiple(num:int, multiple:int) -> int:
    upper = int(np.ceil(num / multiple) * multiple)
    lower = int(np.floor(num / multiple) * multiple)
    
    return upper if (upper - num) <= (num - lower) else lower

class LakeDataset(Dataset):
    def __init__(self, dataset_path, img_transforms=None, label_transforms=None, resize_dims=None, train=True, time_steps=3, skip=1):
        
        self.img_transforms = img_transforms
        self.time_steps = time_steps
        self.label_transfroms = label_transforms
        self.resize_dims = resize_dims
        self.train = train
        self.skip = skip
        self.files = sorted(glob(f'{dataset_path}/*.png'))
        print(f'Loaded {len(self.files)} images from {dataset_path} dataset')
        
    def __len__(self):
        return len(self.files) - (self.time_steps + 1) * self.skip
    
    

    def __getitem__(self, idx):
        img_files = self.files[idx:idx+self.time_steps*self.skip:self.skip]
        images = [
            torchvision.io.read_image(file, mode=torchvision.io.ImageReadMode.GRAY) for file in img_files
        ]
        images = torch.stack(images, dim=0)
        label = torchvision.io.read_image(self.files[idx+self.time_steps*self.skip], mode=torchvision.io.ImageReadMode.GRAY)

        images = rearrange(images, 't c h w -> c t h w')
        label = rearrange(label, 'c h w -> c 1 h w')
            
        # print('-'*50)
        # print(f'images.shape: {images.shape}')
        # print(f'label.shape: {label.shape}')
        # print('-'*50)
        # B C T H W
        # B C 1 H W
        return images/255, label/255