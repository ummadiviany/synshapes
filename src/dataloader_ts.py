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
    
    
if __name__ == '__main__':
    
    # C, H, W
    # resize_dims = (get_nearest_multiple(140, 16), get_nearest_multiple(129, 16))
    # print(f'resize_dims: {resize_dims}')
    
    # circle_dataset = LakeDataset('circle_1/train', train=True, time_steps=3)
    # circle_dataloader = DataLoader(circle_dataset, batch_size=2, shuffle=True)
    
    # imgs, labels = next(iter(circle_dataloader))
    # print(f'imgs.shape: {imgs.shape}')
    # print(f'labels.shape: {labels.shape}')
    
    datasets = {
    f"{dataset_dir}/{set_dir}": LakeDataset(dataset_path=f'{dataset_dir}/{set_dir}') \
        for set_dir in ['train', 'test'] \
            for dataset_dir in glob(f'data/*')
    }
    train_loaders = {
        f"{dataset_dir}/train": DataLoader(datasets[f"{dataset_dir}/train"], batch_size=2, shuffle=True) \
            for dataset_dir in glob(f'data/*')
    }
    test_loaders = {
        f"{dataset_dir}/test": DataLoader(datasets[f"{dataset_dir}/test"], batch_size=2, shuffle=False) \
            for dataset_dir in glob(f'data/*')
    }
    
    print(*train_loaders.keys(), sep='\n')
    
    # train_batch = [next(iter(loader)) for loader in test_loaders.values()]
    # img_batch, label_batch = zip(*train_batch)
    # img_batch = torch.cat(img_batch, dim=0)
    # label_batch = torch.cat(label_batch, dim=0)
    # # print(f'train_batch.shape: {train_batch.shape}')
    # print(f'img_batch.shape: {img_batch.shape}')
    # print(f'label_batch.shape: {label_batch.shape}')