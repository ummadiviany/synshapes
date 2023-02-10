import torch, os, time, torch, monai, torchvision, wandb
from torchvision.utils import make_grid
import numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
from skimage.metrics import *
from monai.inferers import sliding_window_inference
import torch.nn.functional as F
from einops import rearrange
torch.manual_seed(2000)


# Timing
start = time.time()

# Hyperparameters
wandb_log = False
epochs = 100
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1

# Dataloader
img_transforms = transforms.Compose(
    [
        # transfroms.Normalize()
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.GaussianBlur(kernel_size=3)
    ]
)
from src.dataloader_ts import LakeDataset, get_nearest_multiple
# Spatial dims : H x W
datasets = {
    f"{dataset_dir}/{set_dir}": LakeDataset(dataset_path=f'{dataset_dir}/{set_dir}', skip=2) \
        for set_dir in ['train', 'test'] \
            for dataset_dir in glob(f'data/*')
    }
train_loaders = {
    f"{dataset_dir}/train": DataLoader(datasets[f"{dataset_dir}/train"], batch_size=1, shuffle=True) \
        for dataset_dir in glob(f'data/*')
}
test_loaders = {
    f"{dataset_dir}/test": DataLoader(datasets[f"{dataset_dir}/test"], batch_size=1, shuffle=True) \
        for dataset_dir in glob(f'data/*')
}
# train_dataset = LakeDataset(dataset_path='data/circle_1/train', skip=2)
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# test_dataset = LakeDataset(dataset_path='data/circle_1/test', skip=2)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)


if wandb_log:
    config = {
        'epochs' : epochs,
        'loss' : 'L1',
        'Augumentations' : None,
        'batch_size' : batch_size,
    }
    wandb.login()
    wandb.init(project="synshapes", config = config)

from src.model_ts import get_model
model = get_model().to(device)


# mse_loss = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()
ce_loss = nn.CrossEntropyLoss()
# from src.loss_utils import apply_ce_loss
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-5)

# Training loop
train_loader_len = len(train_loaders['data\circle_1/train'])
test_loader_len = len(test_loaders['data\circle_1/test'])

for epoch in range(epochs):
    model.train()
    for i in range(train_loader_len):
        optimizer.zero_grad()
        
        train_batch = [next(iter(loader)) for loader in train_loaders.values()]
        img_batch, label_batch = zip(*train_batch)
        imgs = torch.cat(img_batch, dim=0)
        labels = torch.cat(label_batch, dim=0)
        # imgs, labels = next(iter(train_loader))
        torch.cuda.empty_cache()
        outputs = model(imgs.to(device))
        # loss = mse_loss(outputs, labels.to(device))
        loss = mae_loss(outputs.unsqueeze(1), labels.to(device))
        loss.backward()
        optimizer.step()
        # break

        if i % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Step {i+1}/{train_loader_len}, Loss: {loss.item():.4f}')
    # break        

    # predict next image prediction every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        nrmse, psnr, ssim, loss = 0, 0, 0, 0
        test_len = test_loader_len
        
        # img_stack, out_stack, label_stack  = [], [], []

        for i in range(test_loader_len):
            test_batch = [next(iter(loader)) for loader in test_loaders.values()]
            img_batch, label_batch = zip(*test_batch)
            imgs = torch.cat(img_batch, dim=0)
            labels = torch.cat(label_batch, dim=0)
            # imgs, labels = next(iter(test_loader))
            
            torch.cuda.empty_cache()
            outputs = model(imgs.to(device))
            # outputs = sliding_window_inference(inputs=imgs.to(device), roi_size=(160, 160), sw_batch_size=4,
                                                # predictor=model, overlap = 0.5, mode = 'gaussian', device=device)
            outputs=outputs.clip(0,1).unsqueeze(1)
            # print(f'output shape {outputs.shape} and labels shape is {labels.shape}')
            loss += mae_loss(outputs, labels.to(device))
            
            nrmse += normalized_root_mse(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            psnr += peak_signal_noise_ratio(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            ssim += structural_similarity(labels.squeeze().detach().cpu().numpy(), outputs.squeeze().detach().cpu().numpy(), channel_axis=0)
            
            # img_stack.append(imgs), out_stack.append(outputs.cpu()), label_stack.append(labels)

        nrmse /= test_len
        psnr /= test_len
        ssim /= test_len
        print(f'Epoch {epoch+1}/{epochs}, Test Loss: {loss/test_loader_len:.4f}, NRMSE: {nrmse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')

        if wandb_log:
            wandb.log({
                'epoch' : epoch, 'nrmse_n' : nrmse,
                'psnr_n' : psnr,   'ssim_n' : ssim 
            })

        
            # imgs :  1, 1, 3, 416, 384
            # label : B 1 1 H W
            # pred :  B 1 1 H W
            if epoch % 10 == 0:

                img_stack = imgs.cpu()
                label_stack = labels.squeeze(dim=1).cpu()
                out_stack = outputs.squeeze(dim=1).cpu()
                
                
                img0_stack = img_stack[:,:,0]
                img1_stack = img_stack[:,:,1]
                img2_stack = img_stack[:,:,2]
                
                f = make_grid(
                    torch.cat(
                    [img0_stack, img1_stack, img2_stack, label_stack, out_stack], dim=3
                    ), nrow=1, padding=15, pad_value=1
                )
                images = wandb.Image(f, caption="First three : Input, Fourth : Ground Truth, Last: Prediction")
                wandb.log({f"Test Predictions": images, "epoch" : epoch})
                print(f'Logged predictions to wandb')
        #   break
    
    # break
    scheduler.step()


# Save the model checkpoint 
model_name = 'unet_sawa_patch_sli'
# torch.save(model.state_dict(), f'artifacts/models/{model_name}.pth')

# Timing
print(f'Time elapsed: {time.time() - start:.2f} seconds')