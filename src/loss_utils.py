from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
ce_loss = nn.CrossEntropyLoss()

def apply_ce_loss(outputs, labels):
    """_summary_

    Args:
        outputs (_type_): Shape (batch_size, 1, height, width) torch.rand(8, 1, 256, 256)
        labels (_type_): Shape (batch_size, 1, 1, height, width) torch.ones(8, 1, 1, 256, 256)

    """
    
    th = 0.5
    outputs = (outputs > th).float()
    # print((f'Outputs unique values: {torch.unique(outputs)}'))
    # Convert outputs to 2 classes (0, 1)
    outputs = F.one_hot(outputs.squeeze().long(), num_classes=2)
    outputs = rearrange(outputs, 'b h w c -> b c h w')
    labels = labels.squeeze().long()
    # print(f'Outputs shape: {outputs.shape}, Labels shape: {labels.shape}')
    loss = ce_loss(outputs.float(), labels)
    # print(f'Loss: {loss.item():.4f}')
    
    return loss