import torch
import torch.nn as nn

import kornia.metrics as metrics


def psnr_loss(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    r"""Function that computes the PSNR loss.

    The loss is computed as follows:

     .. math::

        \text{loss} = -\text{psnr(x, y)}

    See :meth:`~kornia.losses.psnr` for details abut PSNR.

    Args:
        input: the input image with shape :math:`(*)`.
        labels : the labels image with shape :math:`(*)`.
        max_val: The maximum value in the input tensor.

    Return:
        the computed loss as a scalar.

    Examples:
        >>> ones = torch.ones(1)
        >>> psnr_loss(ones, 1.2 * ones, 2.) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(-20.0000)
    """
    return -1.0 * metrics.psnr(input, target, max_val)


class PSNRLoss(nn.Module):
    r"""Create a criterion that calculates the PSNR loss.

    The loss is computed as follows:

     .. math::

        \text{loss} = -\text{psnr(x, y)}

    See :meth:`~kornia.losses.psnr` for details abut PSNR.

    Args:
        max_val: The maximum value in the input tensor.

    Shape:
        - Input: arbitrary dimensional tensor :math:`(*)`.
        - Target: arbitrary dimensional tensor :math:`(*)` same shape as input.
        - Output: a scalar.

    Examples:
        >>> ones = torch.ones(1)
        >>> criterion = PSNRLoss(2.)
        >>> criterion(ones, 1.2 * ones) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(-20.0000)
    """

    def __init__(self, max_val: float) -> None:
        super().__init__()
        self.max_val: float = max_val

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return psnr_loss(input, target, self.max_val)
    
    
if __name__ == "__main__":
    pred = torch.rand(1, 1, 256, 256)
    label = torch.randint_like(pred, 0, 2)
    print(f'Pred Min : {pred.min()}, Max: {pred.max()}')
    print(f'Label Min: {label.min()}, Max: {label.max()}')
    psnr_loss_func = PSNRLoss(1.0)
    loss_value = psnr_loss_func(pred, label)
    print(f'PSNR loss: {loss_value}')
    