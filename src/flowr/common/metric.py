"""Implementations of evaluation metrics."""

import torch
from torch import Tensor
from torch.nn import functional as F


def psnr(image_gt: Tensor, image_pred: Tensor, reduction: str = "mean", eps: float = 1e-8) -> Tensor:
    """Compute PSNR between two images.

    Args:
        image_gt: Ground truth image.
        image_pred: Predicted image.
        reduction: Reduction type ('mean' or 'none').
        eps: Small value to avoid log(0).

    Returns:
        PSNR value(s).
    """
    return -10 * torch.log10(F.mse_loss(image_pred, image_gt, reduction=reduction).clamp(min=eps))
