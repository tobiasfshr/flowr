"""Data transformations.

Applying data transformations is useful when you need to, e.g., crop images to a certain size,
however keep in mind that dynamically manipulating input samples restricts you from refining
intrinsic camera parameters due to dynamically changing intrinsics.
"""
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class DataKeys:
    """Class that holds the default data keys."""

    IDX = "image_idx"
    PATH = "image_path"
    IMAGE = "image"
    CAM = "camera"
    MASK = "mask"
    DEPTH = "depth"


@dataclass
class Transform(ABC):
    """Base Tranformation class.

    This class implements a simple interface for manipulating a data sample
    with some data transformation, e.g. resizing or cropping. It modififes
    the data in-place to avoid unnecessary memory overhead.
    """

    @abstractmethod
    def __call__(self, data: Dict) -> None:
        raise NotImplementedError


@dataclass
class HWCtoCHW(Transform):
    """Transform images from HWC to CHW. Assumes that data contains an image tensor.

    Note: This should be applied after all other transformations, because we assume
    the images to be HWC by default for all transformations."""

    def __call__(self, data: Dict) -> None:
        if DataKeys.IMAGE in data:
            data[DataKeys.IMAGE] = data[DataKeys.IMAGE].permute(2, 0, 1)


@dataclass
class Resize(Transform):
    """Resize transformation."""

    size: int = 0
    """Target size to scale all images to while keeping aspect ratio."""
    short_edge: bool = True
    """If the target size is w.r.t. the short or long edge."""
    randomize: bool = False
    """If True, randomize the resizing between the original and the target size."""

    def _interpolate(
        self, tensor: Tensor, size: Tuple[int, int], mode: Literal["nearest", "bilinear", "bicubic"] = "bilinear"
    ) -> Tensor:
        result = F.interpolate(
            tensor.permute(2, 0, 1)[None], size=size, mode=mode, antialias=mode in ["bilinear", "bicubic"]
        )
        # clamp for numerical stability --> strictly keeping [0.0, 1.0] interval
        return result.squeeze(0).permute(1, 2, 0).contiguous().clamp(0.0, 1.0)

    def __call__(self, data: Dict) -> None:
        """Resize data."""
        if DataKeys.IMAGE in data:
            im_h, im_w = data[DataKeys.IMAGE].shape[:2]
        elif DataKeys.CAM in data:
            im_h, im_w = data[DataKeys.CAM].height.item(), data[DataKeys.CAM].width.item()
        else:
            raise RuntimeError(f"Invalid data format: {data.keys()}. Expected to contain image or camera")

        # Align numerically with 'rescale_output_resolution' in camera (do all computation in float32, add eps for numerical stability)
        im_size = torch.tensor([im_h, im_w])
        tgt_size = torch.tensor([self.size, self.size])
        scale_factor = (tgt_size / im_size + 1e-6).max() if self.short_edge else (tgt_size / im_size + 1e-6).min()

        if self.randomize:
            p = random.random()
            scale_factor = (p * (1 - scale_factor**3) + scale_factor**3) ** (1 / 3.0)

        tgt_size = tuple((im_size * scale_factor).int().cpu().numpy().tolist())

        if DataKeys.IMAGE in data:
            data[DataKeys.IMAGE] = self._interpolate(data[DataKeys.IMAGE], tgt_size)
        if DataKeys.MASK in data:
            data[DataKeys.MASK] = self._interpolate(data[DataKeys.MASK].float(), tgt_size, "nearest").bool()
        if DataKeys.DEPTH in data:
            data[DataKeys.DEPTH] = self._interpolate(data[DataKeys.DEPTH], tgt_size, "nearest")
        if DataKeys.CAM in data:
            data[DataKeys.CAM].rescale_output_resolution(scale_factor.unsqueeze(0))
            cam_h, cam_w = data[DataKeys.CAM].height.item(), data[DataKeys.CAM].width.item()
            assert (
                cam_h == tgt_size[0] and cam_w == tgt_size[1]
            ), f"Size mismatch image and camera: {tgt_size} vs. ({cam_h}, {cam_w})"


@dataclass
class Crop(Transform):
    """Cropping transformation.

    Uses size to crop an image with the given method, or if size is None,
    uses the crop method to align the image size with a size divisor.
    """

    size: Optional[Tuple[int, int]] = None
    """Crop size in (H, W)."""
    size_divisor: int = 8
    """Divisor that the image dimensions should be compatible with."""
    method: Literal["topleft", "bottomright", "center", "random"] = "topleft"
    """Determines the crop location."""

    def __call__(self, data: Dict) -> None:
        """Crop the image."""
        if DataKeys.IMAGE in data:
            im_height, im_width = data[DataKeys.IMAGE].shape[:2]
        elif DataKeys.CAM in data:
            im_height, im_width = data[DataKeys.CAM].height.item(), data[DataKeys.CAM].width.item()
        else:
            raise RuntimeError(f"Invalid data format: {data.keys()}. Expected to contain image or camera")

        if self.size is None:
            width = (im_width // self.size_divisor) * self.size_divisor
            height = (im_height // self.size_divisor) * self.size_divisor
            crop_size = (height, width)
        else:
            crop_size = self.size

        h, w = crop_size
        # if the crop is bigger than the image, do nothing
        if not (im_height >= h and im_width >= w):
            return

        if self.method == "topleft":
            sy, sx = 0, 0
        elif self.method == "bottomright":
            sy, sx = im_height - h, im_width - w
        elif self.method == "center":
            sy = (im_height - h) // 2
            sx = (im_width - w) // 2
        elif self.method == "random":
            sy = random.randint(0, im_height - h)
            sx = random.randint(0, im_width - w)
        else:
            raise ValueError(f"Invalid crop method {self.method}")

        if DataKeys.IMAGE in data:
            data[DataKeys.IMAGE] = data[DataKeys.IMAGE][sy : sy + h, sx : sx + w]
            assert data[DataKeys.IMAGE].shape[:2] == crop_size
        if DataKeys.MASK in data:
            data[DataKeys.MASK] = data[DataKeys.MASK][sy : sy + h, sx : sx + w]
        if DataKeys.DEPTH in data:
            data[DataKeys.DEPTH] = data[DataKeys.DEPTH][sy : sy + h, sx : sx + w]
        if DataKeys.CAM in data:
            data[DataKeys.CAM].width = torch.full_like(data[DataKeys.CAM].width, w)
            data[DataKeys.CAM].height = torch.full_like(data[DataKeys.CAM].height, h)
            data[DataKeys.CAM].cx -= sx
            data[DataKeys.CAM].cy -= sy
