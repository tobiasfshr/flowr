"""Test the data transformations."""
import random

import torch
from torchvision.transforms.functional import resize as image_resize

from flowr.common.random import set_random_seed
from flowr.data.transforms import Crop, DataKeys, HWCtoCHW, Resize
from flowr.struct.cameras import Cameras


def random_crop(img, w, h=None):
    """Reference implementation."""
    if h is None:
        h = w

    h = min(h, img.shape[-2])
    w = min(w, img.shape[-1])

    min_scale = max(w / img.shape[-1], h / img.shape[-2])
    p = random.random()
    rand_scale = (p * (1 - min_scale**3) + min_scale**3) ** (1 / 3.0)
    min_size = int(rand_scale * img.shape[-2]), int(rand_scale * img.shape[-1])

    scaled_img = image_resize(img, min_size)
    sy = random.randint(0, scaled_img.shape[-2] - h)
    sx = random.randint(0, scaled_img.shape[-1] - w)
    return scaled_img[..., sy : sy + h, sx : sx + w]


def test_transforms():
    """Test transformation function against reference implementation."""
    for height, width in [(1920, 1280), (1280, 1920)]:
        test_image = torch.rand((height, width, 3))
        test_mask = torch.rand((height, width, 1)) > 0.5
        test_depth = torch.rand((height, width, 1))
        camera = Cameras(
            fx=300.0,
            fy=300.0,
            cx=645.0,  # 5px left from middle
            cy=955.0,  # 5px top from middle
            camera_to_worlds=torch.eye(4)[:3].unsqueeze(0),
            width=width,
            height=height,
        )
        size = (512, 512)
        my_seed = random.randint(0, 1000)

        set_random_seed(my_seed)
        reference_image = random_crop(test_image.permute(2, 0, 1), *size)

        set_random_seed(my_seed)
        data = {DataKeys.IMAGE: test_image, DataKeys.MASK: test_mask, DataKeys.DEPTH: test_depth, DataKeys.CAM: camera}
        transforms = [Resize(size=512, randomize=True), Crop(size=size, method="random"), HWCtoCHW()]
        for t in transforms:
            t(data)

        assert torch.allclose(
            data[DataKeys.IMAGE], reference_image
        ), f"Not aligned. Diff sum: {(data[DataKeys.IMAGE] - reference_image).abs().sum()}"
        assert data[DataKeys.IMAGE].shape[-2:] == data[DataKeys.MASK].shape[:2] == data[DataKeys.DEPTH].shape[:2]
        assert data[DataKeys.IMAGE].shape[2] == data[DataKeys.CAM].width.item()
        assert data[DataKeys.IMAGE].shape[1] == data[DataKeys.CAM].height.item()
