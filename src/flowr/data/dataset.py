"""Dataset that loads the inputs from the parsed data sources using file-type specific ops."""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Type, Union

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from flowr.common.io import get_depth_image_from_path, get_image_mask_from_path, load_image
from flowr.data.transforms import DataKeys, Transform
from flowr.struct.cameras import Cameras


@dataclass
class InputDatasetConfig(InstantiateConfig):
    """Default input dataset config."""

    _target: Type = field(default_factory=lambda: InputDataset)
    """target class to instantiate."""
    scale_factor: float = 1.0
    """Scale factor for globally resizing images. Contrary to dynamic resizing with a transformation, it allows refining intrinsics."""
    transforms: List[Transform] = field(default_factory=lambda: [])
    """Data transformations to apply after loading."""
    load_depth: bool = True
    """Whether to load depth maps (if available)."""


class InputDataset(Dataset):
    """Default input dataset.

    This class uses the dataparser outputs of the current data split,
    and loads the data into a format that the model expects. If
    applicable, it also handles data transformations that are needed
    to transform the loaded data into the expected format.
    """

    def __init__(self, config: InputDatasetConfig, dataparser_outputs: DataparserOutputs):
        super().__init__()
        self.config = config
        self._dataparser_outputs = dataparser_outputs
        self.image_filenames = dataparser_outputs.image_filenames
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        if self.config.scale_factor != 1.0:
            self.cameras.rescale_output_resolution(scaling_factor=self.config.scale_factor)
        self._with_depth = False
        if (
            self.config.load_depth
            and self.metadata is not None
            and "depth_filenames" in self.metadata
            and self.metadata["depth_filenames"] is not None
        ):
            self.depth_filenames = self.metadata["depth_filenames"]
            assert "depth_unit_scale_factor" in self.metadata
            self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]
            self._with_depth = True

    def __len__(self) -> int:
        return len(self.image_filenames)

    def get_numpy_image(self, image_filename: Union[str, Path]) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        pil_image = load_image(image_filename)
        if self.config.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.config.scale_factor), int(height * self.config.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.Resampling.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is incorrect."
        return image

    def get_image(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in float torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self.image_filenames[image_idx]
        image = torch.from_numpy(self.get_numpy_image(image_filename) / 255.0).float()
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image, image_filename

    def get_camera(self, image_idx: int) -> Cameras:
        """Select the camera corresponding to the image."""
        # TODO cameras might need to be handled differently bc of cam optimizer
        return self.cameras[image_idx : image_idx + 1]

    def get_mask(self, image_idx: int) -> Float[Tensor, "image_height image_width 1"]:
        """Load image masks from file."""
        mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
        mask = get_image_mask_from_path(filepath=mask_filepath, scale_factor=self.config.scale_factor)
        return torch.from_numpy(mask)

    def get_depth(self, image_idx: int, height: int, width: int) -> Float[Tensor, "image_height image_width 1"]:
        """Load the depth map corresponding to the input image."""
        depth_path = self.depth_filenames[image_idx]
        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        # resize depth image if needed
        if self.config.scale_factor != 1.0:
            width, height = int(width * self.config.scale_factor), int(height * self.config.scale_factor)
        return torch.from_numpy(get_depth_image_from_path(depth_path, height, width, scale_factor))

    def __getitem__(self, image_idx: int) -> Dict:
        image, image_path = self.get_image(image_idx)
        height, width = image.shape[:2]
        data = {DataKeys.IDX: image_idx, DataKeys.PATH: str(image_path), DataKeys.IMAGE: image}

        if self.cameras is not None:
            camera = self.get_camera(image_idx)
            data[DataKeys.CAM] = camera
            assert (
                camera.width == width and camera.height == height
            ), f"Inconsistent w/h: camera ({camera.width}, {camera.height}) vs. image ({width}, {height})"

        if (
            self._dataparser_outputs.mask_filenames is not None
            and self._dataparser_outputs.mask_filenames[image_idx] is not None
        ):
            mask = self.get_mask(image_idx)
            assert (
                mask.shape[:2] == image.shape[:2]
            ), f"Incompatible mask and image shapes: {mask.shape[:2]} vs {image.shape[:2]}"
            data[DataKeys.MASK] = mask

        if self._with_depth:
            data[DataKeys.DEPTH] = self.get_depth(image_idx, height, width)

        # apply data transformations
        for transform in self.config.transforms:
            transform(data)
        return data
