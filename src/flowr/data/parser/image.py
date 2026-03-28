"""Parser for image/camera file folders."""
import fnmatch
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.rich_utils import CONSOLE

from flowr.common.io import from_ply, load_json, zip_exists, zip_listdir
from flowr.struct.cameras import Cameras
from flowr.util.colmap.create_pointcloud import create_ply_from_posed_images as create_ply_from_posed_images_colmap
from flowr.util.colmap.scale_to_metric import scale_reconstruction


@dataclass
class ImageDataParserConfig(DataParserConfig):
    """Config for loading image / camera files from a folder."""

    _target: Type = field(default_factory=lambda: ImageDataParser)
    """_target: target class to instantiate"""
    pattern: str = "*"
    """Which file pattern(s) to use."""
    input_filetype: Literal["jpg", "png", "auto"] = "auto"
    """Input filetype for search for."""
    train_split_fraction: float = 1.0
    """Splits the dataset into train/eval sets. If 1.0, use all images for training and evaluation."""
    load_3D_points: bool = True
    """Whether to triangulate or load 3D points."""
    use_generated: bool = False
    """Whether to include generated frames in training."""
    pointcloud_path: Optional[Path] = None
    """Path to load or save pointcloud."""
    scale_path: Optional[Path] = None
    """Path to load scene scaling factor from (if any)."""
    pointcloud_method: Literal["colmap", "mast3r"] = "mast3r"
    """Method to create pointcloud."""
    min_views_for_retrieval: int = 12
    """Minimum number of images to use for sparse view graph."""
    use_mast3r_GA: bool = False
    """Whether to use MASt3R-SfM GA or colmap triangulation of matches."""
    window_size: int = 5
    """Window size for MASt3R-SfM correspondence graph creation."""
    sparse_matching: bool = False
    """Whether to sparsify MASt3R-SfM matches."""
    vocab_tree_path: Optional[str] = None
    """Path to vocabulary tree file for colmap."""
    sift_gpu: bool = False
    """Whether to use GPU implementation of SIFT for feature extraction."""
    max_colmap_images: int = 1024
    """Maximum number of images to use for COLMAP reconstruction."""
    apply_scale: bool = True
    """Whether to apply the scale factor to the pointcloud."""


class ImageDataParser(DataParser):
    """Image folder dataparser.

    Either load files from a folder directly, from subfolders 'train' and 'test' or '{split}/images'.

    Loads images from a folder given the desired name pattern and file extensions (if any).
    Loads camera dicts if available or estimates poses/sparse pointclouds.
    """

    config: ImageDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs: Optional[Dict]) -> DataparserOutputs:
        filetypes = ["jpg", "png", "jpeg"]
        if self.config.input_filetype == "jpg":
            filetypes = ["jpg", "jpeg"]
        elif self.config.input_filetype == "png":
            filetypes = ["png"]

        input_dir = self.config.data
        assert split in ["train", "test", "other", "val"]
        split = split if split != "val" else "test"

        if input_dir.suffix == ".zip":
            listdir = zip_listdir
            exists = zip_exists
        else:
            listdir = os.listdir
            exists = os.path.exists

        if "train" in listdir(input_dir):
            assert "test" in listdir(input_dir), f"Expected both train and test set, but got {listdir(input_dir)}"
            image_dir = self.config.data / split
        else:
            image_dir = input_dir

        if exists(image_dir / "images"):
            mask_dir = image_dir / "masks"
            image_dir = image_dir / "images"
        else:
            image_dir = input_dir
            mask_dir = None

        image_files = []
        for im_file in fnmatch.filter(listdir(image_dir), self.config.pattern):
            if any([im_file.lower().endswith(filetype) for filetype in filetypes]):
                image_files.append(Path(os.path.join(image_dir, im_file)))
        image_files = sorted(image_files)
        assert len(image_files) > 0

        # load masks (if any)
        mask_files = []
        if mask_dir is not None and mask_dir.exists():
            for im_file in image_files:
                mask_file = mask_dir / im_file.name
                assert mask_file.exists(), f"Mask file {mask_file} for image {im_file.name} does not exist!"
                mask_files.append(Path(os.path.join(mask_dir, mask_file)))

        # check for a cameras.json file
        if exists(input_dir / "cameras.json"):
            cameras_path = input_dir / "cameras.json"
        elif exists(input_dir / f"{split}_cameras.json"):
            cameras_path = input_dir / f"{split}_cameras.json"
            if split == "other" and (input_dir / "other_cameras_new.json").exists():
                cameras_path = input_dir / "other_cameras_new.json"
        else:
            raise FileNotFoundError(
                f"No camera files found in {input_dir}. Provide cameras.json or per-split camera files."
            )

        cameras_dict = load_json(cameras_path)

        assert set(cameras_dict.keys()) == set(
            im_file.name for im_file in image_files
        ), f"Mismatch between images and cameras. Diff: {set(cameras_dict.keys()) ^ set(im_file.name for im_file in image_files)}"
        cameras = Cameras.cat([Cameras.from_dict(cameras_dict[im_file.name]).unsqueeze(0) for im_file in image_files])
        cameras.camera_type = torch.ones_like(cameras.camera_type)  # set all cameras to perspective

        # init required metadata
        if cameras.metadata is None:
            cameras.metadata = {}

        if "is_generated" not in cameras.metadata:
            generated_mask = torch.zeros_like(cameras.fx).bool()
            cameras.metadata["is_generated"] = generated_mask

        # add generated views (other split) to training set
        if split == "train" and self.config.use_generated:
            other_dpo = self._generate_dataparser_outputs(split="other")
            other_dpo.cameras.metadata["is_generated"] = torch.ones_like(other_dpo.cameras.fx).bool()
            cameras = Cameras.cat([cameras, other_dpo.cameras])
            image_files += other_dpo.image_filenames

        if self.config.train_split_fraction < 1.0:
            i_train, i_eval = get_train_eval_split_fraction(image_files, self.config.train_split_fraction)
            if split == "train":
                indices = i_train
            else:
                indices = i_eval
            image_files = [image_files[i] for i in indices]
            cameras = cameras[torch.tensor(indices, dtype=torch.long)]

        metadata = {}
        if split == "train" and self.config.load_3D_points:
            pointcloud_path = (
                self.config.pointcloud_path if self.config.pointcloud_path is not None else input_dir / "pointcloud.ply"
            )
            if not exists(pointcloud_path):
                assert ".zip/" not in str(pointcloud_path), "Cannot create pointcloud in zip file"
                if self.config.pointcloud_method == "colmap":
                    CONSOLE.log("Creating initial pointcloud with COLMAP...")
                    match_method = "vocab_tree" if self.config.vocab_tree_path is not None else "exhaustive"
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        create_ply_from_posed_images_colmap(
                            temp_path,
                            image_files,
                            cameras,
                            save_path=pointcloud_path,
                            matching_method=match_method,
                            vocab_tree_path=self.config.vocab_tree_path,
                            use_gpu=self.config.sift_gpu,
                            max_images=self.config.max_colmap_images,
                        )
                        if (
                            self.config.scale_path is not None
                            and not (self.config.scale_path / "scale_factor.txt").exists()
                        ):
                            scale_factor = scale_reconstruction(temp_path / "colmap" / "sparse", image_dir)
                            with open(self.config.scale_path / "scale_factor.txt", "w") as f:
                                f.write(str(scale_factor))
                else:
                    CONSOLE.log("Creating initial pointcloud with MASt3R+COLMAP...")
                    from flowr.util.mast3r.create_pointcloud import create_ply_from_posed_images

                    create_ply_from_posed_images(
                        image_files,
                        cameras,
                        pointcloud_path,
                        use_mast3r_GA=self.config.use_mast3r_GA,
                        winsize=self.config.window_size,
                        dense_matching=not self.config.sparse_matching,
                        min_views_for_retrieval=self.config.min_views_for_retrieval,
                        scale_path=self.config.scale_path,
                    )

            assert exists(pointcloud_path), "Failed to create initial pointcloud"
            CONSOLE.log(f"Using cached 3D points at {pointcloud_path}")
            points3D, points3D_rgb = from_ply(pointcloud_path)
            if len(points3D) > 1000:
                points3D = torch.from_numpy(points3D)
                points3D_rgb = torch.from_numpy(points3D_rgb)
                metadata.update(
                    {
                        "points3D_xyz": points3D,
                        "points3D_rgb": points3D_rgb,
                    }
                )
            else:
                CONSOLE.log(
                    "Sparse Reconstruction does not have enough points for initialization. Please use random initialization."
                )

        # scale poses, pointcloud
        scale_factor = 1.0
        if self.config.scale_path is not None and self.config.apply_scale:
            with open(self.config.scale_path / "scale_factor.txt", "r") as f:
                scale_factor = float(f.read())

        cameras.camera_to_worlds[:, :3, 3] *= scale_factor
        if "points3D_xyz" in metadata:
            metadata["points3D_xyz"] *= scale_factor

        if split == "train":
            if cameras.metadata is None:
                cameras.metadata = {}
            cameras.metadata["cam_idx"] = torch.arange(len(cameras), dtype=torch.long).unsqueeze(-1)

        return DataparserOutputs(
            image_filenames=image_files,
            mask_filenames=mask_files if len(mask_files) > 0 else None,
            cameras=cameras,
            metadata=metadata,
            dataparser_scale=scale_factor,
        )


def get_train_eval_split_fraction(image_filenames: List, train_split_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split fraction based on the number of images and the train split fraction.

    Args:
        image_filenames: list of image filenames
        train_split_fraction: fraction of images to use for training
    """

    # filter image_filenames and poses based on train/eval split percentage
    num_images = len(image_filenames)
    num_train_images = math.ceil(num_images * train_split_fraction)
    num_eval_images = num_images - num_train_images
    i_all = np.arange(num_images)
    i_train = np.linspace(
        0, num_images - 1, num_train_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
    assert len(i_eval) == num_eval_images

    return i_train, i_eval


ImageDataParserSpecification = DataParserSpecification(
    config=ImageDataParserConfig(),
    description="Dataparser for generic image folders.",
)
