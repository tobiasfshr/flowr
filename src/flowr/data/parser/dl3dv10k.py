import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import numpy as np
import torch
from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.rich_utils import CONSOLE

from flowr.common.io import from_ply
from flowr.struct.cameras import Cameras
from flowr.util.colmap.create_pointcloud import create_ply_from_posed_images


@dataclass
class DL3DV10KDataParserConfig(NerfstudioDataParserConfig):
    """Dataparser for dl3dv dataset supporting camera drop-out."""

    _target: Type = field(default_factory=lambda: DL3DV10KParser)
    """target class to instantiate"""
    sparse_training_fraction: float | int = 0.5
    """Training / other view fraction (sparse training). If 1.0, it is dense training."""
    pointcloud_path: Optional[Path] = None
    """Path to save pointcloud."""
    scale_path: Optional[Path] = None
    """Path to load scene scaling factor from (if any)."""
    vocab_tree_path: Optional[str] = None
    """Path to vocabulary tree file for colmap."""


class DL3DV10KParser(Nerfstudio):
    """DL3DV10K Parser."""

    def _load_outputs(self, split):
        load_3D_points = self.config.load_3D_points
        self.config.load_3D_points = False
        if split == "other":
            split = "train"
        if self.config.scale_path is not None:
            self.config.auto_scale_poses = False
        outputs = super()._generate_dataparser_outputs(split)
        self.config.load_3D_points = load_3D_points
        return outputs

    def _generate_dataparser_outputs(self, split="train"):
        outputs = self._load_outputs(split)

        if outputs.metadata["depth_filenames"] is None:
            del outputs.metadata["depth_filenames"]

        if split in ["train", "other"]:
            if not self.config.sparse_training_fraction < 1.0:
                assert isinstance(self.config.sparse_training_fraction, int)
                # filter image_filenames and poses based on train/eval split percentage
                num_images = len(outputs.image_filenames)
                num_train_images = self.config.sparse_training_fraction
                i_all = np.arange(num_images)
                indices = np.linspace(0, num_images - 1, num_train_images, dtype=int)
                i_other = np.setdiff1d(i_all, indices)
            else:
                # filter training views for sparse view training
                indices, i_other = get_train_eval_split_fraction(
                    outputs.image_filenames, self.config.sparse_training_fraction
                )
            if split == "other":
                indices = i_other
        else:
            assert split != "other"
            indices = list(range(len(outputs.image_filenames)))

        # filter by indices
        outputs.cameras = Cameras.cat([outputs.cameras[i : i + 1] for i in indices])
        outputs.image_filenames = [outputs.image_filenames[i] for i in indices]
        if outputs.mask_filenames is not None:
            outputs.mask_filenames = [outputs.mask_filenames[i] for i in indices]

        # create 3D points from training view set
        if self.config.load_3D_points and split == "train":
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                pointcloud_path = temp_path if self.config.pointcloud_path is None else self.config.pointcloud_path
                if not os.path.basename(pointcloud_path).endswith(".ply"):
                    pointcloud_path = pointcloud_path / f"pointcloud_{len(outputs.image_filenames)}views.ply"
                os.makedirs(os.path.dirname(pointcloud_path), exist_ok=True)
                if not pointcloud_path.exists():
                    CONSOLE.log("Creating initial pointcloud with COLMAP...")
                    match_method = "vocab_tree" if self.config.vocab_tree_path is not None else "exhaustive"
                    create_ply_from_posed_images(
                        temp_path,
                        outputs.image_filenames,
                        outputs.cameras,
                        save_path=pointcloud_path,
                        matching_method=match_method,
                        vocab_tree_path=self.config.vocab_tree_path,
                    )

                assert pointcloud_path.exists(), "Failed to create initial pointcloud"
                CONSOLE.log(f"Using cached 3D points at {pointcloud_path}")
                points3D, points3D_rgb = from_ply(pointcloud_path)
            if len(points3D) > 1000:
                points3D = torch.from_numpy(points3D)
                points3D_rgb = torch.from_numpy(points3D_rgb)
                outputs.metadata.update(
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
        scale_factor = outputs.dataparser_scale
        if self.config.scale_path is not None:
            with open(self.config.scale_path / "scale_factor.txt", "r") as f:
                scale_factor = float(f.read())
            outputs.cameras.camera_to_worlds[:, :3, 3] *= scale_factor
            if "points3D_xyz" in outputs.metadata:
                outputs.metadata["points3D_xyz"] *= scale_factor

        if "points3D_xyz" in outputs.metadata:
            assert not self.config.auto_scale_poses
            # get aabb from pointcloud (ignoring outliers)
            mi, ma = torch.quantile(outputs.metadata["points3D_xyz"], torch.FloatTensor([0.02, 0.98]), dim=0)
            outputs.scene_box = SceneBox(aabb=torch.stack([mi, ma]).float())
        else:
            outputs.scene_box.aabb *= scale_factor

        CONSOLE.log(f"Num {split} views: {len(outputs.image_filenames)}")
        return outputs


DL3DV10KDataParserSpecification = DataParserSpecification(
    config=DL3DV10KDataParserConfig(),
    description="Dataparser for DL3DV10K dataset.",
)
