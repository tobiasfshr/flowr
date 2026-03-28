import math
from dataclasses import dataclass, field

import numpy as np
import torch
from nerfstudio.data.dataparsers.scannetpp_dataparser import ScanNetpp as NSScanNetpp
from nerfstudio.data.dataparsers.scannetpp_dataparser import ScanNetppDataParserConfig as NSScanNetppDataParserConfig
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from flowr.common.pointcloud import sample_closest_points, sample_farthest_points
from flowr.struct.cameras import Cameras


@dataclass
class ScanNetppDataParserConfig(NSScanNetppDataParserConfig):
    """Config for ScanNet++ data parser with sparse train fraction support."""

    _target: type = field(default_factory=lambda: ScanNetpp)
    sparse_training_fraction: float = 1.0
    """Training / other view fraction (sparse training). If 1.0, it is dense training."""


class ScanNetpp(NSScanNetpp):
    """ScanNet++ data parser with sparse train fraction support."""

    def _generate_dataparser_outputs(self, split: str = "train"):
        split_name = "train" if split == "other" else split
        dpo = super()._generate_dataparser_outputs(split=split_name)

        # nerfstudio to flowr cameras
        dpo.cameras = Cameras(
            fx=dpo.cameras.fx,
            fy=dpo.cameras.fy,
            cx=dpo.cameras.cx,
            cy=dpo.cameras.cy,
            distortion_params=dpo.cameras.distortion_params,
            height=dpo.cameras.height,
            width=dpo.cameras.width,
            camera_to_worlds=dpo.cameras.camera_to_worlds,
            camera_type=dpo.cameras.camera_type,
        )

        if split not in ["train", "other"]:
            return dpo

        if self.config.sparse_training_fraction < 1.0:
            positions = dpo.cameras.camera_to_worlds[:, :3, 3]
            num_views = int(self.config.sparse_training_fraction * len(dpo.cameras))
            num_keyframes = int(math.sqrt(len(dpo.cameras)))
            keyframe_ids = sample_farthest_points(positions, num_keyframes)
            i_train = (
                torch.cat([keyframe_ids, sample_closest_points(positions, keyframe_ids, num_views - num_keyframes)])
                .cpu()
                .numpy()
            )
            i_other = np.setdiff1d(np.arange(len(dpo.cameras)), i_train)
            dpo.image_filenames = (
                [dpo.image_filenames[i] for i in i_train]
                if split == "train"
                else [dpo.image_filenames[i] for i in i_other]
            )
            dpo.cameras = (
                Cameras.cat([dpo.cameras[i : i + 1] for i in i_train])
                if split == "train"
                else Cameras.cat([dpo.cameras[i : i + 1] for i in i_other])
            )
            if dpo.mask_filenames is not None:
                dpo.mask_filenames = (
                    [dpo.mask_filenames[i] for i in i_train]
                    if split == "train"
                    else [dpo.mask_filenames[i] for i in i_other]
                )
        elif split == "other":
            dpo.image_filenames = []
            dpo.cameras = []
            dpo.mask_filenames = None

        return dpo


ScannetppDataParserSpecification = DataParserSpecification(config=ScanNetppDataParserConfig())
