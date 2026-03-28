"""Cameras subclass of nerfstudio that supports concatenation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from jaxtyping import Float
from nerfstudio.cameras.cameras import Cameras as NerfstudioCameras
from nerfstudio.cameras.cameras import CameraType  # noqa: F401
from torch import Tensor


class Cameras(NerfstudioCameras):
    """Cameras subclass supporting concatenation."""

    def clone(self):
        if self.metadata is None:
            metadata = None
        else:
            metadata = {}
            for key in self.metadata:
                assert isinstance(self.metadata[key], torch.Tensor)
                metadata[key] = self.metadata[key].clone()

        return Cameras(
            camera_to_worlds=self.camera_to_worlds.clone(),
            fx=self.fx.clone(),
            fy=self.fy.clone(),
            cx=self.cx.clone(),
            cy=self.cy.clone(),
            width=self.width.clone(),
            height=self.height.clone(),
            distortion_params=self.distortion_params.clone() if self.distortion_params is not None else None,
            times=self.times.clone() if self.times is not None else None,
            metadata=metadata,
        )

    def unsqueeze(self, dim: int = 0):
        if len(self.camera_to_worlds.shape) > 2:
            return self
        self.camera_to_worlds = self.camera_to_worlds.unsqueeze(dim)
        self.fx = self.fx.unsqueeze(dim)
        self.fy = self.fy.unsqueeze(dim)
        self.cx = self.cx.unsqueeze(dim)
        self.cy = self.cy.unsqueeze(dim)
        self.width = self.width.unsqueeze(dim)
        self.height = self.height.unsqueeze(dim)
        if self.distortion_params is not None:
            self.distortion_params = self.distortion_params.unsqueeze(dim)
        if self.times is not None:
            self.times = self.times.unsqueeze(dim)
        if self.metadata is not None:
            for key in self.metadata:
                self.metadata[key] = self.metadata[key].unsqueeze(dim)
        return self

    @classmethod
    def cat(cls, cameras: List[Cameras]) -> Cameras:
        """Concatenate multiple Cameras Objects."""
        cam2worlds = torch.cat([cam.camera_to_worlds for cam in cameras])
        fxs = torch.cat([cam.fx for cam in cameras])
        fys = torch.cat([cam.fy for cam in cameras])
        cxs = torch.cat([cam.cx for cam in cameras])
        cys = torch.cat([cam.cy for cam in cameras])
        widths = torch.cat([cam.width for cam in cameras])
        heights = torch.cat([cam.height for cam in cameras])
        if any(cam.distortion_params is not None for cam in cameras):
            for cam in cameras:
                if cam.distortion_params is None:
                    cam.distortion_params = cam.camera_to_worlds.new_zeros(len(cam), 6)
            distortion_params = torch.cat([cam.distortion_params for cam in cameras])
        else:
            distortion_params = None
        camera_types = torch.cat([cam.camera_type for cam in cameras])

        # Check if optional time attribute is consistent across cameras
        if cameras[0].times is not None:
            assert all(cam.times is not None for cam in cameras)
            times = torch.cat([cam.times for cam in cameras])
        else:
            assert all(cam.times is None for cam in cameras)
            times = None

        # Union of all metadata, given they are the same
        if cameras[0].metadata is not None:
            metadata_keys = cameras[0].metadata.keys()
            metadata = {}
            for key in metadata_keys:
                metadata[key] = torch.cat([cam.metadata[key] for cam in cameras])
        else:
            metadata = None
            assert all(cam.metadata is None for cam in cameras), "Cameras have inconsistent metadata."

        return Cameras(
            camera_to_worlds=cam2worlds,
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            width=widths,
            height=heights,
            distortion_params=distortion_params,
            camera_type=camera_types,
            times=times,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Tensor]:
        """Puts Cameras Object info into a dictionary that can be saved."""
        data = {
            "camera_to_worlds": self.camera_to_worlds.cpu().numpy().tolist(),
            "fx": self.fx.cpu().numpy().tolist(),
            "fy": self.fy.cpu().numpy().tolist(),
            "cx": self.cx.cpu().numpy().tolist(),
            "cy": self.cy.cpu().numpy().tolist(),
            "width": self.width.cpu().numpy().tolist(),
            "height": self.height.cpu().numpy().tolist(),
            "distortion_params": self.distortion_params.cpu().numpy().tolist()
            if self.distortion_params is not None
            else None,
            "camera_type": self.camera_type.cpu().numpy().tolist(),
            "times": self.times.cpu().numpy().tolist() if self.times is not None else None,
            "metadata": convert_for_json(self.metadata) if self.metadata is not None else None,
        }
        return data

    def to_json(self, file_path: str | Path):
        """Export camera object to json."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, data: Dict[str, Tensor]) -> Cameras:
        """Generate Cameras object from dict."""
        for key, val in data.items():
            if key == "metadata" and val is not None:
                metadata = {}
                for k, v in data[key].items():
                    metadata[k] = torch.tensor(v)
                data[key] = metadata
            elif val is not None:
                data[key] = torch.tensor(val)
        return cls(**data)

    @classmethod
    def from_json(cls, file_path: str | Path) -> Cameras:
        """Import camera object from json."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_intrinsics_matrices(self) -> Float[Tensor, "*num_cameras 3 3"]:
        """Returns the intrinsic matrices for each camera.

        Returns:
            Pinhole camera intrinsics matrices
        """
        K = torch.zeros((*self.shape, 3, 3), dtype=torch.float32, device=self.device)
        K[..., 0, 0] = self.fx.squeeze(-1)
        K[..., 1, 1] = self.fy.squeeze(-1)
        K[..., 0, 2] = self.cx.squeeze(-1)
        K[..., 1, 2] = self.cy.squeeze(-1)
        K[..., 2, 2] = 1.0
        return K

    def get_frustums(self, near: float = 0.1, far: float = 1.0) -> Float[Tensor, "*num_cameras 8 3"]:
        """Create frustum vertices for a batch of cameras using their intrinsic matrices.

            (4) +-----------------+ (5)
                |                 |
                |                 |
                |     (0) +---------+ (1)
                |         |       | |
            (7) +-----------------+. (6)
                 ` .      |         |
                      (3) +---------+ (2)

        Args:
            near: Near plane distance
            far: Far plane distance

        Returns:
            frustum_vertices: Frustum vertices (Bx8x3 tensor)
        """
        tlx = (-self.cx / self.fx).squeeze(-1)
        tly = (-self.cy / self.fy).squeeze(-1)
        brx = ((self.width - self.cx) / self.fx).squeeze(-1)
        bry = ((self.height - self.cy) / self.fy).squeeze(-1)
        ones = torch.ones_like(tlx)
        plane = torch.stack(
            [
                torch.stack([tlx, tly, ones], dim=-1),
                torch.stack([brx, tly, ones], dim=-1),
                torch.stack([brx, bry, ones], dim=-1),
                torch.stack([tlx, bry, ones], dim=-1),
            ],
            dim=-2,
        )
        near_fars = torch.cat([plane * near, plane * far], dim=-2)
        # frustum to world
        near_fars = (self.camera_to_worlds[..., None, :3, :3] @ near_fars[..., None]).squeeze(
            -1
        ) + self.camera_to_worlds[..., None, :3, 3]
        return near_fars


def convert_for_json(tensor_dict: Dict[str, Tensor]) -> Dict:
    """Converts a dictionary with tensor values to a JSON-serializable format."""
    json_ready_dict = {}
    for key, tensor in tensor_dict.items():
        converted = tensor.detach().cpu().numpy().tolist()
        json_ready_dict[key] = converted
    return json_ready_dict
