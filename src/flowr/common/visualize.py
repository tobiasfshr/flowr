"""Visualization functions."""
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from nerfstudio.utils import colormaps
from PIL import Image

from flowr.common.io import get_depth_image_from_path
from flowr.struct.cameras import Cameras


def apply_depth_colormap(
    depth: torch.Tensor, min_d=None, max_d=None, colormap_options=colormaps.ColormapOptions("inferno_r")
):
    """Apply a colormap to a depth image

    Args:
        depth (torch.Tensor): depth image
        min_d (float, optional): minimum depth value.
        max_d (float, optional): maximum depth value.
        colormap_options (colormaps.ColormapOptions, optional): colormap options. Defaults to colormaps.ColormapOptions("inferno_r").

    Returns:
        torch.Tensor: color coded image
    """
    if min_d is None:
        min_d = depth.min()
    if max_d is None:
        max_d = depth.max()

    depth = (depth - min_d) / (max_d - min_d + 1e-10)
    depth[depth < 0] = 0
    depth[depth > 1] = torch.inf
    colored_image = colormaps.apply_colormap(depth, colormap_options=colormap_options)
    return colored_image


def visualize_scene(
    log_path: str,
    cameras: Cameras | dict[str, Cameras],
    points: Optional[torch.Tensor] = None,
    rgb: Optional[torch.Tensor] = None,
    images: Optional[List[str | np.ndarray] | dict[str, List[str | np.ndarray]]] = None,
    depths: Optional[List[str | np.ndarray] | dict[str, List[str | np.ndarray]]] = None,
    depth_scale_factor: float = 1.0,
    server_address: Optional[str] = None,
    image_plane_distance: float = 0.5,
    image_max_size: int = 0,
    save_path: Optional[str] = None,
):
    """Rerun visualization of a scene with pointcloud and camera views."""
    import rerun as rr

    rr.init(log_path)
    rr.connect_tcp(server_address)
    if save_path is not None:
        rr.save(save_path)
    rr.set_time_seconds("stable_time", 0)
    rr.log(log_path, rr.ViewCoordinates.RFU, static=True)

    def _add_cameras(cams, ims=None, deps=None, name="cameras"):
        for idx, camera in enumerate(cams):
            camera: Cameras
            width, height = int(camera.width), int(camera.height)
            if image_max_size > 0:
                scale_factor = image_max_size / max((width, height))
                if scale_factor < 1.0:
                    width = int(width * scale_factor)
                    height = int(height * scale_factor)
                    camera.rescale_output_resolution(scale_factor)

            intrinsic = camera.get_intrinsics_matrices()
            rr.log(
                f"{log_path}/{name}/{idx}",
                rr.Transform3D(
                    translation=camera.camera_to_worlds[:3, 3],
                    mat3x3=camera.camera_to_worlds[:3, :3],
                    from_parent=False,
                ),
            )
            rr.log(
                f"{log_path}/{name}/{idx}",
                rr.Pinhole(
                    image_from_camera=intrinsic,
                    height=height,
                    width=width,
                    camera_xyz=rr.ViewCoordinates.RUB,
                    image_plane_distance=image_plane_distance,
                ),
            )
            if ims is not None:
                if isinstance(ims[idx], (str, Path)):
                    image = Image.open(ims[idx])
                    image = image.resize((width, height))
                else:
                    image = np.array(ims[idx]).astype(np.uint8)
                rr.log(
                    f"{log_path}/{name}/{idx}/rgb",
                    rr.Image(image),
                )
            if deps is not None:
                if isinstance(deps[idx], (str, Path)):
                    depth = get_depth_image_from_path(deps[idx], height, width, depth_scale_factor).squeeze(-1)
                else:
                    depth = np.array(deps[idx]).astype(np.uint8)
                rr.log(
                    f"{log_path}/{name}/{idx}/depth",
                    rr.DepthImage(depth),
                )

    if isinstance(cameras, dict):
        if images is not None:
            assert isinstance(images, dict)
        if depths is not None:
            assert isinstance(depths, dict)
        for split, cams in cameras.items():
            ims = images[split] if images is not None else None
            deps = depths[split] if depths is not None else None
            _add_cameras(cams, ims, deps, name=split)
    else:
        _add_cameras(cameras, images, depths)

    if points is not None:
        rr.log(
            f"{log_path}/pointcloud",
            rr.Points3D(
                positions=points,
                colors=rgb,
            ),
        )
    rr.disconnect()
