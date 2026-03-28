"""Render viewpoints, trajectories, etc from a reconstruction model."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

import mediapy as media
import numpy as np
import torch
import tyro
from nerfstudio.cameras.camera_utils import get_interpolated_poses
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.scripts.render import BaseRender, _render_trajectory_video
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from tqdm import tqdm
from typing_extensions import Annotated

from flowr.common.io import save_image
from flowr.common.visualize import apply_depth_colormap
from flowr.struct.cameras import Cameras, CameraType
from flowr.util.cameras import perturb_cameras, three_js_perspective_camera_focal_length
from flowr.util.eval import eval_setup


def trajectory_from_data(
    dataparser_outputs: DataparserOutputs,
    image_height: int = 1080,
    image_width: int = 1920,
    camera_id: int = -1,
    fov: int = 50,
    speed: float = 1.0,  # m/s
    fps: int = 30,
):
    """Generate a smooth trajectory from a dataset."""
    cams = []
    for cam in dataparser_outputs.cameras:
        if (
            camera_id >= 0
            and cam.metadata is not None
            and "camera_ids" in cam.metadata
            and cam.metadata["camera_ids"] != camera_id
        ):
            continue
        cams.append(cam)

    has_time = False
    if cams[0].times is not None:
        # assume all cameras have time
        cams = sorted(cams, key=lambda x: x.times[0])
        has_time = True

    dataparser_scale = dataparser_outputs.dataparser_scale
    cam2worlds = torch.stack([cam.camera_to_worlds for cam in cams])
    # keyframe selection
    keyframes = [cam2worlds[0]]
    for c2w in cam2worlds[1:]:
        # if translation is larger than 1m, add as keyframe
        if torch.linalg.norm(c2w[:3, 3] - keyframes[-1][:3, 3]) > 1.0 * dataparser_scale:
            keyframes.append(c2w)
    keyframes = torch.stack(keyframes)

    # interpolate trajectory from keyframes
    traj = []
    for idx in range(keyframes.shape[0] - 1):
        pose_a = keyframes[idx]
        pose_b = keyframes[idx + 1]
        steps = int(torch.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]) / dataparser_scale / speed * fps)
        poses_ab = get_interpolated_poses(pose_a, pose_b, steps=steps)[:-1]
        traj += poses_ab

    cam2worlds = np.stack(traj, axis=0)
    cam2worlds = torch.from_numpy(cam2worlds).float()[:, :3]
    focal_length = three_js_perspective_camera_focal_length(fov, image_height)
    fxs = torch.tensor(focal_length)
    fys = torch.tensor(focal_length)
    if has_time:
        times = torch.linspace(-1.0, 1.0, len(cam2worlds)).unsqueeze(-1).float()
    else:
        times = None

    cams = Cameras(
        fx=fxs,
        fy=fys,
        cx=image_width / 2,
        cy=image_height / 2,
        height=image_height,
        width=image_width,
        camera_to_worlds=cam2worlds,
        camera_type=CameraType.PERSPECTIVE,
        times=times,
    )
    return cams


@dataclass
class PathRender(BaseRender):
    """Render a smooth camera path from the input data."""

    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    fps: int = 30
    """Frames per second for the output video."""
    speed: float = 1.0
    """Speed of the camera path in m/s."""
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions("inferno_r")
    """Options for the colormap."""
    min_depth = 0.0
    """Minimum depth for visualization."""
    max_depth = 10.0
    """Maximum depth for visualization."""
    split: Literal["train", "test"] = "test"
    """Which split is the basis for keyframe interpolation."""
    camera_id: int = 0
    """If multi-cam data, choose ref camera to follow."""
    resolution: Optional[tuple[int, int]] = None
    """Image resolution to render (h, w)."""

    @torch.no_grad()
    def main(self) -> None:
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )
        install_checks.check_ffmpeg_installed()

        parser = pipeline.datamanager.dataparser
        res_dict = (
            dict(image_height=self.resolution[0], image_width=self.resolution[1]) if self.resolution is not None else {}
        )
        camera_path = trajectory_from_data(
            parser.get_dataparser_outputs(split=self.split),
            speed=self.speed,
            fps=self.fps,
            camera_id=self.camera_id,
            **res_dict,
        )
        crop_data = None
        seconds = len(camera_path) / self.fps

        dataparser_scale = pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        min_d, max_d = self.min_depth, self.max_depth
        min_d, max_d = min_d * dataparser_scale, max_d * dataparser_scale

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=crop_data,
            seconds=seconds,
            output_format=self.output_format,
            colormap_options=self.colormap_options,
            depth_near_plane=min_d,
            depth_far_plane=max_d,
        )


@dataclass
class ViewRender(BaseRender):
    """Render a certain evaluation image given its index."""

    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc."""
    image_idx: Union[int, List[int]] = -1
    """Index of the image to render."""
    output_path: Path = Path("renders")
    """Path to output directory."""
    render_debugging_images: bool = False
    """Whether to render debugging images."""
    min_depth = 0.0
    """Minimum depth for visualization."""
    max_depth = 10.0
    """Maximum depth for visualization."""
    concat_gt: bool = False
    """Whether to concatenate the rendered image with the GT image for comparison."""
    split: Literal["train", "test", "other"] = "test"
    """Which split to render from."""

    def main(self) -> None:
        """Main function."""
        test_mode = "inference"
        if self.concat_gt:
            test_mode = "test"
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode=test_mode,
        )

        if self.split == "train":
            cameras = pipeline.datamanager.train_dataset.cameras
            im_filenames = pipeline.datamanager.train_dataset.image_filenames
        elif self.split == "test":
            cameras = pipeline.datamanager.eval_dataset.cameras
            im_filenames = pipeline.datamanager.eval_dataset.image_filenames
        else:
            assert self.split == "other"
            dpo = pipeline.datamanager.dataparser.get_dataparser_outputs(split="other")
            cameras = dpo.cameras
            im_filenames = dpo.image_filenames

        cameras.rescale_output_resolution(1.0 / self.downscale_factor)
        cameras = cameras.to(pipeline.device)

        dataparser_scale = pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        min_d, max_d = self.min_depth, self.max_depth
        min_d, max_d = min_d * dataparser_scale, max_d * dataparser_scale

        if self.image_idx == -1:
            self.image_idx = list(range(len(cameras)))
        elif isinstance(self.image_idx, int):
            self.image_idx = [self.image_idx]

        os.makedirs(self.output_path, exist_ok=True)
        for im_idx in self.image_idx:
            for rendered_output_name in self.rendered_output_names:
                with torch.no_grad():
                    cam = cameras[im_idx : im_idx + 1]
                    outputs = pipeline.model.get_outputs_for_camera(cam)

                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                output_image = outputs[rendered_output_name]
                if rendered_output_name == "depth":
                    output_image = apply_depth_colormap(output_image, min_d, max_d)
                    output_image = output_image.cpu().numpy()
                else:
                    output_image = colormaps.apply_colormap(image=output_image).cpu().numpy()

                if self.concat_gt:
                    gt_image = np.array(Image.open(im_filenames[im_idx])).astype(np.float32) / 255
                    output_image = np.concatenate([gt_image, output_image], -2)

                outpath = (
                    self.output_path
                    / f"{im_idx:05d}_{rendered_output_name}.{'png' if self.image_format == 'png' else 'jpg'}"
                )
                media.write_image(outpath, output_image)
                CONSOLE.print(f"Output written to {outpath}.", justify="center")


@dataclass
class NovelViewRender:
    """Run rendering on novel views generated with random perturbations."""

    load_config: Path
    """Path to trained reconstruction model config."""
    output_dir: Path = Path("renders/novel_views")
    """The location to output the augmented training dataset."""
    translation_range: float = 0.25
    """Range of translation perturbation."""
    rotation_range: float = 30
    """Range of rotation angle (yaw, pitch) perturbation."""

    def main(self) -> None:
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            test_mode="val",
        )
        os.makedirs(self.output_dir, exist_ok=True)
        dpo = pipeline.datamanager.train_dataparser_outputs
        new_cameras = perturb_cameras(dpo.cameras, self.translation_range, self.rotation_range)
        for idx in tqdm(range(len(new_cameras))):
            image = pipeline.model.get_outputs_for_camera(camera=new_cameras[idx : idx + 1])["rgb"]
            image_orig = pipeline.model.get_outputs_for_camera(camera=dpo.cameras[idx : idx + 1])["rgb"]
            save_path = os.path.join(self.output_dir, f"{idx:04d}.jpg")
            save_image(torch.cat([image, image_orig], 1), save_path)
        new_cameras.to_json(f"{self.output_dir}/cameras.json")


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ViewRender, tyro.conf.subcommand(name="view")],
        Annotated[PathRender, tyro.conf.subcommand(name="path")],
        Annotated[NovelViewRender, tyro.conf.subcommand(name="novel")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()
