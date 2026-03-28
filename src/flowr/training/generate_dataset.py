import argparse
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from nerfstudio.utils.rich_utils import CONSOLE
from torch import Tensor
from tqdm.auto import tqdm

from flowr.common.io import to_ply
from flowr.common.random import set_random_seed
from flowr.common.visualize import visualize_scene
from flowr.config.workflow import instant_splatfacto
from flowr.data.parser.image import ImageDataParserConfig
from flowr.struct.cameras import Cameras
from flowr.util.cameras import generate_interpolated_path, perturb_cameras_fib
from flowr.util.eval import eval_setup


def _get_cam(ref_cam: Cameras, pose: Tensor) -> Cameras:
    new_cam = deepcopy(ref_cam)
    new_cam.camera_to_worlds = torch.from_numpy(pose).float().to(ref_cam.device)
    return new_cam.unsqueeze(0)


def sample_farthest_cameras(cameras: Cameras, num_samples: int) -> torch.Tensor:
    """Farthest point sampling on camera poses (translation + rotation distance)."""
    device = cameras.device
    N = len(cameras)

    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
    distances = torch.full((N,), float("inf"), device=device)
    farthest_index = torch.randint(0, N, (1,), device=device).item()

    c2w = cameras.camera_to_worlds
    for i in range(num_samples):
        sampled_indices[i] = farthest_index
        selected_mask[farthest_index] = True
        ref_c2w = cameras[farthest_index].unsqueeze(0).camera_to_worlds

        t_distance = (ref_c2w[:, None, :3, 3] - c2w[None, :, :3, 3]).norm(dim=-1)
        r_distance = (ref_c2w[:, None, :3, :3] - c2w[None, :, :3, :3]).pow_(2).sum((-1, -2))
        dist = (0.6 * r_distance + 0.1 * t_distance).squeeze()

        distances = torch.minimum(distances, dist)
        distances[selected_mask] = -float("inf")
        farthest_index = torch.argmax(distances).item()

    return sampled_indices


def _subsample_cameras(cameras: Cameras, max_keyframes: int) -> Cameras:
    if len(cameras) <= max_keyframes:
        return cameras
    indices = sample_farthest_cameras(cameras, max_keyframes)
    return Cameras.cat([cameras[i].unsqueeze() for i in indices])


def view_selection(
    num_views: int,
    train_cameras: Cameras,
    method: str = "interpolation",
    max_keyframes: int = 32,
    loop: bool = False,
) -> Cameras:
    """Select novel camera viewpoints for stage 2 dataset generation.

    Args:
        num_views: Number of novel views to generate.
        train_cameras: Training cameras from the scene.
        method: 'interpolation' (B-spline through keyframes) or 'perturbation' (random perturbation of training views).
        max_keyframes: Maximum keyframes for interpolation.
        loop: Whether to loop the interpolation path.

    Returns:
        Cameras object with selected novel viewpoints.
    """
    if method == "interpolation":
        keyframes = _subsample_cameras(train_cameras, max_keyframes)
        num_interp = num_views // (len(keyframes) - 1)
        poses = keyframes.camera_to_worlds.cpu().numpy()
        if loop:
            poses = np.concatenate([poses, poses[:1]], axis=0)
        poses_ab = generate_interpolated_path(poses, num_interp)
    elif method == "perturbation":
        cameras = _subsample_cameras(train_cameras, num_views)
        cameras = perturb_cameras_fib(cameras, train_cameras, (0.2, 0.5), 30)
        poses_ab = cameras.camera_to_worlds.cpu().numpy()
    else:
        raise ValueError(f"Unknown view selection method: {method}. Choose 'interpolation' or 'perturbation'.")

    cameras = Cameras.cat([_get_cam(train_cameras[0], pose) for pose in poses_ab])
    return cameras


def generate_dataset(
    model_path: str,
    output_dir: str,
    input_dir: str | None = None,
    num_views: int = 64,
    method: str = "interpolation",
    visualize: bool = False,
    loop: bool = False,
):
    _update_cfg = None
    input_dir = Path(input_dir) if input_dir is not None else None
    if input_dir is not None:

        def _update_cfg(cfg):
            cfg.pipeline.datamanager.dataparser.data = input_dir
            if scale_path is not None:
                cfg.pipeline.datamanager.dataparser.scale_path = scale_path
            cfg.pipeline.datamanager.dataparser.pointcloud_path = input_dir / "pointcloud.ply"
            return cfg

    scale_path = None
    if input_dir is not None and (input_dir / "scale_factor.txt").exists():
        scale_path = input_dir
    if model_path.endswith(".yml"):
        _, pipeline, _, _ = eval_setup(
            Path(model_path),
            test_mode="inference",
            update_config_callback=_update_cfg,
        )
    elif model_path.endswith(".ply"):
        assert input_dir is not None
        config = instant_splatfacto()
        config.pipeline.datamanager.dataparser = ImageDataParserConfig(
            data=input_dir,
            load_3D_points=True,
            pointcloud_path=input_dir / "pointcloud.ply",
            scale_path=scale_path,
        )
        set_random_seed(config.machine.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = config.pipeline.setup(device=device, test_mode="inference")

        pipeline.model.load_from_ply(model_path)
        pipeline.eval()
    else:
        raise ValueError(f"Invalid model path: {model_path}")

    CONSOLE.log("Copying data...")
    train_cameras, test_cameras, train_filenames, test_filenames = None, None, None, None
    points3D, points3D_rgb = None, None
    for split in ["train", "test"]:
        dpo = pipeline.datamanager.dataparser.get_dataparser_outputs(split=split)
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        for im_file in dpo.image_filenames:
            shutil.copy(im_file, os.path.join(output_dir, split, "images", os.path.basename(im_file)))

        cam_dict = {os.path.basename(im): cam.to_dict() for im, cam in zip(dpo.image_filenames, dpo.cameras)}
        with open(os.path.join(output_dir, f"{split}_cameras.json"), "w") as f:
            json.dump(cam_dict, f)

        if split == "train":
            train_filenames = dpo.image_filenames
            train_cameras = dpo.cameras
            points3D = dpo.metadata["points3D_xyz"]
            points3D_rgb = dpo.metadata["points3D_rgb"]
            to_ply(os.path.join(output_dir, "pointcloud.ply"), points3D, points3D_rgb)
        else:
            test_cameras = dpo.cameras
            test_filenames = dpo.image_filenames

    scale_factor = 1.0
    if scale_path is not None:
        with open(os.path.join(scale_path, "scale_factor.txt"), "r") as f:
            scale_factor = float(f.read())
        with open(os.path.join(output_dir, "scale_factor.txt"), "w") as f:
            f.write(str(1 / scale_factor))

    assert train_cameras is not None and test_cameras is not None

    CONSOLE.log("Selecting cameras...")
    cameras = view_selection(num_views, train_cameras, method=method, loop=loop)

    if visualize:
        if len(points3D) > 100_000:
            subsample = int(len(points3D) / 100_000)
            points3D = points3D[::subsample]
            points3D_rgb = points3D_rgb[::subsample]
        visualize_scene(
            "view_selection",
            {"train": train_cameras, "other": cameras, "test": test_cameras},
            points3D,
            points3D_rgb,
            image_plane_distance=0.1,
            save_path=os.path.join(output_dir, "visualization.rrd"),
        )

    CONSOLE.log("Rendering...")
    cam_dict = {}
    progress_bar = tqdm(range(len(cameras)), desc="Rendering novel cameras")
    for idx, camera in enumerate(cameras):
        os.makedirs(os.path.join(output_dir, "other", "renders"), exist_ok=True)
        pred_outpath = os.path.join(output_dir, "other", "renders", f"{idx:05d}.png")
        camera.camera_to_worlds[:3, 3] /= scale_factor
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera.unsqueeze().to(pipeline.device))
        camera.camera_to_worlds[:, :3, 3] *= scale_factor

        media.write_image(pred_outpath, outputs["rgb"].cpu().numpy())
        cam_dict[f"{idx:05d}.png"] = camera.unsqueeze().to_dict()
        progress_bar.update(1)
    progress_bar.close()

    progress_bar = tqdm(range(len(test_cameras)), desc="Rendering test cameras")
    for filename, camera in zip(test_filenames, test_cameras):
        os.makedirs(os.path.join(output_dir, "test", "renders"), exist_ok=True)
        pred_outpath = os.path.join(output_dir, "test", "renders", os.path.basename(filename))
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera.unsqueeze().to(pipeline.device))
        media.write_image(pred_outpath, outputs["rgb"].cpu().numpy())
        progress_bar.update(1)
    progress_bar.close()

    progress_bar = tqdm(range(len(train_cameras)), desc="Rendering train cameras")
    for filename, camera in zip(train_filenames, train_cameras):
        os.makedirs(os.path.join(output_dir, "train", "renders"), exist_ok=True)
        pred_outpath = os.path.join(output_dir, "train", "renders", os.path.basename(filename))
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera.unsqueeze().to(pipeline.device))
        media.write_image(pred_outpath, outputs["rgb"].cpu().numpy())
        progress_bar.update(1)
    progress_bar.close()

    with open(os.path.join(output_dir, "other_cameras.json"), "w") as f:
        json.dump(cam_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset with renderings to generate views from.")
    parser.add_argument("--model", required=True, type=str, help="Path to reconstruction. Either .yml or .ply")
    parser.add_argument("--output_dir", required=True, type=str, help="Dataset output dir.")
    parser.add_argument("--input_dir", type=str, default=None, help="Dataset input dir.")
    parser.add_argument("--num_views", type=int, default=64, help="Number of views to generate.")
    parser.add_argument(
        "--method", type=str, default="interpolation", help="View selection method: interpolation or perturbation."
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize the view selection.")
    parser.add_argument("--loop", action="store_true", help="Loop the interpolation.")

    parser_args = parser.parse_args()
    generate_dataset(
        parser_args.model,
        parser_args.output_dir,
        parser_args.input_dir,
        parser_args.num_views,
        parser_args.method,
        parser_args.visualize,
        parser_args.loop,
    )
