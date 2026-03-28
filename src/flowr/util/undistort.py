"""Image undistortion for DataparserOutputs."""
import atexit
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from nerfstudio.cameras.camera_utils import fisheye624_project, fisheye624_unproject_helper
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from tqdm import tqdm

from flowr.common.io import get_depth_image_from_path, get_image_mask_from_path
from flowr.struct.cameras import Cameras, CameraType


def _execute(inputs):
    """Wrapper for multiprocessing."""
    return undistort_image(*inputs)


def _undistort(
    camera: Cameras, distortion_params: np.ndarray, data: dict, image: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    mask = None
    if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
        assert distortion_params[3] == 0, (
            "We don't support the 4th Brown parameter for image undistortion, "
            "Only k1, k2, k3, p1, p2 can be non-zero."
        )
        distortion_params = np.array(
            [
                distortion_params[0],
                distortion_params[1],
                distortion_params[4],
                distortion_params[5],
                distortion_params[2],
                distortion_params[3],
                0,
                0,
            ]
        )
        # because OpenCV expects the pixel coord to be top-left, we need to shift the principal point by 0.5
        # see https://github.com/nerfstudio-project/nerfstudio/issues/3048
        K[0, 2] = K[0, 2] - 0.5
        K[1, 2] = K[1, 2] - 0.5
        if np.any(distortion_params):
            newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
            image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
        else:
            newK = K
            roi = 0, 0, image.shape[1], image.shape[0]
        # crop the image and update the intrinsics accordingly
        x, y, w, h = roi
        image = image[y : y + h, x : x + w]
        if "depth_image" in data:
            data["depth_image"] = data["depth_image"][y : y + h, x : x + w]
        if "mask" in data:
            mask = data["mask"]
            mask = mask.astype(np.uint8) * 255
            if np.any(distortion_params):
                mask = cv2.undistort(mask, K, distortion_params, None, newK)  # type: ignore
            mask = mask[y : y + h, x : x + w]
            mask = mask.astype(bool)
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        newK[0, 2] = newK[0, 2] + 0.5
        newK[1, 2] = newK[1, 2] + 0.5
        K = newK

    elif camera.camera_type.item() == CameraType.FISHEYE.value:
        K[0, 2] = K[0, 2] - 0.5
        K[1, 2] = K[1, 2] - 0.5
        distortion_params = np.array(
            [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
        )
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1
        )
        # and then remap:
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        if "mask" in data:
            mask = data["mask"]
            mask = mask.astype(np.uint8) * 255
            mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, newK)
            mask = mask.astype(bool)
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        newK[0, 2] = newK[0, 2] + 0.5
        newK[1, 2] = newK[1, 2] + 0.5
        K = newK
    elif camera.camera_type.item() == CameraType.FISHEYE624.value:
        fisheye624_params = torch.cat(
            [camera.fx, camera.fy, camera.cx, camera.cy, torch.from_numpy(distortion_params)], dim=0
        )
        assert fisheye624_params.shape == (16,)
        assert (
            "mask" not in data
            and camera.metadata is not None
            and "fisheye_crop_radius" in camera.metadata
            and isinstance(camera.metadata["fisheye_crop_radius"], float)
        )
        fisheye_crop_radius = camera.metadata["fisheye_crop_radius"]

        # Approximate the FOV of the unmasked region of the camera.
        upper, lower, left, right = fisheye624_unproject_helper(
            torch.tensor(
                [
                    [camera.cx, camera.cy - fisheye_crop_radius],
                    [camera.cx, camera.cy + fisheye_crop_radius],
                    [camera.cx - fisheye_crop_radius, camera.cy],
                    [camera.cx + fisheye_crop_radius, camera.cy],
                ],
                dtype=torch.float32,
            )[None],
            params=fisheye624_params[None],
        ).squeeze(dim=0)
        fov_radians = torch.max(
            torch.acos(torch.sum(upper * lower / torch.linalg.norm(upper) / torch.linalg.norm(lower))),
            torch.acos(torch.sum(left * right / torch.linalg.norm(left) / torch.linalg.norm(right))),
        )

        # Heuristics to determine parameters of an undistorted image.
        undist_h = int(fisheye_crop_radius * 2)
        undist_w = int(fisheye_crop_radius * 2)
        undistort_focal = undist_h / (2 * torch.tan(fov_radians / 2.0))
        undist_K = torch.eye(3)
        undist_K[0, 0] = undistort_focal  # fx
        undist_K[1, 1] = undistort_focal  # fy
        undist_K[0, 2] = (undist_w - 1) / 2.0  # cx; for a 1x1 image, center should be at (0, 0).
        undist_K[1, 2] = (undist_h - 1) / 2.0  # cy

        # Undistorted 2D coordinates -> rays -> reproject to distorted UV coordinates.
        undist_uv_homog = torch.stack(
            [
                *torch.meshgrid(
                    torch.arange(undist_w, dtype=torch.float32),
                    torch.arange(undist_h, dtype=torch.float32),
                ),
                torch.ones((undist_w, undist_h), dtype=torch.float32),
            ],
            dim=-1,
        )
        assert undist_uv_homog.shape == (undist_w, undist_h, 3)
        dist_uv = (
            fisheye624_project(
                xyz=(
                    torch.einsum(
                        "ij,bj->bi",
                        torch.linalg.inv(undist_K),
                        undist_uv_homog.reshape((undist_w * undist_h, 3)),
                    )[None]
                ),
                params=fisheye624_params[None, :],
            )
            .reshape((undist_w, undist_h, 2))
            .numpy()
        )
        map1 = dist_uv[..., 1]
        map2 = dist_uv[..., 0]

        # Use correspondence to undistort image.
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

        # Compute undistorted mask as well.
        dist_h = camera.height.item()
        dist_w = camera.width.item()
        mask = np.mgrid[:dist_h, :dist_w]
        mask[0, ...] -= dist_h // 2
        mask[1, ...] -= dist_w // 2
        mask = np.linalg.norm(mask, axis=0) < fisheye_crop_radius
        mask = (
            cv2.remap(
                mask.astype(np.uint8) * 255,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            / 255.0
        ).astype(bool)[..., None]
        if len(mask.shape) == 2:
            mask = mask[:, :, None]
        assert mask.shape == (undist_h, undist_w, 1)
        K = undist_K.numpy()
    else:
        raise NotImplementedError("Only perspective and fisheye cameras are supported")

    return K, image, mask


def undistort_image(
    save_path: Path, im_file: Path, camera: Cameras, mask_file: Optional[Path] = None, depth_file: Optional[Path] = None
):
    """Do undistortion of a single image."""
    # create target folders
    image_dir = os.path.join(save_path, "images")
    os.makedirs(image_dir, exist_ok=True)
    new_image_file = os.path.join(image_dir, os.path.basename(im_file))

    if mask_file is not None:
        image_dir = os.path.join(save_path, "masks")
        os.makedirs(image_dir, exist_ok=True)
        new_mask_file = os.path.join(image_dir, os.path.basename(mask_file))
    else:
        new_mask_file = None

    new_depth_file = None
    if depth_file is not None:
        image_dir = os.path.join(save_path, "depth")
        os.makedirs(image_dir, exist_ok=True)
        new_depth_file = os.path.join(image_dir, os.path.basename(depth_file))
    else:
        new_depth_file = None

    # get image / cam info
    K = camera.get_intrinsics_matrices().numpy()
    distortion_params = camera.distortion_params.numpy()
    image = np.array(Image.open(im_file))
    height, width = image.shape[:2]

    # apply undistortion to mask / depth (if any)
    data = {}
    if mask_file is not None:
        # use numpy here to work around python multiprocessing issues with torch
        data["mask"] = get_image_mask_from_path(mask_file)
    if depth_file is not None:
        # strangely, this works with multiprocessing
        data["depth_image"] = torch.from_numpy(get_depth_image_from_path(depth_file, height, width, 1.0))

    # do undistortion
    K, image, mask = _undistort(camera, distortion_params, data, image, K)

    # save out results
    Image.fromarray(image).save(new_image_file)
    if mask is not None:
        mask = mask[..., 0].astype(np.uint8) * 255
        Image.fromarray(mask, "L").save(new_mask_file)
    if "depth_image" in data:
        if not new_depth_file.endswith(".npy"):
            new_depth_file = new_depth_file + ".npy"
        np.save(new_depth_file, data["depth_image"].squeeze(-1).numpy())

    new_image_file = Path(new_image_file)
    if new_mask_file is not None:
        new_mask_file = Path(new_mask_file)
    if new_depth_file is not None:
        new_depth_file = Path(new_depth_file)
    return new_image_file, [K, *image.shape[:2]], new_mask_file, new_depth_file


# TODO multi GPU needs rank zero and broadcast here
def parallel_undistort_images(dpo: DataparserOutputs, save_path: Optional[Path] = None, nprocs: int = 16) -> None:
    """Undistort and save out images/masks/depth maps in parallel, modify dpo in-place."""
    if all(cam.distortion_params is None or (cam.distortion_params == 0.0).all() for cam in dpo.cameras):
        CONSOLE.log("Cameras do not contain any distortion. Skipping undistortion step.")
        return

    mask_filenames = dpo.mask_filenames
    depth_filenames = (
        dpo.metadata["depth_filenames"] if dpo.metadata is not None and "depth_filenames" in dpo.metadata else None
    )

    new_image_files = []
    if mask_filenames is not None:
        new_mask_files = []
    else:
        new_mask_files = None
    if depth_filenames is not None:
        new_depth_files = []
    else:
        new_depth_files = None

    if save_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        atexit.register(tmp_dir.cleanup)
        save_path = tmp_dir.name

    input_data = []
    for idx, im_file in enumerate(dpo.image_filenames):
        data = (
            save_path,
            im_file,
            dpo.cameras[idx],
            mask_filenames[idx] if mask_filenames is not None else None,
            depth_filenames[idx] if depth_filenames is not None else None,
        )
        input_data.append(data)

    nprocs = min(nprocs, len(input_data))
    if nprocs > 0:
        with torch.multiprocessing.Pool(processes=nprocs) as pool:
            outputs = []
            with tqdm(total=len(input_data)) as progress_bar:
                for result in pool.imap(_execute, input_data):
                    outputs.append(result)
                    progress_bar.update(1)
            pool.close()
            pool.join()
    else:
        outputs = [_execute(inp) for inp in tqdm(input_data)]

    for idx, (im_file, Khw, mask_file, depth_file) in enumerate(outputs):
        K, height, width = Khw
        # adjust camera (in-place)
        dpo.cameras.fx[idx] = float(K[0, 0])
        dpo.cameras.fy[idx] = float(K[1, 1])
        dpo.cameras.cx[idx] = float(K[0, 2])
        dpo.cameras.cy[idx] = float(K[1, 2])
        dpo.cameras.height[idx] = height
        dpo.cameras.width[idx] = width
        # remove distortion params
        dpo.cameras.distortion_params = None

        # add paths to lists
        new_image_files.append(im_file)
        if new_mask_files is not None:
            new_mask_files.append(mask_file)
        if new_depth_files is not None:
            new_depth_files.append(depth_file)

    dpo.image_filenames = new_image_files
    dpo.mask_filenames = new_mask_files
    if dpo.metadata is not None and "depth_filenames" in dpo.metadata:
        dpo.metadata["depth_filenames"] = new_depth_files
