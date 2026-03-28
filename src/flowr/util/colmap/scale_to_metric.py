import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from nerfstudio.utils.rich_utils import CONSOLE
from scipy.optimize import minimize
from tqdm import tqdm

from flowr.common.io import to_ply
from flowr.util.colmap.model import Camera, Image, read_model, read_points3D_binary


def project_3d_to_depth_map(points_3d: np.ndarray, image: Image, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Converts a 3D point array in world coordinates to a depth map.

    Args:
        points_3d (np.ndarray): Array of shape (N, 3), representing 3D points in world coordinates.
        image (Image): Image object with fields: id, qvec, tvec, camera_id, name, xys, point3D_ids.
        camera (Camera): Camera object with fields: id, model, width, height, params.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the depth map and intrinsic matrix.
    """
    assert camera.model in ["PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"]
    R = image.qvec2rotmat()
    t = image.tvec
    rvec, _ = cv2.Rodrigues(R)

    # Extract intrinsic parameters based on the camera model
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = camera.params[0]
        cx, cy = camera.params[1:3]
        dist_coeffs = np.zeros(4)

    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
        dist_coeffs = np.zeros(4)

    elif camera.model == "SIMPLE_RADIAL":
        fx = fy = camera.params[0]
        cx, cy = camera.params[1:3]
        dist_coeffs = np.array([camera.params[3], 0, 0, 0])

    elif camera.model == "RADIAL":
        fx = fy = camera.params[0]
        cx, cy = camera.params[1:3]
        dist_coeffs = np.array([camera.params[3], camera.params[4], 0, 0])

    elif camera.model == "OPENCV":
        fx, fy, cx, cy = camera.params[:4]
        dist_coeffs = np.array(camera.params[4:8])

    elif camera.model == "FULL_OPENCV":
        fx, fy, cx, cy = camera.params[:4]
        dist_coeffs = np.array(camera.params[4:])

    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")

    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Project points from world coordinates to image coordinates
    image_points, _ = cv2.projectPoints(points_3d, rvec, t, camera_matrix, dist_coeffs)
    image_points = image_points.squeeze()
    u = np.round(image_points[:, 0]).astype(int)
    v = np.round(image_points[:, 1]).astype(int)

    valid_mask = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    u = u[valid_mask]
    v = v[valid_mask]
    depths = points_3d[valid_mask, 2]
    depth_map = np.full((camera.height, camera.width), np.inf)

    # Vectorized operation to populate the depth map
    indices = np.ravel_multi_index((v, u), (camera.height, camera.width))
    np.minimum.at(depth_map.ravel(), indices, depths)
    depth_map[np.isinf(depth_map)] = 0
    return depth_map, camera_matrix


def calculate_scale_factor(colmap_depths: np.ndarray, pred_depths: np.ndarray, confidences: np.ndarray) -> float:
    """Calculate the optimal scale factor to align COLMAP depths with predicted depths.

    Args:
        colmap_depths (np.ndarray): COLMAP depth values.
        pred_depths (np.ndarray): Predicted depth values.
        confidences (np.ndarray): Confidence values associated with predictions.

    Returns:
        float: Optimal scale factor.
    """

    def error_func(scale):
        return np.sum(confidences * np.abs(scale * colmap_depths - pred_depths))

    init_val = np.median(pred_depths / colmap_depths)
    res = minimize(error_func, x0=init_val, method="L-BFGS-B")
    return res.x[0]


def scale_reconstruction(
    model_path: str | Path, image_path: str | Path, device: torch.device = torch.device("cuda")
) -> float:
    """Scales the 3D reconstruction to metric scale using monocular depth priors.

    Args:
        model_path (str): Path to the COLMAP model directory.
        image_path (str): Path to the image directory.
        device (torch.device, optional): Device to run the monocular depth estimation model. Defaults to "cuda".

    Returns:
        float: Scale factor to convert the 3D reconstruction to metric scale.
    """
    model_path = str(model_path)
    image_path = str(image_path)
    cameras, images, points3D = read_model(model_path)
    metric3d_model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_large", pretrain=True).to(device)

    pred_depths, confidences, colmap_depths = [], [], []
    for image_id, image in tqdm(images.items(), desc="Processing images"):
        if not os.path.exists(os.path.join(image_path, image.name)):
            continue
        camera = cameras[image.camera_id]
        # get colmap depth from visible 3D points
        xyz_in_image = []
        for point3D_id in points3D:
            point3D = points3D[point3D_id]
            if image_id in point3D.image_ids:
                xyz_in_image.append(point3D.xyz)
        xyz_in_image = np.array(xyz_in_image)
        depth_map, intrinsic_matrix = project_3d_to_depth_map(xyz_in_image, image, camera)

        # predict depth using monocular depth estimation model
        image_cv2 = cv2.imread(os.path.join(image_path, image.name))[..., [2, 1, 0]]
        height, width = image_cv2.shape[:2]
        scale_factor = camera.width / width
        if scale_factor != 1.0:
            depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_NEAREST)
            intrinsic_matrix[:2] /= scale_factor
        intrinsic = [intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]]
        from flowr.util.metric3d_util import get_prediction, transform_test_data_scalecano

        rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(image_cv2, intrinsic)
        pred_depth, confidence = get_prediction(
            rgb_input, cam_models_stacks, pad, metric3d_model, height, width, label_scale_factor
        )

        mask = (depth_map > 0) & (pred_depth > 0)
        pred_depths.append(pred_depth[mask])
        confidences.append(confidence[mask])
        colmap_depths.append(depth_map[mask])

    colmap_depths = np.concatenate(colmap_depths)
    pred_depths = np.concatenate(pred_depths)
    confidences = np.concatenate(confidences)

    # calculate scale factor
    CONSOLE.log("Calculating scale factor...")
    scale_factor = calculate_scale_factor(colmap_depths, pred_depths, confidences)
    CONSOLE.log(f"Scale factor: {scale_factor}")
    return scale_factor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scale 3D reconstruction to metric scale using monocular depth priors."
    )
    parser.add_argument("model_path", type=str, help="Path to the COLMAP model directory.")
    parser.add_argument("image_path", type=str, help="Path to the image directory.")
    args = parser.parse_args()
    scale_factor = scale_reconstruction(args.model_path, args.image_path)
    CONSOLE.log(scale_factor)
    points3D = read_points3D_binary(os.path.join(args.model_path, "points3D.bin"))
    xyz, rgb = [], []
    for point3D_id in points3D:
        point3D = points3D[point3D_id]
        xyz.append(point3D.xyz)
        rgb.append(point3D.rgb)
    xyz = np.array(xyz)
    rgb = np.array(rgb)
    xyz *= scale_factor
    to_ply("scaled_pointcloud.ply", xyz, rgb)
