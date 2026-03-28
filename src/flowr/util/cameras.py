import warnings
from copy import deepcopy
from typing import List

import numpy as np
import scipy
import torch

from flowr.common.rotation import euler_angles_to_matrix
from flowr.struct.cameras import Cameras


def three_js_perspective_camera_focal_length(fov: float, image_height: int):
    """Returns the focal length of a three.js perspective camera."""
    if fov is None:
        warnings.warn("fov is None, using default value of 50")
        return 50
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    return focal_length


def rays_to_plucker(rays) -> torch.Tensor:
    """Given rays, return plucker ray representation as [..., 6] Tensor."""
    ray_directions = rays.directions / torch.norm(rays.directions, dim=-1, keepdim=True)
    plucker_normal = torch.cross(rays.origins, ray_directions, dim=-1)
    return torch.cat([plucker_normal, ray_directions], dim=-1)


def perturb_cameras(cameras: Cameras, translation_range: tuple[float, float] | float, rotation_range: float) -> Cameras:
    """Given a set of initial cameras, generate new cameras with random rotation / translation offsets."""
    if isinstance(translation_range, float):
        translation_min = 0.0
    else:
        translation_min = translation_range[0]
        translation_range = translation_range[1] - translation_range[0]
    cameras = deepcopy(cameras)
    random_translation = torch.rand((len(cameras), 3), device=cameras.device) - 0.5
    norm = torch.norm(random_translation, dim=-1, keepdim=True)
    random_translation = (
        random_translation / norm * (torch.rand((1,), device=cameras.device) * translation_range + translation_min)
    )

    random_rotation = torch.rand((len(cameras), 3), device=cameras.device) - 0.5
    random_rotation *= np.deg2rad(rotation_range)
    random_rotation[:, 2] = 0.0
    random_rotation = euler_angles_to_matrix(random_rotation, convention="XYZ")

    random_translation = (cameras.camera_to_worlds[:, :3, :3] @ random_translation.unsqueeze(-1)).squeeze(-1)
    cameras.camera_to_worlds[:, :3, 3] = cameras.camera_to_worlds[:, :3, 3] + random_translation
    cameras.camera_to_worlds[:, :3, :3] = cameras.camera_to_worlds[:, :3, :3] @ random_rotation
    return cameras


def perturb_cameras_fib(
    cameras: Cameras, reference_cams: Cameras, translation_range: tuple[float, float] | float, rotation_range: float
) -> Cameras:
    """Generate new cameras with translation maximizing minimum distance to reference cameras."""
    if isinstance(translation_range, float):
        translation_min = 0.0
    else:
        translation_min = translation_range[0]
        translation_range = translation_range[1] - translation_range[0]
    cameras = deepcopy(cameras)

    ref_positions = cameras.camera_to_worlds[:, :3, 3]
    existing_positions = reference_cams.camera_to_worlds[:, :3, 3]

    radius = (torch.rand((1,)) * translation_range + translation_min).item()
    candidates = _fibonacci_sphere(1000, radius, cameras.device)

    diff = (ref_positions.unsqueeze(1) + candidates.unsqueeze(0)).unsqueeze(2) - existing_positions[None, None]
    diff = torch.norm(diff, dim=-1)
    min_diff, _ = diff.min(dim=-1)
    random_translation = candidates[torch.argmax(min_diff, dim=1)]

    random_rotation = torch.rand((len(cameras), 3), device=cameras.device) - 0.5
    random_rotation *= np.deg2rad(rotation_range)
    random_rotation[:, 2] = 0.0
    random_rotation = euler_angles_to_matrix(random_rotation, convention="XYZ")

    random_translation = (cameras.camera_to_worlds[:, :3, :3] @ random_translation.unsqueeze(-1)).squeeze(-1)
    cameras.camera_to_worlds[:, :3, 3] = cameras.camera_to_worlds[:, :3, 3] + random_translation
    cameras.camera_to_worlds[:, :3, :3] = cameras.camera_to_worlds[:, :3, :3] @ random_rotation
    return cameras


def _fibonacci_sphere(n: int, radius: float = 1.0, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Generates n points uniformly distributed on a sphere using the Fibonacci algorithm."""
    points = []
    offset = 2.0 / n
    increment = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)
        phi = i * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append(torch.tensor([x, y, z], dtype=torch.float32, device=device) * radius)
    return torch.stack(points, dim=0)


def select_distance(
    initial_cameras: Cameras, cameras_to_select_from: Cameras, num_views: int, largest: bool = False
) -> List[int]:
    """Select cameras by combined translation + rotation distance to initial cameras."""
    init_c2w, select_c2w = initial_cameras.camera_to_worlds, cameras_to_select_from.camera_to_worlds
    t_distance = (init_c2w[:, None, :3, 3] - select_c2w[None, :, :3, 3]).norm(dim=-1)
    r_distance = (init_c2w[:, None, :3, :3] - select_c2w[None, :, :3, :3]).pow_(2).sum((-1, -2))
    distance = (0.6 * r_distance + 0.1 * t_distance).mean(dim=0)
    indices = torch.topk(distance, k=num_views, largest=largest)[1]
    return indices.cpu().numpy().tolist()


def generate_interpolated_path(poses, n_interp, spline_degree=3, smoothness=0.01, rot_weight=0.1):
    """Creates a smooth B-spline path between input keyframe camera poses."""

    def _normalize(x):
        return x / np.linalg.norm(x)

    def _viewmatrix(lookdir, up, position):
        vec2 = _normalize(lookdir)
        vec0 = _normalize(np.cross(up, vec2))
        vec1 = _normalize(np.cross(vec2, vec0))
        return np.stack([vec0, vec1, vec2, position], axis=1)

    def _poses_to_points(poses, dist):
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def _points_to_poses(points):
        return np.array([_viewmatrix(p - l, u - p, p) for p, l, u in points])  # noqa: E741

    def _interp(points, n, k, s):
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        return np.reshape(new_points.T, (n, sh[1], sh[2]))

    points = _poses_to_points(poses, dist=rot_weight)
    new_points = _interp(points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness)
    return _points_to_poses(new_points)
