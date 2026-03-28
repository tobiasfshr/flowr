"""Test camera perturbation."""
import torch

from flowr.common.random import set_random_seed
from flowr.common.rotation import euler_angles_to_matrix
from flowr.struct.cameras import Cameras
from flowr.util.cameras import perturb_cameras


def test_perturb_cameras():
    """Test the perturb camera function."""
    c2w = torch.eye(4)[None, :3, :]
    c2w[:, :3, 3] = torch.rand((1, 3))
    c2w[:, :3, :3] = euler_angles_to_matrix(torch.rand((1, 3)) * torch.pi * 2 - torch.pi, convention="XYZ")
    pinhole_camera = Cameras(cx=400.0, cy=400.0, fx=10.0, fy=10.0, camera_to_worlds=c2w)
    new_camera = perturb_cameras(pinhole_camera, translation_range=0.1, rotation_range=10)

    # check if translation within translation range
    translation_diff = torch.norm(
        new_camera.camera_to_worlds[:, :3, 3] - pinhole_camera.camera_to_worlds[:, :3, 3], dim=1
    )
    assert torch.all(translation_diff <= 0.1), "Translation out of range"

    # check if new rotation is within 10 degrees offset
    rotation_diff = torch.matmul(
        new_camera.camera_to_worlds[:, :3, :3], pinhole_camera.camera_to_worlds[:, :3, :3].transpose(1, 2)
    )
    pitch = torch.atan2(-rotation_diff[:, 1, 2], torch.sqrt(rotation_diff[:, 0, 0] ** 2 + rotation_diff[:, 1, 0] ** 2))
    yaw = torch.atan2(rotation_diff[:, 1, 0], rotation_diff[:, 0, 0])
    assert torch.all(pitch <= 10 * torch.pi / 180), "Pitch out of range"
    assert torch.all(yaw <= 10 * torch.pi / 180), "Yaw out of range"

    # check if reproducible when random seed is set
    set_random_seed(42)
    new_camera1 = perturb_cameras(pinhole_camera, translation_range=0.1, rotation_range=10)
    set_random_seed(42)
    new_camera2 = perturb_cameras(pinhole_camera, translation_range=0.1, rotation_range=10)
    assert torch.isclose(new_camera1.camera_to_worlds, new_camera2.camera_to_worlds).all()
