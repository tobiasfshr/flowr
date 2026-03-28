import torch

from flowr.struct.cameras import Cameras
from flowr.util.cameras import select_distance


def test_select_distance():
    def create_cam(positions):
        camera_to_worlds = torch.tensor(
            [[[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]] for x, y, z in positions]
        ).float()
        return Cameras(
            camera_to_worlds=camera_to_worlds,
            fx=300.0,
            fy=300.0,
            cx=64.0,
            cy=64.0,
            width=128,
            height=128,
        )

    initial_cameras = create_cam([(0, 0, 0)])
    cameras_to_select_from = create_cam([(1, 1, 1), (2, 2, 2), (3, 3, 3)])

    expected_indices = [0]  # Closest camera
    result = select_distance(initial_cameras, cameras_to_select_from, num_views=1, largest=False)
    assert result == expected_indices

    expected_indices = [2]  # Farthest camera
    result = select_distance(initial_cameras, cameras_to_select_from, num_views=1, largest=True)
    assert result == expected_indices
