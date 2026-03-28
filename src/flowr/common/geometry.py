"""Geometry related functions."""
import numpy as np
import torch
from torch import Tensor

opencv_to_opengl = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


def inverse_pinhole(intrinsic_matrix: Tensor) -> Tensor:
    """Calculate inverse of pinhole projection matrix.

    Args:
        intrinsic_matrix (Tensor): [..., 3, 3] intrinsics or single [3, 3]
            intrinsics.

    Returns:
        Tensor:  Inverse of input intrinisics.
    """
    squeeze = False
    inv = intrinsic_matrix.clone()
    if len(intrinsic_matrix.shape) == 2:
        inv = inv.unsqueeze(0)
        squeeze = True

    inv[..., 0, 0] = 1.0 / inv[..., 0, 0]
    inv[..., 1, 1] = 1.0 / inv[..., 1, 1]
    inv[..., 0, 2] = -inv[..., 0, 2] * inv[..., 0, 0]
    inv[..., 1, 2] = -inv[..., 1, 2] * inv[..., 1, 1]

    if squeeze:
        inv = inv.squeeze(0)
    return inv


def unproject_points(points: torch.Tensor, depths: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Un-projects pixel coordinates to 3D coordinates with given intrinsics.

    Args:
        points: (N, 2) or (B, N, 2) 2D pixel coordinates.
        depths: (N,) / (N, 1) or (B, N,) / (B, N, 1) depth values.
        intrinsics: (3, 3) or (B, 3, 3) intrinsic camera matrices.

    Returns:
        torch.Tensor: (N, 3) or (B, N, 3) 3D coordinates.

    Raises:
        ValueError: Shape of input points is not valid for computation.
    """
    if len(points.shape) == 2:
        assert len(intrinsics.shape) == 2 or intrinsics.shape[0] == 1, "Got multiple intrinsics for single point set!"
        if len(intrinsics.shape) == 3:
            intrinsics = intrinsics.squeeze(0)
        inv_intrinsics = inverse_pinhole(intrinsics).transpose(0, 1)
        if len(depths.shape) == 1:
            depths = depths.unsqueeze(-1)
        assert len(depths.shape) == 2, "depths must have same dims as points"
    elif len(points.shape) == 3:
        inv_intrinsics = inverse_pinhole(intrinsics).transpose(-2, -1)
        if len(depths.shape) == 2:
            depths = depths.unsqueeze(-1)
        assert len(depths.shape) == 3, "depths must have same dims as points"
    else:
        raise ValueError(f"Shape of input points not valid: {points.shape}")
    hom_coords = torch.cat([points, torch.ones_like(points)[..., 0:1]], -1)
    pts_3d = hom_coords @ inv_intrinsics
    pts_3d *= depths
    return pts_3d


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates=True,
    device=torch.device("cpu"),
) -> torch.Tensor:
    """Generates a coordinate grid for an image.
    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample
    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.
    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(-1, 1, height, device=device, dtype=torch.float)
    else:
        xs = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
    # generate grid by stacking coordinates
    base_grid = torch.stack(torch.meshgrid([xs, ys], indexing="ij")).permute(2, 1, 0).contiguous()
    return base_grid


def depth_to_points(depth_maps: Tensor, intrinsics: Tensor) -> Tensor:
    """Convert depth map(s) to pointcloud(s).

    Args:
        depth_map (Tensor): [B, H, W] or [H, W] depth values.
        intrinsics (Tensor): [B, 3, 3] or [3, 3] intrinsic matrix.

    Returns:
        Tensor: [B, H*W, 3] or [H*W, 3] 3D points.
    """
    squeeze = False
    if len(depth_maps.shape) == 2:
        depth_maps = depth_maps.unsqueeze(0)
        squeeze = True
    batch_size, height, width = depth_maps.shape
    points2d = create_meshgrid(height, width, normalized_coordinates=False, device=depth_maps.device)
    points2d = points2d.view(1, -1, 2).repeat(batch_size, 1, 1)
    points_ref = unproject_points(points2d, depth_maps.view(batch_size, -1), intrinsics)
    if squeeze:
        points_ref = points_ref.squeeze(0)
    return points_ref


def inverse_rigid_transform(transformation: Tensor) -> Tensor:
    """Calculate inverse of rigid body transformation(s).

    Args:
        transformation (Tensor): [..., 3/4, 4] transformations or single [3/4, 4]
            transformation.

    Returns:
        Tensor: Inverse of input transformation(s).
    """
    padded = transformation.shape[-2] == 4
    squeeze = False
    if len(transformation.shape) == 2:
        transformation = transformation.unsqueeze(0)
        squeeze = True
    rotation, translation = transformation[..., :3, :3], transformation[..., :3, 3]
    rot = rotation.transpose(-2, -1)
    t = -rot @ translation[..., None]
    inv = torch.cat([rot, t], -1)
    if padded:
        inv = torch.cat([inv, transformation[..., 3:4, :]], 1)
    if squeeze:
        inv = inv.squeeze(0)
    return inv


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    """Applies transform to points.

    Args:
        points (Tensor): points of shape (N, D) or (B, N, D).
        transform (Tensor): transforms of shape (D+1, D+1) or (B, D+1, D+1).

    Returns:
        Tensor: (N, D) / (B, N, D) transformed points.

    Raises:
        ValueError: Either points or transform have incorrect shape
    """
    hom_coords = torch.cat([points, torch.ones_like(points[..., 0:1])], -1)
    if len(points.shape) == 2:
        if len(transform.shape) == 3:
            assert transform.shape[0] == 1, "Got multiple transforms for single point set!"
            transform = transform.squeeze(0)
        transform = transform.T
    elif len(points.shape) == 3:
        if len(transform.shape) == 2:
            transform = transform.T.unsqueeze(0)
        elif len(transform.shape) == 3:
            transform = transform.permute(0, 2, 1)
        else:
            raise ValueError(f"Shape of transform invalid: {transform.shape}")
    else:
        raise ValueError(f"Shape of input points invalid: {points.shape}")
    points_transformed = hom_coords @ transform
    return points_transformed[..., : points.shape[-1]]
