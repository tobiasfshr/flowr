"""Point cloud utilities."""

import torch


def sample_farthest_points(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Performs farthest point sampling (FPS) on a set of 3D points.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) containing N points.
        num_samples (int): Number of points to sample.

    Returns:
        torch.Tensor: Tensor of shape (num_samples,) containing the indices of the sampled points.
    """
    # Ensure points is of shape (N, 3)
    assert points.ndim == 2 and points.size(1) == 3, "points must have shape (N, 3)"

    device = points.device
    N = points.shape[0]

    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
    distances = torch.full((N,), float("inf"), device=device)
    farthest_index = torch.randint(0, N, (1,), device=device).item()

    for i in range(num_samples):
        sampled_indices[i] = farthest_index
        selected_mask[farthest_index] = True  # Mark as selected
        current_point = points[farthest_index].unsqueeze(0)

        dist = torch.sum((points - current_point) ** 2, dim=1)
        distances = torch.minimum(distances, dist)
        distances[selected_mask] = -float("inf")  # Ignore already selected points
        farthest_index = torch.argmax(distances).item()

    return sampled_indices


def sample_closest_points(all_points: torch.Tensor, keyframe_ids: torch.Tensor, num_points: int) -> torch.Tensor:
    """Given a set of points and a subset of keyframe indices, sample `num_points`
    indices from the remaining points that are closest to any of the keyframe points.

    Args:
        all_points (torch.Tensor): Tensor of shape (N, 3) containing all points.
        keyframe_ids (torch.Tensor): 1D tensor of indices (dtype=torch.long)
                                     corresponding to keyframe positions in all_points.
        num_points (int): Number of additional points to sample.

    Returns:
        torch.Tensor: 1D tensor containing indices (from all_points) of the sampled points.
    """
    N = all_points.shape[0]
    device = all_points.device

    # Create an index tensor for all points.
    all_ids = torch.arange(N, device=device)

    # Create a mask that is True for indices not in keyframe_ids.
    mask = torch.ones(N, dtype=torch.bool, device=device)
    mask[keyframe_ids] = False
    remaining_ids = all_ids[mask]

    # If no remaining points, return an empty tensor.
    if remaining_ids.numel() == 0:
        return remaining_ids

    # Get the positions of the remaining points and the keyframes.
    remaining_points = all_points[remaining_ids]  # shape: (R, 3)
    keyframe_points = all_points[keyframe_ids]  # shape: (K, 3)

    # Compute pairwise distances between each remaining point and each keyframe.
    # The result will have shape (R, K).
    diffs = remaining_points[:, None, :] - keyframe_points[None, :, :]
    dists = torch.norm(diffs, dim=2)

    # For each remaining point, compute the minimum distance to any keyframe.
    min_dists, _ = torch.min(dists, dim=1)  # shape: (R,)

    # Get the indices of the remaining points sorted by increasing minimum distance.
    sorted_indices = torch.argsort(min_dists)

    # Select the top `num_points` from the sorted remaining indices.
    selected_remaining_ids = remaining_ids[sorted_indices][:num_points]

    return selected_remaining_ids
