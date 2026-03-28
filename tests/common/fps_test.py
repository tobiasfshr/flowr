import torch

from flowr.common.pointcloud import sample_farthest_points


def test_unique_and_valid_indices():
    """Test that the output indices are of the right shape, type, within valid range, and all are unique."""
    N = 100
    num_samples = 99
    points = torch.rand(N, 3)

    indices = sample_farthest_points(points, num_samples)

    # Check the shape and dtype.
    assert indices.shape == (num_samples,)
    assert indices.dtype == torch.long

    # Check that each index is within the valid range [0, N).
    assert torch.all((indices >= 0) & (indices < N))

    # Check that all indices are unique.
    assert len(torch.unique(indices)) == num_samples


def test_sampling_all_points():
    """If num_samples equals the total number of points, then all indices 0..N-1 should appear."""
    N = 50
    points = torch.rand(N, 3)
    indices = sample_farthest_points(points, N)

    # Ensure that every index is unique and covers 0 to N-1.
    assert len(torch.unique(indices)) == N
    assert set(indices.tolist()) == set(range(N))
