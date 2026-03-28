"""Dummy test file until we have tests implemented."""
import pytest
import torch

from flowr.util.trainer import to_device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_to_device():
    """Test to_device function."""
    data = {
        "tensor1": torch.randn(2, 2),
        "nested": {"tensor2": torch.randn(3), "value": 5},
        "list_of_tensors": [torch.randn(2, 2), [torch.randn(3, 3), torch.randn(4, 4)]],
    }
    device = torch.device("cuda:0")
    to_device(data, device)
    assert data["tensor1"].device == device
    assert data["nested"]["tensor2"].device == device
    assert data["list_of_tensors"][0].device == device
    assert data["list_of_tensors"][1][0].device == device
    assert data["list_of_tensors"][1][1].device == device
