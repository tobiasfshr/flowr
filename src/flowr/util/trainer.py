"""Trainer utils."""
import torch


def to_device(data, device: torch.device | str) -> None:
    """Recursively moves tensors to the specified device."""
    if isinstance(data, torch.Tensor):
        data.data = data.to(device)
    elif isinstance(data, dict):
        for value in data.values():
            to_device(value, device)
    elif isinstance(data, list):
        for value in data:
            to_device(value, device)
