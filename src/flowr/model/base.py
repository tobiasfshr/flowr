import itertools
import os
from abc import ABC, abstractmethod

import torch
from accelerate import Accelerator
from diffusers.models.modeling_utils import get_parameter_device, get_parameter_dtype
from torch import nn

from flowr.training.config import Config


class BaseModel(nn.Module, ABC):
    def __init__(self, args: Config):
        super().__init__()
        self.args = args

    @abstractmethod
    def setup_train(self, accelerator: Accelerator) -> None:
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: dict, step: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def setup_test(self, accelerator: Accelerator) -> None:
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    def on_test_start(self):
        pass

    def on_test_end(self):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def save_pretrained(self, save_path: str) -> None:
        torch.save(self.state_dict(), os.path.join(save_path, "model.pth"))

    def load_pretrained(self, input_dir: str) -> None:
        self.load_state_dict(torch.load(os.path.join(input_dir, "model.pth")))

    def get_trainable_modules(self) -> tuple[nn.Module]:
        return (self,)

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "params": itertools.chain(
                *[list(filter(lambda p: p.requires_grad, m.parameters())) for m in self.get_trainable_modules()]
            )
        }

    def get_trainable_params(self) -> list[nn.Parameter]:
        return itertools.chain(
            *[list(filter(lambda p: p.requires_grad, m.parameters())) for m in self.get_trainable_modules()]
        )

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)


def get_weight_dtype(accelerator: Accelerator) -> torch.dtype:
    """For mixed precision training weights used only for inference are cast to half-precision."""
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    return weight_dtype
