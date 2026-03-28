"""Basic Runner class."""
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

from nerfstudio.utils.profiler import setup_profiler
from nerfstudio.utils.writer import put_config, setup_local_writer

from flowr.config.base import SimpleExperimentConfig


@dataclass
class RunnerConfig(SimpleExperimentConfig):
    """Configuration for basic runner."""

    _target: Type = field(default_factory=lambda: Runner)
    """target class to instantiate"""
    load_config: Optional[Path] = None
    """Path to config YAML file."""


class Runner(ABC):
    """Basic Runner.

    It implements the basics of running any workflow in a distributed setting.
    """

    config: RunnerConfig

    def __init__(self, config: RunnerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = config.machine.device_type
        if self.device == "cuda":
            self.device += f":{local_rank}"
        self.base_dir: Path = config.get_base_dir()

    def setup(self) -> None:
        """Setup the Runner by calling other setup functions."""
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        setup_local_writer(self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=None)
        put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        setup_profiler(self.config.logging, writer_log_path)

    @abstractmethod
    def run(self) -> None:
        """Run the program."""
        raise NotImplementedError
