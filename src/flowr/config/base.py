"""Simplified version of the nerfstudio Experiment Configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from nerfstudio.configs.base_config import InstantiateConfig, LoggingConfig, MachineConfig
from nerfstudio.configs.base_config import ViewerConfig as NSViewerConfig
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class SimpleExperimentConfig(InstantiateConfig):
    """Simple experiment config.

    This configuration does not assume the workflow to be a subclass of VanillaPipeline,
    and further does not assume sophisticated loggers or visualizers.
    """

    output_dir: Path = Path("outputs")
    """relative or absolute output directory to save all checkpoints and logging"""
    method_name: Optional[str] = None
    """Method name. Required to set in python or via cli"""
    experiment_name: Optional[str] = None
    """Experiment name. If None, will automatically be set to dataset name"""
    project_name: Optional[str] = "nerfstudio-project"
    """Project name."""
    timestamp: str = "{timestamp}"
    """Experiment timestamp."""
    machine: MachineConfig = field(default_factory=MachineConfig)
    """Machine configuration"""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    """Logging configuration"""
    data: Optional[Path] = None
    """Alias for --pipeline.datamanager.data"""
    relative_model_dir: Path = Path("nerfstudio_models/")
    """Relative path to save all checkpoints."""
    max_num_iterations: int = 1000000
    """Maximum number of iterations to run."""

    def set_timestamp(self) -> None:
        """Dynamically set the experiment timestamp"""
        if self.timestamp == "{timestamp}":
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def set_experiment_name(self) -> None:
        """Dynamically set the experiment name"""
        if self.experiment_name is None:
            datapath = self.data
            if datapath is not None:
                datapath = datapath.parent if datapath.is_file() else datapath
                self.experiment_name = str(datapath.stem)
            else:
                self.experiment_name = "unnamed"

    def get_base_dir(self) -> Path:
        """Retrieve the base directory to set relative paths"""
        # check the experiment and method names
        assert self.method_name is not None, "Please set method name in config or via the cli"
        self.set_experiment_name()
        return Path(f"{self.output_dir}/{self.experiment_name}/{self.method_name}/{self.timestamp}")

    def get_checkpoint_dir(self) -> Path:
        """Retrieve the checkpoint directory"""
        return Path(self.get_base_dir() / self.relative_model_dir)

    def print_to_terminal(self) -> None:
        """Helper to pretty print config to terminal"""
        CONSOLE.rule("Config")
        CONSOLE.print(self)
        CONSOLE.rule("")

    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir = self.get_base_dir()
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / "config.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self), "utf8")


@dataclass
class ViewerConfig(NSViewerConfig):
    """ViewerConfig with adapted default port."""

    websocket_port_default: int = 3008
    """The default websocket port to connect to if websocket_port is not specified"""
