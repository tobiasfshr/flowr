"""Default DataManager that handles the creation of the Datasets and the multiprocessing Dataloaders."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from nerfstudio.data.datamanagers.base_datamanager import DataManager as NerfstudioDataManager
from nerfstudio.data.datamanagers.base_datamanager import DataManagerConfig as NerfstudioDataManagerConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.rich_utils import CONSOLE
from torch.utils.data import DataLoader, DistributedSampler

from flowr.config.parser import AnnotatedDataParserUnion
from flowr.data.dataset import InputDataset, InputDatasetConfig
from flowr.data.parser.image import ImageDataParserConfig
from flowr.struct.cameras import Cameras
from flowr.util.trainer import to_device
from flowr.util.undistort import parallel_undistort_images

# to avoid issues with open file limit
torch.multiprocessing.set_sharing_strategy("file_system")


@dataclass
class DataManagerConfig(NerfstudioDataManagerConfig):
    """Basic DataManager config."""

    _target: Type = field(default_factory=lambda: DataManager)
    """target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = field(default_factory=ImageDataParserConfig)
    """DataParser config."""
    train_dataset: InputDatasetConfig = field(default_factory=InputDatasetConfig)
    """Training Dataset configuration."""
    eval_dataset: InputDatasetConfig = field(default_factory=InputDatasetConfig)
    """Evaluation Dataset configuration."""
    train_batch_size: int = 1
    """Batch size to use for training."""
    eval_batch_size: int = 1
    """Batch size to use for evaluation."""
    num_train_workers: int = 2
    """Dataloader workers per GPU during training."""
    num_eval_workers: int = 1
    """Dataloader workers per GPU during intermediate eval steps (NOT full eval)."""
    collate_fn: Callable[[List[Dict]], Dict] = field(default_factory=lambda: nerfstudio_collate)
    """Collate function to use for merging multiple data samples."""
    undistort_images: bool = True
    """Undistort all input images if True and Cameras have necessary distortion parameters."""


class DataManager(NerfstudioDataManager):
    """DataManager class.

    It handles the creation of the Datasets and the multiprocessing Dataloaders.
    """

    config: DataManagerConfig
    train_dataset: Optional[InputDataset] = None
    eval_dataset: Optional[InputDataset] = None
    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler] = None
    train_dataparser_outputs: DataparserOutputs
    includes_time: bool = False

    def __init__(
        self,
        config: DataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"

        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.config.dataparser.setup()
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        self._fixed_indices_eval_dataloader = None
        self._fixed_indices_train_dataloader = None

        super().__init__()
        if self.test_mode == "val":
            self.setup_train()
        if self.test_mode != "inference":
            self.setup_eval()

    def create_train_dataset(self) -> InputDataset:
        """Training dataset creation."""
        # run image undistortion and save if cameras have distortion parameters
        if (
            self.train_dataparser_outputs.cameras is not None
            and self.config.undistort_images
            and self.test_mode == "val"
        ):
            CONSOLE.log("Undistorting training images...")
            parallel_undistort_images(self.train_dataparser_outputs)
        return self.config.train_dataset.setup(dataparser_outputs=self.train_dataparser_outputs)

    def create_eval_dataset(self) -> InputDataset:
        """Eval dataset creation."""
        # run image undistortion and save if cameras have distortion parameters
        test_dpo = self.dataparser.get_dataparser_outputs(self.test_split)
        if test_dpo.cameras is not None and self.config.undistort_images and self.test_mode != "inference":
            CONSOLE.log("Undistorting eval images...")
            parallel_undistort_images(test_dpo)
        return self.config.eval_dataset.setup(dataparser_outputs=test_dpo)

    def setup_train(self):
        """Sets up the dataloaders for training."""
        if self.world_size > 0:
            self.train_sampler = DistributedSampler(self.train_dataset, self.world_size, self.local_rank)
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.train_batch_size,
                sampler=self.train_sampler,
                num_workers=self.config.num_train_workers,
                persistent_workers=self.config.num_train_workers > 0,
                collate_fn=self.config.collate_fn,
            )
        else:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.train_batch_size,
                shuffle=True,
                num_workers=self.config.num_train_workers,
                persistent_workers=self.config.num_train_workers > 0,
                collate_fn=self.config.collate_fn,
            )

        self.iter_train_dataloader = iter(self.train_dataloader)

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        if self.world_size > 0:
            self.eval_sampler = DistributedSampler(self.eval_dataset, self.world_size, self.local_rank)
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.eval_batch_size,
                sampler=self.eval_sampler,
                num_workers=self.config.num_eval_workers,
                persistent_workers=self.config.num_eval_workers > 0,
                collate_fn=self.config.collate_fn,
            )
        else:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=True,
                num_workers=self.config.num_eval_workers,
                persistent_workers=self.config.num_eval_workers > 0,
                collate_fn=self.config.collate_fn,
            )

        self.iter_eval_dataloader = iter(self.eval_dataloader)

    @property
    def fixed_indices_eval_dataloader(self):
        """Return dataloader for evaluation with fixed indices."""
        if self._fixed_indices_eval_dataloader is None:
            if self.world_size > 0:
                eval_sampler = DistributedSampler(self.eval_dataset, self.world_size, self.local_rank, shuffle=False)
                eval_dataloader = DataLoader(
                    self.eval_dataset,
                    batch_size=1,
                    sampler=eval_sampler,
                    num_workers=self.config.num_eval_workers,
                    persistent_workers=self.config.num_eval_workers > 0,
                    collate_fn=self.config.collate_fn,
                )
            else:
                eval_dataloader = DataLoader(
                    self.eval_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self.config.num_eval_workers,
                    persistent_workers=self.config.num_eval_workers > 0,
                    collate_fn=self.config.collate_fn,
                )
            self._fixed_indices_eval_dataloader = eval_dataloader
        return iter(self._fixed_indices_eval_dataloader)

    @property
    def fixed_indices_train_dataloader(self):
        """Return dataloader for training with fixed indices."""
        if self._fixed_indices_train_dataloader is None:
            if self.world_size > 0:
                train_sampler = DistributedSampler(self.train_dataset, self.world_size, self.local_rank, shuffle=False)
                train_dataloader = DataLoader(
                    self.train_dataset,
                    batch_size=1,
                    sampler=train_sampler,
                    num_workers=1,
                    persistent_workers=True,
                    collate_fn=self.config.collate_fn,
                )
            else:
                train_dataloader = DataLoader(
                    self.train_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    persistent_workers=True,
                    collate_fn=self.config.collate_fn,
                )
            self._fixed_indices_train_dataloader = train_dataloader
        return iter(self._fixed_indices_train_dataloader)

    def get_train_rays_per_batch(self):
        return 800 * 800

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def next_train(self, step: int) -> Tuple[Optional[Cameras], Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        data = next(self.iter_train_dataloader, None)
        if data is None:
            self.iter_train_dataloader = iter(self.train_dataloader)
            data = next(self.iter_train_dataloader)
        camera = data.pop("camera").to(self.device) if "camera" in data else None
        to_device(data, self.device)
        return camera, data

    def next_eval(self, step: int) -> Tuple[Optional[Cameras], Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        data = next(self.iter_eval_dataloader, None)
        if data is None:
            self.iter_eval_dataloader = iter(self.eval_dataloader)
            data = next(self.iter_eval_dataloader)
        camera = data.pop("camera").to(self.device) if "camera" in data else None
        to_device(data, self.device)
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Optional[Cameras], Dict]:
        """Returns the next evaluation image batch.

        Returns a Camera instead of raybundle"""
        data = next(self.iter_eval_dataloader, None)
        if data is None:
            self.iter_eval_dataloader = iter(self.eval_dataloader)
            data = next(self.iter_eval_dataloader)
        camera = data.pop("camera").to(self.device) if "camera" in data else None
        to_device(data, self.device)
        return camera, data
