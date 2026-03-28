"""Put all workflow configurations here."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import tyro
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from flowr.config.base import ViewerConfig
from flowr.data.manager import DataManagerConfig
from flowr.data.parser.image import ImageDataParserConfig
from flowr.engine.trainer import TrainerConfig
from flowr.model.splatfacto import SplatfactoModelConfig
from flowr.pipeline.default import PipelineConfig

workflow_configs: Dict[str, TrainerConfig] = {}
descriptions = {
    "splatfacto-default": "Default splatfacto training config.",
}
# Add new configurations here, descriptions above


def collate_fn(batch):
    """Collate fn that is compatible with nerfstudio BS=1 training."""
    return batch[0]


## STAGE 1 Initial Reconstruction ##
def instant_splatfacto():
    return TrainerConfig(
        method_name="splatfacto-instant",
        steps_per_eval_image=1000,
        steps_per_eval_batch=500,
        steps_per_save=5000,
        steps_per_eval_all_images=5000,
        max_num_iterations=5001,
        mixed_precision=False,
        pipeline=PipelineConfig(
            datamanager=DataManagerConfig(
                dataparser=ImageDataParserConfig(),
                collate_fn=collate_fn,
                undistort_images=True,
            ),
            model=SplatfactoModelConfig(
                warmup_length=200, stop_split_at=2500, num_downscales=0, sh_degree_interval=1, stop_screen_size_at=1500
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=5000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000, warmup_steps=200, lr_pre_warmup=0
                ),
            },
            "appearance_embeddings": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000, warmup_steps=200, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    )


workflow_configs["splatfacto-instant"] = instant_splatfacto()

## STAGE 2: Refined Reconstruction ##
workflow_configs["splatfacto-default"] = TrainerConfig(
    method_name="splatfacto-default",
    steps_per_eval_image=500,
    steps_per_eval_batch=100,
    steps_per_save=1000,
    steps_per_eval_all_images=1000,
    max_num_iterations=30001,
    mixed_precision=False,
    pipeline=PipelineConfig(
        datamanager=DataManagerConfig(
            dataparser=ImageDataParserConfig(),
            collate_fn=collate_fn,
            undistort_images=True,
        ),
        model=SplatfactoModelConfig(),
    ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "appearance_embeddings": {
            "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "bilateral_grid": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-6, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="tensorboard",
)


# add configs to cli
def sort(workflows, workflow_descriptions):
    """Sort workflows and descriptions by workflow name."""
    workflows = OrderedDict(sorted(workflows.items(), key=lambda x: x[0]))
    workflow_descriptions = OrderedDict(sorted(workflow_descriptions.items(), key=lambda x: x[0]))
    return workflows, workflow_descriptions


all_workflows, all_descriptions = sort(workflow_configs, descriptions)
AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=all_workflows, descriptions=all_descriptions)
    ]
]
