import argparse
import logging
import os
import shutil
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import get_args

import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, broadcast_object_list, set_seed
from torch.nn import functional as F

import flowr.model as model_classes
from flowr.training.config import Config

logger = get_logger(__name__)


def inverse_preproc(image, im_wh=None):
    if im_wh is not None:
        image = F.interpolate(image, size=(im_wh[1], im_wh[0]), mode="bilinear")
    return (image + 1.0) / 2


def setup(args: Config, output_dir: str):
    logging_dir = Path(output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=list(args.experiment_logger),
        project_config=accelerator_project_config,
        dynamo_backend="NO",
    )
    output_dir = broadcast_object_list([output_dir])[0]

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Load model
    logger.info("Initializing model")
    model_cls = getattr(model_classes, args.model_cls)
    model = model_cls(args)
    return accelerator, model, output_dir


def get_checkpoint_dir(model_dir: str) -> str:
    if os.path.exists(os.path.join(model_dir, "final_model")):
        return os.path.join(model_dir, "final_model")
    ckpts = get_checkpoints(model_dir)
    if len(ckpts) > 0:
        return os.path.join(model_dir, ckpts[-1])
    raise FileNotFoundError(f"No checkpoint in {model_dir}")


def get_checkpoints(output_dir: str) -> list[str]:
    """Given an output directory, get all checkpoint dir names that exist."""
    checkpoints = os.listdir(output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    return checkpoints


def remove_validation_result(output_dir: str):
    val_result = os.listdir(output_dir)
    val_result = [d for d in val_result if d.startswith("validation")]
    if len(val_result):
        shutil.rmtree(os.path.join(output_dir, val_result[0]))


def remove_checkpoints(output_dir: str, ckpt_limit: int):
    # before we save a new checkpoint, we need to have at _most_ `ckpt_limit - 1` checkpoints
    checkpoints = get_checkpoints(output_dir)
    if len(checkpoints) >= ckpt_limit:
        num_to_remove = len(checkpoints) - ckpt_limit + 1
        removing_checkpoints = checkpoints[0:num_to_remove]

        logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
            shutil.rmtree(removing_checkpoint)


def add_config_args(cfg_class: type, parser: argparse.ArgumentParser):
    """Add the config arguments to the parser."""

    if not is_dataclass(cfg_class):
        raise TypeError("cfg_class must be a dataclass")

    for field in fields(cfg_class):
        try:
            type(getattr(Config, field.name))
        except Exception:
            continue
        if type(getattr(Config, field.name)) in [tuple, list]:
            assert hasattr(field.type, "__origin__")
            parser.add_argument(
                f"--{field.name}", nargs="+", type=get_args(field.type)[0], help=f"Override the {field.name}"
            )
        else:
            parser.add_argument(
                f"--{field.name}", type=type(getattr(Config, field.name)), help=f"Override the {field.name}"
            )


def override_config_args(config: object, parser_args: argparse.Namespace):
    """Override config fields with the parser args."""

    if not is_dataclass(config):
        raise TypeError("config must be an instance of a dataclass")

    parser_args_dict = vars(parser_args)
    for field in fields(type(config)):
        if field.name not in parser_args_dict:
            continue
        field_value = parser_args_dict[field.name]
        if field_value is not None:
            setattr(config, field.name, field_value)
