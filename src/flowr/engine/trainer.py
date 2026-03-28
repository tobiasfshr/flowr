"""Default Training workflow for 3D Reconstruction."""

import functools
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type, cast

import torch
from nerfstudio.engine.trainer import TRAIN_INTERATION_OUTPUT
from nerfstudio.engine.trainer import Trainer as NerfstudioTrainer
from nerfstudio.engine.trainer import TrainerConfig as NerfstudioTrainerConfig
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter


@dataclass
class TrainerConfig(NerfstudioTrainerConfig):
    """Configuration for default trainer instantiation"""

    _target: Type = field(default_factory=lambda: Trainer)
    """target class to instantiate"""
    scale_lrs: Literal["none", "linear", "sqrt"] = "none"
    """specifies the learning rate scaling method for multi gpu."""
    use_tf32: bool = False
    """If to use tf32 tensor calculations in torch."""
    write_validation_outputs: bool = True
    """If to write validation output metrics in a file."""
    base_dir: Optional[Path] = None
    """Base dir override option."""

    def get_base_dir(self) -> Path:
        """Retrieve the base directory to set relative paths"""
        if self.base_dir is not None:
            return self.base_dir
        # check the experiment and method names
        assert self.method_name is not None, "Please set method name in config or via the cli"
        self.set_experiment_name()
        return Path(f"{self.output_dir}/{self.experiment_name}/{self.method_name}/{self.timestamp}")


class Trainer(NerfstudioTrainer):
    """Default Trainer."""

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        torch.backends.cuda.matmul.allow_tf32 = config.use_tf32
        config.pipeline.resume = config.load_dir is not None or config.load_checkpoint is not None
        super().__init__(config, local_rank, world_size)

    def run(self) -> None:
        """Alias for Trainer.train"""
        self.train()

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        needs_zero = [
            group for group in self.optimizers.parameters if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())

        if not torch.isfinite(loss):
            raise ValueError(f"Train Loss is not finite: {loss_dict}.")

        self.grad_scaler.scale(loss).backward()  # type: ignore
        needs_step = [
            group
            for group in self.optimizers.parameters
            if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
        ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            self.optimizers.zero_grad_all()

            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())

            if "camera_opt" in self.optimizers.optimizers:
                if not torch.isfinite(eval_loss):
                    raise ValueError(f"Eval Loss is not finite: {eval_loss_dict}.")

                # only RGB loss for pose refinement
                eval_loss_dict["main_loss"].backward()  # type: ignore
                # fine-tune eval poses when doing training camera optimization
                self.optimizers.optimizer_step("camera_opt")

            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            output_path = (
                None if not self.config.write_validation_outputs else self.config.get_base_dir() / f"outputs_{step}"
            )
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step, output_path=output_path)
            if self.config.write_validation_outputs:
                files = os.listdir(self.config.get_base_dir())
                if len([f for f in files if f.startswith("outputs_")]) > 1:
                    prev_step = min([int(f.split("_")[1]) for f in files if f.startswith("outputs_")])
                    shutil.rmtree(self.config.get_base_dir() / f"outputs_{prev_step}")

                output_path = self.config.get_base_dir() / f"metrics_{step}.json"
                write_output_json(
                    self.config.experiment_name,
                    self.config.method_name,
                    str(self.config.get_checkpoint_dir()),
                    metrics_dict,
                    output_path,
                )
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint. NOTE: Fixed splatfacto resume."""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                CONSOLE.log("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers = self.setup_optimizers()  # re-init optimizers to update params
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers = self.setup_optimizers()  # re-init optimizers to update params
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")


def write_output_json(exp_name, method_name, ckpt_path, metrics_dict, json_path):
    benchmark_info = {
        "experiment_name": exp_name,
        "method_name": method_name,
        "checkpoint": ckpt_path,
        "results": metrics_dict,
    }
    # Save output to output file
    Path(json_path).write_text(json.dumps(benchmark_info, indent=2), "utf8")
