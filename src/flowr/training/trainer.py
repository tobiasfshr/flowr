"""Accelerate train script."""
import math
import os
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import diffusers
import hydra
import torch
import transformers
from accelerate.logging import get_logger
from accelerate.utils import extract_model_from_parallel
from diffusers.optimization import get_scheduler
from hydra.core.hydra_config import HydraConfig
from hydra.experimental.callback import Callback
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from flowr.training.config import Config
from flowr.training.dataset import Dataset, collate_fn
from flowr.training.evaluator import test_loop
from flowr.training.sampler import DistributedBucketSampler
from flowr.training.util import remove_checkpoints, remove_validation_result, setup

logger = get_logger(__name__)


class GitSHACallback(Callback):
    """Store current git commit hash value for each run."""

    def on_job_start(self, config: Config, **kwargs: Any) -> None:
        """Callback implementation."""
        hydra_folder = HydraConfig.get().runtime.output_dir
        commit_sha_file = Path(hydra_folder) / "git_sha.txt"
        with open(commit_sha_file, "w") as f:
            try:
                f.write(subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip("\n"))
            except subprocess.CalledProcessError:
                warnings.warn("Could not get git commit sha.")


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(args: Config):
    torch._dynamo.config.optimize_ddp = False

    if args.resume_from_checkpoint is not None:
        output_dir = HydraConfig.get().runtime.config_sources[1].path
    else:
        output_dir = HydraConfig.get().runtime.output_dir

    accelerator, model, output_dir = setup(args, output_dir)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            i = len(weights) - 1
            while len(weights) > 0:
                models[i].save_pretrained(os.path.join(output_dir, type(models[i]).__name__))
                weights.pop()
                i -= 1

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()

            # load diffusers style into model
            load_model = type(model).from_pretrained(input_dir, subfolder=type(model).__name__)
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if args.scale_lr is not None:
        total_batch_size = args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        if args.scale_lr == "linear":
            scale_factor = total_batch_size
        elif args.scale_lr == "sqrt":
            scale_factor = math.sqrt(total_batch_size)
        else:
            raise ValueError(f"Invalid LR scaling method {args.scale_lr}")
        args.learning_rate = args.learning_rate * scale_factor

    # Optimizer creation
    # Use 8-bit AdamW for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    logger.info(f"Initializing optimizer {str(optimizer_class)}")
    optimizer = optimizer_class(
        [
            {
                "params": v,
                "lr": args.learning_rate * args.lr_param_group_multipliers[k]
                if k in args.lr_param_group_multipliers
                else args.learning_rate,
            }
            for k, v in model.get_param_groups().items()
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = Dataset(args, training=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=DistributedBucketSampler(train_dataset, args.train_batch_size, shuffle=True),
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Move model to device and cast to appropriate weight dtype
    model.setup_train(accelerator)

    # load pre-trained weights if any
    if args.ckpt_dir != "":
        model.load_pretrained(args.ckpt_dir)

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    for m in model.get_trainable_modules():
        if extract_model_from_parallel(m).dtype != torch.float32:
            raise ValueError(
                f"Model loaded as datatype {extract_model_from_parallel(m).dtype}. {low_precision_error_string}"
            )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        cfg_dict = OmegaConf.to_container(args, resolve=True)
        job_name = HydraConfig.get().job.config_name.replace(".yaml", "")
        job_name = f"{job_name}_{time.strftime('%Y-%m-%d-%H_%M_%S')}"
        accelerator.init_trackers(args.model_cls, init_kwargs={"wandb": {"name": job_name, "config": cfg_dict}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total parameters: {total_params}")
    logger.info(f"  Trainable parameters: {trainable_params}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    for epoch in range(first_epoch, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(*model.get_trainable_modules()):
                loss_dict = model.train_step(batch, global_step)
                loss = sum(loss_dict.values())
                if not torch.isfinite(loss):
                    raise ValueError(f"Train Loss is not finite: {loss}.")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.get_trainable_params(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            remove_checkpoints(output_dir, args.checkpoints_total_limit)
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % args.validation_steps == 0:
                    model.setup_test(accelerator)
                    save_path = os.path.join(output_dir, f"validation-{global_step}")
                    if accelerator.is_main_process:
                        remove_validation_result(output_dir)
                        os.makedirs(save_path, exist_ok=True)
                    accelerator.wait_for_everyone()
                    test_loop(args, model, accelerator, mode="validation", step=global_step, output_dir=save_path)

            loss_dict = {
                k: accelerator.gather(v.repeat(args.train_batch_size)).mean().item() for k, v in loss_dict.items()
            }
            logs = {
                **loss_dict,
                "lr": lr_scheduler.get_last_lr()[0],
                "data_time": load_data_time,
                "time": time.perf_counter() - begin,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            begin = time.perf_counter()

    progress_bar.close()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model.save_pretrained(os.path.join(output_dir, "final_model"))
    accelerator.wait_for_everyone()

    # Run final test loop.
    model.setup_test(accelerator)
    model.load_pretrained(os.path.join(output_dir, "final_model"))
    test_loop(args, model, accelerator, mode="test", output_dir=output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
