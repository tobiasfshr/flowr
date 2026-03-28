import gc
import os
import random
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import extract_model_from_parallel
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
    retrieve_timesteps,
)
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from torch import nn

from flowr.model.base import BaseModel, get_weight_dtype
from flowr.model.modules import MultiViewTransformer2DModel
from flowr.training.config import Config

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = get_logger(__name__)


class MultiViewFlowModel(BaseModel):
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium"

    def __init__(self, args: Config):
        super().__init__(args)
        self.weighting_scheme = "logit_normal"
        self.logit_mean = 0.0
        self.logit_std = 1.0
        self.mode_scale = 1.29
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
        )
        self.vae.requires_grad_(False)
        MultiViewTransformer2DModel.num_views = args.num_ref_images + 1
        cond_channels = 7 if args.load_rays else 1
        if self.args.use_noise:
            cond_channels += self.vae.latent_channels
        MultiViewTransformer2DModel.cond_channels = cond_channels
        MultiViewTransformer2DModel.use_view_idx = args.use_view_idx

        sd3_transformer = SD3Transformer2DModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="transformer",
            low_cpu_mem_usage=False,
            device_map=None,
        )
        self.transformer = MultiViewTransformer2DModel.from_config(dict(sd3_transformer.config))

        target_state = self.transformer.state_dict()
        pretrained_state = sd3_transformer.state_dict()
        compatible_state = {
            key: value
            for key, value in pretrained_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        self.transformer.load_state_dict(compatible_state, strict=False)
        del sd3_transformer

    def get_trainable_modules(self) -> tuple[nn.Module]:
        return (self.transformer,)

    def get_param_groups(self):
        original_params, multiview_params = [], []
        for name, param in self.transformer.named_parameters():
            if "crossview_" in name:
                multiview_params.append(param)
            else:
                original_params.append(param)
        return {
            "transformer_original": original_params,
            "transformer_multiview": multiview_params,
        }

    def save_pretrained(self, save_path: str):
        transformer = extract_model_from_parallel(self.transformer)
        transformer.save_pretrained(os.path.join(save_path, "MultiViewTransformer2DModel"))

    def load_pretrained(self, input_dir: str):
        load_model = MultiViewTransformer2DModel.from_pretrained(input_dir, subfolder="MultiViewTransformer2DModel")
        if isinstance(self.transformer, torch.nn.parallel.DistributedDataParallel):
            self.transformer.module.register_to_config(**load_model.config)
            self.transformer.module.load_state_dict(load_model.state_dict())
        else:
            self.transformer.register_to_config(**load_model.config)
            self.transformer.load_state_dict(load_model.state_dict())
        del load_model

    def setup_train(self, accelerator: Accelerator) -> None:
        dtype = get_weight_dtype(accelerator)
        device = accelerator.device
        self.vae.to(device, dtype=dtype)
        self.transformer.to(device)
        self.transformer.train()
        if self.args.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.transformer = accelerator.prepare(self.transformer)

    def get_conditioning_inputs(self, batch):
        cond_images = batch["conditioning_pixel_values"].to(dtype=self.vae.dtype)
        batch_size, num_views, channels, height, width = cond_images.shape
        cond_images = cond_images.view(-1, 3, height, width)
        cond_images = self.vae.encode(cond_images).latent_dist.sample()
        cond_images = (cond_images - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        cond_images = cond_images.view(batch_size, num_views, -1, *cond_images.shape[2:])

        if batch["conditioning_ray_values"] is not None:
            ray_maps = batch["conditioning_ray_values"].to(cond_images.dtype)
            cond_images = torch.cat([cond_images, ray_maps], 2)

        if self.training:
            if self.args.tgt_render_drop_rate > 0.0 and random.random() < self.args.tgt_render_drop_rate:
                cond_images[batch["is_target"]] *= 0.0
        elif self.args.tgt_render_drop_rate == 1.0:
            cond_images[batch["is_target"]] *= 0.0

        target_mask = (
            batch["is_target"][:, :, None, None, None].repeat((1, 1, 1, *cond_images.shape[-2:])).to(cond_images.dtype)
        )
        cond_images = torch.cat([cond_images, target_mask], 2)
        return cond_images

    def train_step(self, batch: dict, step: int) -> torch.Tensor:
        target_images = batch["pixel_values"].to(dtype=self.vae.dtype)
        latents = self.vae.encode(target_images.flatten(0, 1)).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latents = latents.view(*target_images.shape[:2], *latents.shape[1:])

        cond_inputs = self.get_conditioning_inputs(batch)

        target_mask = batch["is_target"]
        latents[~target_mask] = cond_inputs[~target_mask][:, : latents.shape[2]]

        if self.args.use_noise:
            noise = torch.randn_like(latents)
        else:
            noise = cond_inputs[:, :, : latents.shape[2]]
            cond_inputs = cond_inputs[:, :, latents.shape[2] :]

        bsz = latents.shape[0]
        if self.args.use_diffusion_forcing:
            latents = latents.flatten(0, 1)
            noise = noise.flatten(0, 1)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=latents.shape[0],
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)

        # Flow matching: z_t = (1 - sigma) * x + sigma * z_0
        sigmas = self._get_sigmas(timesteps, latents.device, n_dim=latents.ndim)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

        if self.args.use_diffusion_forcing:
            latents = latents.view(bsz, -1, *latents.shape[1:])
            noise = noise.view(bsz, -1, *noise.shape[1:])
            noisy_latents = noisy_latents.view(bsz, -1, *noisy_latents.shape[1:])
            sigmas = sigmas.view(bsz, -1, *sigmas.shape[1:])

        hidden_states = torch.cat([noisy_latents, cond_inputs], 2)
        model_pred = self.transformer(
            hidden_states=hidden_states,
            timestep=timesteps,
            return_dict=False,
        )[0]

        # Preconditioning (Section 5 of https://arxiv.org/abs/2206.00364)
        model_pred = model_pred * -sigmas + noisy_latents

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)

        target = latents
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2)[target_mask].reshape(target.shape[0], -1),
            1,
        )
        return {"loss_flow": loss.mean()}

    def setup_test(self, accelerator: Accelerator) -> None:
        dtype = get_weight_dtype(accelerator)
        device = accelerator.device
        self.vae.to(device, dtype=dtype)
        transformer = accelerator.unwrap_model(self.transformer)
        transformer.eval()

        self.pipeline = FlowRPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            transformer=transformer,
            vae=self.vae,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            text_encoder_3=None,
            tokenizer_3=None,
            safety_checker=None,
            torch_dtype=dtype,
        )
        self.pipeline = self.pipeline.to(torch.device(accelerator.device))
        self.pipeline.set_progress_bar_config(disable=True)
        if self.args.seed is None:
            self.generator = None
        else:
            self.generator = torch.Generator(device=self.vae.device).manual_seed(self.args.seed)
        torch.cuda.empty_cache()

    def test_step(self, batch: dict) -> torch.Tensor:
        cond_inputs = self.get_conditioning_inputs(batch)
        latents = cond_inputs

        device, dtype = cond_inputs.device, cond_inputs.dtype
        batch_size, num_views, _, height, width = cond_inputs.shape
        assert batch_size == 1, "only BS=1 inference"

        if self.args.use_noise:
            noise = randn_tensor(
                (batch_size, num_views, self.vae.latent_channels, height, width),
                generator=self.generator,
                device=device,
                dtype=dtype,
            )
            latents = torch.cat([noise, cond_inputs], 2)

        height, width = height * self.pipeline.vae_scale_factor, width * self.pipeline.vae_scale_factor

        image = self.pipeline(
            latents=latents,
            width=width,
            height=height,
            guidance_scale=0.0,
            num_inference_steps=20,
            generator=self.generator,
        ).images
        image = torch.from_numpy(np.array(image).astype(np.float32) / 255).permute(0, 3, 1, 2)
        return image

    def on_test_end(self):
        del self.pipeline
        gc.collect()
        torch.cuda.empty_cache()
        self.transformer.train()

    def _get_sigmas(self, timesteps, device, n_dim=4):
        sigmas = self.noise_scheduler.sigmas.to(device=device)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma


class FlowRPipeline(StableDiffusion3Pipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        assert latents is not None, "We need conditioning inputs as latents"
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = latents.shape[0]

        device = self._execution_device

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        num_channels_latents = self.vae.latent_channels if self.vae is not None else 16
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            latents.dtype,
            device,
            generator,
            latents,
        )

        latents_cond = latents[:, :, num_channels_latents:]
        latents = latents[:, :, :num_channels_latents]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents, latents_cond], 2)

                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        latents = latents.flatten(0, 1)
        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
