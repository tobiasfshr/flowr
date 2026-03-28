from typing import Any, Dict, List, Literal, Optional, Union

import torch
from accelerate.logging import get_logger
from diffusers import SD3Transformer2DModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward, JointTransformerBlock, _chunked_feed_forward
from diffusers.models.attention_processor import JointAttnProcessor2_0
from diffusers.models.embeddings import PatchEmbed, TimestepEmbedding, Timesteps, get_2d_sincos_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, SD35AdaLayerNormZeroX
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers
from einops import rearrange
from torch import nn

logger = get_logger(__name__)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class Joint3DTransformerBlock(JointTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
    ):
        super(JointTransformerBlock, self).__init__()
        self.use_dual_attention = use_dual_attention
        assert context_pre_only, "This block doesn't support text conditioning"

        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(dim)
        else:
            self.norm1 = AdaLayerNormZero(dim)

        processor = JointAttnProcessor2_0()

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=True,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=1e-6,
        )

        if use_dual_attention:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                processor=processor,
                qk_norm=qk_norm,
                eps=1e-6,
            )
        else:
            self.attn2 = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        # Add 3D Attention layer
        self.crossview_proj = nn.Linear(2 * dim, dim)
        self.crossview_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.crossview_attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=None,
        )

        # zero linear
        self.crossview_zero = zero_module(nn.Linear(dim, dim))

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        conditioning: torch.FloatTensor,
    ):
        # Collapse view dim into batch dim. Following https://arxiv.org/pdf/2403.03206 (Sec 5.3.3)
        batch_size, num_views, _, channels = hidden_states.shape
        hidden_states = rearrange(hidden_states, "b t s c -> (b t) s c")

        # prepare time embedding
        if temb.shape[0] == batch_size:
            temb_flat = temb.unsqueeze(1).repeat(1, num_views, 1).view(batch_size * num_views, channels)
        else:
            temb_flat = temb

        # self attn norm
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                hidden_states, emb=temb_flat
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb_flat)

        # 2D Spatial Attention.
        attn_output = self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=None)

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        # SD3.5 dual attn
        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        # Rearrange, 3D Full Attention. Following https://arxiv.org/pdf/2403.03206 (Sec 5.3.3)
        crossview_input = (
            self.crossview_proj(torch.cat([hidden_states, conditioning], -1))
            if hasattr(self, "crossview_proj")
            else hidden_states + conditioning
        )
        norm_hidden_states = self.crossview_norm(crossview_input)
        norm_hidden_states = rearrange(norm_hidden_states, "(b t) s c -> b (t s) c ", b=batch_size, t=num_views)
        attn_output = self.crossview_attn(hidden_states=norm_hidden_states)

        # zero linear
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            attn_output = _chunked_feed_forward(self.crossview_zero, attn_output, self._chunk_dim, self._chunk_size)
        else:
            attn_output = self.crossview_zero(attn_output)

        attn_output = rearrange(attn_output, "b (t s) c -> (b t) s c ", b=batch_size, t=num_views)
        hidden_states = hidden_states + attn_output

        # Feed forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # unpack
        hidden_states = rearrange(hidden_states, "(b t) s c -> b t s c ", b=batch_size, t=num_views)
        return hidden_states


class MultiViewTransformer2DModel(SD3Transformer2DModel):
    """Transformer model predicting an image from noise, rendering/reference and plucker."""

    num_views: int = 1
    cond_channels: int = 7
    use_view_idx: bool = True
    _no_split_modules = ["Joint3DTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: Optional[str] = None,
    ):
        super(SD3Transformer2DModel, self).__init__()
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = TimestepProjEmbeddings(embedding_dim=self.inner_dim)

        # raymap / target mask / view idx embedding
        crossview_patch_embed = PatchEmbedMulti if self.use_view_idx else PatchEmbed
        self.crossview_embed = crossview_patch_embed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.cond_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.crossview_zero_linear = zero_module(nn.Linear(self.inner_dim * 2, self.inner_dim))

        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        assert self.num_views > 1

        self.transformer_blocks = nn.ModuleList(
            [
                Joint3DTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    context_pre_only=True,
                    qk_norm=qk_norm,
                    use_dual_attention=True if i in dual_attention_layers else False,
                )
                for i in range(self.config.num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_views, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.view(-1, channels, height, width)

        hidden_states, cond_states = hidden_states.split(
            [self.config.in_channels, channels - self.config.in_channels], dim=1
        )

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embedding
        cond_states = (
            self.crossview_embed(cond_states, num_views)
            if isinstance(self.crossview_embed, PatchEmbedMulti)
            else self.crossview_embed(cond_states)
        )
        hidden_states = hidden_states + self.crossview_zero_linear(torch.cat([hidden_states, cond_states], -1))

        _, sequence_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, num_views, sequence_len, hidden_dim)

        temb = self.time_text_embed(timestep)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    cond_states,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    conditioning=cond_states,
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = hidden_states.view(batch_size * num_views, sequence_len, hidden_dim)

        # prepare time embedding
        if temb.shape[0] == batch_size:
            temb = temb.unsqueeze(1).repeat(1, num_views, 1).view(batch_size * num_views, hidden_dim)

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        # multi view
        output = output.view(batch_size, num_views, *output.shape[1:])

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, num=10, den=10000):
    """Get 1D positional embedding for a given grid of positions.

    See https://github.com/huggingface/diffusers/blob/067eab1b3aaf4d09f85edf21d8b147e0980c662a/src/diffusers/models/embeddings.py#L320

    Args:
        embed_dim: int, embedding dimension
        pos: torch.Tensor, (M,)
        num: int, number of frequencies
        den: int, denominator
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = 1.0 / den ** torch.linspace(0, 1, embed_dim // 2, device=pos.device)  # (D/2,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(num * out)  # (M, D/2)
    emb_cos = torch.cos(num * out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class PatchEmbedMulti(PatchEmbed):
    """2D spatial image patch embed with per-view positional encoding."""

    def __init__(
        self,
        *args,
        index_pos_embed: Literal["flexible", "fixed", "none"] = "flexible",
        max_num_views: int = 1000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_num_views = max_num_views
        self.embed_dim = kwargs["embed_dim"]
        self.index_pos_embed = index_pos_embed

    def forward(self, latent, num_views: int):
        dtype, device = latent.dtype, latent.device
        batch_size = latent.shape[0] // num_views

        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(dtype)

        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
            height, width = height // self.patch_size, width // self.patch_size
        else:
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                    output_type="pt",
                ).unsqueeze(0)
            else:
                pos_embed = self.pos_embed

        if self.index_pos_embed == "flexible":
            view_positions = (
                torch.randint(0, self.max_num_views, (batch_size, num_views), device=device).float()
                / self.max_num_views
                * 2
                * torch.pi
                if self.training
                else torch.linspace(0, 2 * torch.pi, num_views, device=device)[None].repeat(batch_size, 1)
            )
            view_embed = get_1d_sincos_pos_embed_from_grid(
                embed_dim=self.embed_dim, pos=view_positions.reshape(-1)
            ).view(batch_size, num_views, -1)
            view_embed = view_embed.reshape(batch_size, num_views, 1, 1, self.embed_dim).repeat(
                1, 1, height, width, 1
            )
        elif self.index_pos_embed == "fixed":
            view_positions = torch.linspace(0, 2 * torch.pi, num_views, device=device)[None].repeat(batch_size, 1)
            view_embed = get_1d_sincos_pos_embed_from_grid(
                embed_dim=self.embed_dim, pos=view_positions.reshape(-1)
            ).view(batch_size, num_views, -1)
            view_embed = view_embed.reshape(batch_size, num_views, 1, 1, self.embed_dim).repeat(
                1, 1, height, width, 1
            )
        elif self.index_pos_embed == "none":
            view_embed = torch.zeros(batch_size, num_views, height, width, self.embed_dim, device=device)
        else:
            raise ValueError(f"Unknown index_pos_embed: {self.index_pos_embed}")

        view_embed = rearrange(view_embed, "b n h w c -> (b n) c h w")
        if self.flatten:
            view_embed = view_embed.flatten(2).transpose(1, 2)

        return (latent + pos_embed + view_embed).to(dtype)
