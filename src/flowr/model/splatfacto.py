from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Type, Union

import numpy as np
import torch
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.model_components.lib_bilagrid import color_affine_transform
from nerfstudio.models.splatfacto import SplatfactoModel as NSSplatfactoModel
from nerfstudio.models.splatfacto import SplatfactoModelConfig as NSSplatfactoModelConfig
from nerfstudio.utils import colormaps
from plyfile import PlyData
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from flowr.common.io import close_zipfile, open_zipfile
from flowr.common.metric import psnr
from flowr.struct.cameras import Cameras


@dataclass
class SplatfactoModelConfig(NSSplatfactoModelConfig):
    """Splatfacto with diffusion guidance loss."""

    _target: Type = field(default_factory=lambda: SplatfactoModel)

    # hyperparameters
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """

    # rendering
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """

    # training settings
    resolution_schedule: int = 0
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""

    # ADC settings
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    use_absgrad: bool = True
    """Whether to use absgrad to densify gaussians, if False, will use grad rather than absgrad"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    stop_split_at: int = 15000
    """stop splitting at this step"""

    # Camera optimizer settings
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""

    # Appearance embedding settings
    use_appearance_embedding: bool = False
    """If to use appearance embedding."""
    appearance_embed_dim: int = 12  # 3x4 matrix
    """By default a 3x4 matrix for affine color transformation."""
    grid_shape: Tuple[int, int, int] = (16, 16, 8)
    """Shape of the bilateral grid (X, Y, W)"""
    color_corrected_metrics: bool = False
    """If True, apply color correction to the rendered images before computing the metrics."""
    gen_ssim_scale: float = 0.1
    """Scale down factor for SSIM in generated samples."""
    lpips_lambda: float = 0.02
    """LPIPS weight for generated samples."""
    generated_sample_weight: float = 1.0
    """Weight for generated samples."""
    guidance_type: Literal["none", "dataset"] = "none"
    """Type of guidance to use for training."""


class SplatfactoModel(NSSplatfactoModel):
    """Splatfacto with diffusion guidance loss."""

    def __init__(self, *args, num_eval_data: int, **kwargs):
        self.num_eval_data = num_eval_data
        super().__init__(*args, **kwargs)
        self.generated_view_dir = None
        self.train_dataparser_outputs = None

    def _get_downscale_factor(self):
        return 1

    def _set_training_dpo(self, pipeline, step: int):
        if step == 0:
            self.train_dataparser_outputs = deepcopy(pipeline.datamanager.train_dataparser_outputs)
            self.train_cameras = Cameras.cat(
                [c.unsqueeze() for c in self.train_dataparser_outputs.cameras if not c.metadata["is_generated"]]
            )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = super().get_training_callbacks(training_callback_attributes)
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self._set_training_dpo,
                args=[training_callback_attributes.pipeline],
            )
        )
        return cbs

    def populate_modules(self):
        super().populate_modules()

        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="vgg")

        # add optimizable generated view weight
        # replace cam opt / bil grid with one that accounts for both train / eval data
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data + self.num_eval_data, device="cpu"
        )
        if self.config.use_bilateral_grid:
            assert not self.config.use_appearance_embedding

        if self.config.use_appearance_embedding:
            assert not self.config.use_bilateral_grid
            self.embedding_appearance = Embedding(self.num_train_data, self.config.appearance_embed_dim)
            identity = torch.eye(3, 4).flatten().unsqueeze(0).repeat(self.num_train_data, 1)
            self.embedding_appearance.embedding.weight.data = identity

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        assert camera.distortion_params is None, "Distortion parameters are not supported."
        outputs = super().get_outputs(camera)
        outputs["camera"] = camera

        if self.config.use_appearance_embedding:
            if self.training and camera.metadata is not None and "cam_idx" in camera.metadata:
                affine_mat = torch.tanh(self.embedding_appearance(camera.metadata["cam_idx"]).view(1, 1, 3, 4))
            else:
                affine_mat = torch.tanh(self.embedding_appearance.mean(dim=0)).view(1, 1, 3, 4)

            outputs["rgb"] = color_affine_transform(affine_mat, outputs["rgb"])

        return outputs

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        params = super().get_param_groups()
        if self.config.use_appearance_embedding:
            params["appearance_embeddings"] = list(self.embedding_appearance.parameters())
        return params

    def load_from_ply(self, ply_path: Path | str, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        # Default to max iterations for non-training use (inference/export)
        self.step = 30000

        filename = str(ply_path)
        if ".zip/" in filename:
            zip_path, rel_path = filename.split(".zip/")
            z = open_zipfile(zip_path + ".zip")
            with z.open(rel_path) as f:
                plydata = PlyData.read(BytesIO(f.read()))
            close_zipfile(zip_path + ".zip")
        else:
            plydata = PlyData.read(filename)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.config.sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.config.sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        ply_dict = {
            "gauss_params.means": torch.from_numpy(xyz).float(),
            "gauss_params.scales": torch.from_numpy(scales).float(),
            "gauss_params.quats": torch.from_numpy(rots).float(),
            "gauss_params.features_dc": torch.from_numpy(features_dc).float().squeeze(-1),
            "gauss_params.features_rest": torch.from_numpy(features_extra).float().permute(0, 2, 1),
            "gauss_params.opacities": torch.from_numpy(opacities).float(),
        }
        newp = xyz.shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))

        super().load_state_dict(ply_dict, strict=False, **kwargs)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        if self.config.guidance_type == "none":
            return super().get_loss_dict(outputs, batch, metrics_dict)

        assert self.config.guidance_type == "dataset", f"Invalid guidance type {self.config.guidance_type}"

        # apply vanilla loss for initial sparse GT views
        if "is_generated" not in outputs["camera"].metadata or not outputs["camera"].metadata["is_generated"]:
            return super().get_loss_dict(outputs, batch, metrics_dict)

        # Extract generated GT and prediction, apply alternative losses
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        Ll2 = torch.square(gt_img - pred_img).mean()
        gt_img, pred_img = gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...]
        simloss = 1 - self.ssim(gt_img, pred_img)
        perceptual_loss = self.config.lpips_lambda * self.lpips(gt_img, pred_img)
        gen_ssim_lambda = self.config.ssim_lambda * self.config.gen_ssim_scale
        loss_dict = {
            "main_loss": (1 - gen_ssim_lambda) * Ll2 + gen_ssim_lambda * simloss,
            "perceptual_loss": perceptual_loss,
        }

        loss_dict = {k: self.config.generated_sample_weight * v for k, v in loss_dict.items()}
        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.config.use_appearance_embedding:
            app_std, app_mean = torch.std_mean(self.embedding_appearance.embedding.weight)
            metrics_dict["app_embedding_mean"] = app_mean
            metrics_dict["app_embedding_stddev"] = app_std
        return metrics_dict

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        assert not self.config.color_corrected_metrics
        rgb = outputs["rgb"]
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict, _ = super().get_image_metrics_and_images(outputs, batch)
        images_dict = {"rgb": rgb, "rgb_gt": gt_rgb}
        psnr_perpixel = psnr(gt_rgb, rgb, reduction="none").mean(dim=-1, keepdim=True).clamp(0.0, 100.0)
        psnr_perpixel = (psnr_perpixel - psnr_perpixel.min()) / (psnr_perpixel.max() - psnr_perpixel.min())
        psnr_perpixel = colormaps.apply_colormap(psnr_perpixel)
        images_dict.update({"psnr_heatmap": psnr_perpixel})
        return metrics_dict, images_dict
