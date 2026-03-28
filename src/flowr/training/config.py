from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class Config:
    # model
    model_cls: str = "MultiViewFlowModel"
    use_noise: bool = False
    """Whether to use noise in the input (i.e. source distribution is Gaussian)."""
    use_diffusion_forcing: bool = False
    """Whether to use diffusion forcing (independent noise level per view)."""
    use_view_idx: bool = True
    """Whether to use view indices as positional embedding."""

    # data
    data_dir: tuple[str] = ("data/",)
    """Folder(s) containing the training / eval data."""
    psnr_bounds: Optional[tuple[float, float]] = None
    """Bounds for PSNR based data filtering."""
    num_ref_images: int = 2
    """Number of reference images to load for current target view(s)."""
    num_tgt_images: int = 2
    """Number of target images to fix/generate."""
    num_total_images: int = 6
    """If >(num. ref + num. tgt), select extra tgt/ref randomly."""
    sample_based_inference: bool = False
    """Whether inference should be sample-based, i.e. we only use num_total_images also during inference and measure only the target image metrics."""
    max_inference_images: tuple[int] = (0,)
    """Maximum number of target images to load per scene *during inference*."""
    num_max_ref_images: int = 12
    """Maximum number of reference images *during inference*."""
    candidate_views_fraction: float = 0.05
    """Number of candidate views to sample from for each target view during training as a fraction of total target views."""
    min_candidates: int = 8
    """Minimum number of candidate views."""
    tgt_render_drop_rate: float = 0.0
    """Ratio of samples where target rendering is dropped. Should only be used when load_rays is True."""
    load_rays: bool = True
    """Whether to load plucker ray representation for target / ref views. Used as additional conditioning input."""
    resolution: int = 512
    """The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."""
    train_batch_size: int = 4
    """Training batch size."""
    dataloader_num_workers: int = 2
    """Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."""
    max_train_samples: Optional[int] = None
    """For debugging purposes or quicker training, truncate the number of training examples to this value if set."""
    train_split_fraction: float = 0.95
    """Fraction of scenes to use for training."""
    num_validation_images: int = 0
    """Validation subset size. If 0, use full test set for validation."""
    gt_overfit: bool = False
    """GT Overfit option for verification."""
    single_view_overfit: bool = False
    """single view overfit option."""
    test_split_paths: tuple[str] = tuple()
    """Path to file defining the test split."""
    invalid_sequences_files: tuple[str] = tuple()
    """Paths to files defining invalid training sequences."""

    # training
    seed: int = 42
    """Random seed."""
    num_train_epochs: int = 1
    """Numer of training epochs."""
    max_train_steps: int = 50_000
    """Total number of training steps to perform. If provided, overrides num_train_epochs."""
    validation_steps: int = 5000
    """Run validation every X steps."""
    ckpt_dir: str = ""
    """Pretrained checkpoint to init the model from."""

    # checkpointing
    checkpointing_steps: int = 5000
    """Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`."""
    checkpoints_total_limit: int = 2
    """Max number of checkpoints to store."""
    resume_from_checkpoint: Optional[str] = None
    """Whether training should be resumed from a previous checkpoint."""

    # training & speed optimizations
    gradient_accumulation_steps: int = 1
    """Number of updates steps to accumulate before performing a backward/update pass."""
    gradient_checkpointing: bool = False
    """Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass."""
    allow_tf32: bool = False
    """Whether or not to allow TF32 on Ampere GPUs. Note that this trades off numerical precision for speed."""
    mixed_precision: str = "bf16"
    """Whether to use mixed precision. Choose between no, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.
        Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. 
        Use this argument to override the accelerate config."""
    set_grads_to_none: bool = False
    """Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain behaviors,
     so disable this argument if it causes any problems."""

    # learning rate & scheduler
    learning_rate: float = 5e-6
    """Initial learning rate (after the potential warmup period) to use."""
    scale_lr: Optional[str] = None
    """Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size: linear or sqrt."""
    lr_scheduler: str = "constant"
    """One of linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup."""
    lr_warmup_steps: int = 500
    """Number of steps for the warmup in the lr scheduler."""
    lr_num_cycles: float = 0.5
    """Number of hard resets of the lr in cosine_with_restarts scheduler."""
    lr_power: float = 1.0
    """Power factor of the polynomial scheduler."""
    lr_param_group_multipliers: dict[str, float] = field(default_factory=dict)
    """Multipliers for the learning rate of different parameter groups."""

    # optimizer & grad
    adam_beta1: float = 0.9
    """The beta1 parameter for the Adam optimizer."""
    adam_beta2: float = 0.999
    """The beta2 parameter for the Adam optimizer."""
    adam_weight_decay: float = 1e-2
    """Weight decay to use."""
    adam_epsilon: float = 1e-08
    """Epsilon value for the Adam optimizer."""
    max_grad_norm: float = 1.0
    """Maximum gradient norm."""
    use_8bit_adam: bool = False
    """If to use 8bit adam version for saving GPU memory."""

    # logging
    logging_dir: str = "logs"
    """Logging subdir."""
    experiment_logger: tuple[str] = ("wandb", "tensorboard")
    """Experiment logger to use: tensorboard or wandb."""

    # metrics
    use_color_correction: bool = False

    def __post_init__(self):
        if self.tgt_render_drop_rate > 0.0:
            assert self.load_rays, "Dropping tgt render requires loading rays"
        assert self.resolution % 8 == 0, "resolution must be divisible by 8"
        assert self.mixed_precision in ["no", "fp16", "bf16"]
        assert self.lr_scheduler in [
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ]


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
