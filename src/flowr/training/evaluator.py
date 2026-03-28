import argparse
import copy
import json
import os
import random
import time
from typing import Optional

import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from lpips.lpips import LPIPS
from nerfstudio.model_components.lib_bilagrid import color_correct
from nerfstudio.utils import colormaps
from pytorch_msssim import SSIM
from tqdm.auto import tqdm

from flowr.common.io import save_image
from flowr.common.metric import psnr
from flowr.model.base import BaseModel
from flowr.training.config import Config
from flowr.training.dataset import Dataset, collate_fn
from flowr.training.stats import get_statistics
from flowr.training.util import add_config_args, get_checkpoint_dir, inverse_preproc, override_config_args, setup
from flowr.util.pipeline import average_metrics_list

logger = get_logger(__name__)


@torch.no_grad()
def test_loop(
    args: Config,
    model: BaseModel,
    accelerator: Accelerator,
    mode: str = "test",
    step: int = 0,
    output_dir: Optional[str] = None,
    test_dataset: Optional[Dataset] = None,
):
    """Testing loop.

    Args:
        args (Config): Configuration.
        model (BaseModel): Model.
        accelerator (Accelerator): Accelerator for multi-gpu.
        mode (str, optional): One of test / validation. Defaults to "test".
        step (int, optional): Current step for tracker (e.g. tensorboard). Defaults to 0.
        output_dir (Optional[str], optional): Output dir for saving results. Defaults to None.
    """
    assert mode in ["test", "validation"]

    test_result_folder = "test_results"

    if test_dataset is None:
        test_dataset = Dataset(args, training=False)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=1,
    )

    # Prepare dataloader with our `accelerator`.
    test_dataloader = accelerator.prepare(test_dataloader)

    logger.info(f"***** Running {mode} *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Num batches = {len(test_dataloader)}")

    if len(test_dataloader) == 0:
        logger.info("No test data available, skipping.")
        model.on_test_end()
        return

    if mode == "validation":
        log_idx = random.randint(0, len(test_dataloader) - 1)
        trackers = accelerator.trackers
    else:
        trackers, log_idx = None, -1

    # Create a progress bar
    progress_bar = tqdm(
        range(0, len(test_dataloader)), desc=f"{mode} steps", disable=not accelerator.is_local_main_process
    )
    compute_lpips = LPIPS(net="vgg").to(accelerator.device)
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    metrics_list = []
    begin = time.perf_counter()
    for idx, test_example in enumerate(test_dataloader):
        load_data_time = time.perf_counter() - begin
        gt_images, cond_images = test_example["pixel_values"], test_example["conditioning_pixel_values"]
        with torch.autocast(accelerator.device.type):
            images = model.test_step(test_example)

        pb_logs = {
            "data_time": load_data_time,
            "time": time.perf_counter() - begin,
        }

        # calculate metrics for all target images, save out (and log) images
        for i, gt_path in enumerate(test_example["image_path"][0]):
            # evaluate only the current target view (i=0) if sample based inference
            if args.sample_based_inference and i > 0:
                break

            frac, seq, im = gt_path.split("/")[-5], gt_path.split("/")[-4], os.path.basename(gt_path)
            seq = frac + "_" + seq

            image = images[i : i + 1].to(accelerator.device)
            gt_image = gt_images[:, i]

            gt_image = inverse_preproc(gt_image)
            target_render = inverse_preproc(cond_images[:, i])
            if args.use_color_correction:
                image = color_correct(image.permute(0, 2, 3, 1), gt_image.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            metrics_dict = {
                "psnr": psnr(image, gt_image).item(),
                "ssim": ssim(image, gt_image).item(),
                "lpips": compute_lpips(image, gt_image, normalize=True).item(),
            }

            metrics_dict.update(
                {
                    "psnr_init": psnr(target_render, gt_image).item(),
                    "ssim_init": ssim(target_render, gt_image).item(),
                    "lpips_init": compute_lpips(target_render, gt_image, normalize=True).item(),
                    "sequence": seq,
                }
            )
            metrics_list.append(metrics_dict)

            # save out gt, conditioning and generated image side by side
            psnr_heatmap = psnr(image, gt_image, reduction="none")[0].permute(1, 2, 0).mean(dim=-1, keepdim=True).cpu()
            psnr_heatmap = colormaps.apply_colormap(psnr_heatmap.clamp(0, 60) / 60).permute(2, 0, 1)
            psnr_heatmap_init = (
                psnr(target_render, gt_image, reduction="none")[0].permute(1, 2, 0).mean(dim=-1, keepdim=True).cpu()
            )
            psnr_heatmap_init = colormaps.apply_colormap(psnr_heatmap_init.clamp(0, 60) / 60).permute(2, 0, 1)
            psnr_heatmap = torch.cat([psnr_heatmap, psnr_heatmap_init], -1)

            full_image = torch.cat([gt_image, target_render, image], -1)[0].cpu()
            if output_dir is not None:
                os.makedirs(os.path.join(output_dir, test_result_folder, seq), exist_ok=True)
                save_image(full_image, os.path.join(output_dir, test_result_folder, seq, im))
                save_image(
                    psnr_heatmap, os.path.join(output_dir, test_result_folder, seq, im.replace(".png", "_heatmap.png"))
                )
            # also log one image to tensorboard/wandb
            if trackers is not None and idx == log_idx and accelerator.is_main_process and i == 0:
                for tracker in trackers:
                    if tracker.name in ["tensorboard", "wandb"]:
                        tracker.log_images(
                            {"validation_image": full_image.unsqueeze(0), "psnr_heatmap": psnr_heatmap.unsqueeze(0)},
                            step,
                        )
                    else:
                        logger.warning(f"image logging not implemented for {tracker.name}")

        progress_bar.update(1)
        progress_bar.set_postfix(**pb_logs)
        begin = time.perf_counter()

    progress_bar.close()

    # log average metrics, finish
    metrics_list = accelerator.gather_for_metrics(metrics_list)
    if accelerator.is_main_process:
        metrics_dict = average_metrics_list([_filter_metrics(m) for m in metrics_list])

        # metrics per sparsity level
        if len(test_dataset.get_levels()) > 1:
            for level in test_dataset.get_levels():
                metrics_dict_level = average_metrics_list(
                    [_filter_metrics(m) for m in metrics_list if m["sequence"].startswith(f"{level}_")]
                )
                metrics_dict.update({k + f"_{level}": v for k, v in metrics_dict_level.items()})

        if output_dir is not None:
            save_out_metrics(os.path.join(output_dir, test_result_folder), metrics_list, metrics_dict)
            get_statistics(os.path.join(output_dir, test_result_folder))

        logger.info("\n***** Results ***** \n" + "\n".join(f"{k}: {v}" for k, v in metrics_dict.items()))
        if trackers is not None:
            metrics_dict = {k: v for k, v in metrics_dict.items() if "init" not in k}
            accelerator.log(metrics_dict, step)

    model.on_test_end()


def main(args: Config, config_dir: str, output_dir: Optional[str] = None):
    if output_dir is None:
        output_dir = config_dir
    accelerator, model, output_dir = setup(args, output_dir)

    # Load checkpoint
    ckpt_dir = get_checkpoint_dir(config_dir)
    logger.info(f"Loading checkpoint from {ckpt_dir} ...")
    model.load_pretrained(ckpt_dir)

    model.setup_test(accelerator)
    test_loop(args, model, accelerator, output_dir=output_dir)

    accelerator.end_training()


def _filter_metrics(metrics):
    filtered = copy.deepcopy(metrics)
    if "sequence" in filtered:
        filtered.pop("sequence")
    return filtered


def save_out_metrics(output_dir, metrics_list, metrics_dict):
    sequences = set([m["sequence"] for m in metrics_list])
    for seq in sequences:
        os.makedirs(os.path.join(output_dir, seq), exist_ok=True)
        seq_metrics = average_metrics_list([_filter_metrics(m) for m in metrics_list if m["sequence"] == seq])
        with open(os.path.join(output_dir, seq, "metrics.json"), "w") as f:
            json.dump(seq_metrics, f)

    with open(os.path.join(output_dir, "metrics_list.json"), "w") as f:
        json.dump(metrics_list, f)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="Path to training configuration")
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None, help="eval output dir. By default, it is saved in the config dir."
    )
    add_config_args(Config, parser)
    parser_args = parser.parse_args()

    with open(parser_args.config, "r") as f:
        yaml_content = yaml.load(f, yaml.Loader)
    config = Config(**yaml_content)
    override_config_args(config, parser_args)

    main(config, os.path.dirname(parser_args.config), parser_args.output_dir)
