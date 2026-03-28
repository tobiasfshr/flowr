import json
import os
import random
import shutil
import time
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm import tqdm

from flowr.common.distributed import dict_to_cpu
from flowr.common.random import set_random_seed
from flowr.config.workflow import instant_splatfacto
from flowr.data.dataset import InputDatasetConfig
from flowr.data.parser.dl3dv10k import DL3DV10KDataParserConfig
from flowr.data.parser.image import ImageDataParserConfig
from flowr.data.transforms import DataKeys
from flowr.scripts.datasets.dl3dv.hf_util import CLEAN_CACHE, TEST_HASHES, download
from flowr.scripts.export import ExportGaussianSplat, ExportGaussianSplatCompressed
from flowr.util.eval import eval_setup
from flowr.util.undistort import parallel_undistort_images


def collate_fn(x):
    return x[0]


def render(config_path: Path, sequence_dir: Path, include_other: bool = True):
    _, pipeline, _, _ = eval_setup(
        Path(config_path),
        test_mode="inference",
    )

    splits = ["train", "test"]
    if include_other:
        splits.insert(1, "other")

    for split in splits:
        CONSOLE.log(f"Generating {split} data...")
        dpo = pipeline.datamanager.dataparser.get_dataparser_outputs(split=split)
        dataset = InputDatasetConfig().setup(dataparser_outputs=dpo)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=1,
            num_workers=1,
        )

        os.makedirs(os.path.join(sequence_dir, split, "renders"), exist_ok=True)
        os.makedirs(os.path.join(sequence_dir, split, "render_depths"), exist_ok=True)
        metrics_dict = {}
        for sample in tqdm(dataloader):
            im_file, camera = sample[DataKeys.PATH], sample[DataKeys.CAM]
            pred_outpath = os.path.join(sequence_dir, split, "renders", os.path.basename(im_file))
            depth_outpath = os.path.join(
                sequence_dir, split, "render_depths", os.path.splitext(os.path.basename(im_file))[0] + ".npy"
            )
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera(camera.to(pipeline.device))

            # compute metrics, write image
            current_dict = pipeline.model.get_metrics_dict(outputs, sample)
            dict_to_cpu(current_dict)
            metrics_dict[os.path.basename(im_file)] = current_dict
            media.write_image(pred_outpath, outputs["rgb"].cpu().numpy())

            # save depth as .npy
            np.save(depth_outpath, outputs["depth"].cpu().numpy())

        with open(os.path.join(sequence_dir, split, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f)


def prepare_data(download_path: Path, sequence_dir: Path, scale_path: Path, sparse_train_fn: callable, include_other: bool = True):
    sparse_frac = sparse_train_fn()
    CONSOLE.log(f"Using {sparse_frac} views for training.")
    dataparser = DL3DV10KDataParserConfig(
        data=download_path,
        load_3D_points=False,
        sparse_training_fraction=sparse_frac,
        downscale_factor=4,
        scale_path=scale_path,
    ).setup()
    splits = ["train", "test"]
    if include_other:
        splits.insert(1, "other")

    for split in splits:
        dpo = dataparser._generate_dataparser_outputs(split=split)
        mask_filenames = dpo.mask_filenames
        depth_filenames = (
            dpo.metadata["depth_filenames"] if dpo.metadata is not None and "depth_filenames" in dpo.metadata else None
        )
        if mask_filenames is not None:
            CONSOLE.log(f"NOTE: Found masks in {download_path}. These will not be converted.")
            dpo.mask_filenames = None

        if depth_filenames is not None:
            CONSOLE.log(f"NOTE: Found depth images in {download_path}. These will not be converted.")
            del dpo.metadata["depth_filenames"]

        # run image undistortion and save if cameras have distortion parameters
        parallel_undistort_images(dpo, save_path=sequence_dir / split, nprocs=4)

        # save out cameras
        camera_dict = {}
        for im_file, camera in zip(dpo.image_filenames, dpo.cameras):
            camera_dict[im_file.name] = camera.to_dict()

        json.dump(camera_dict, open(sequence_dir / f"{split}_cameras.json", "w"))


def reconstruct(work_dir: Path, sequence_dir: Path):
    # workaround to avoid cyclic import
    from nerfstudio.utils import writer

    config = instant_splatfacto()

    # set up config
    config.timestamp = "{timestamp}"
    config.output_dir = work_dir / "models"
    config.experiment_name = os.path.relpath(sequence_dir, work_dir)
    config.pipeline.datamanager.dataparser = ImageDataParserConfig(
        data=sequence_dir,
        load_3D_points=True,
        pointcloud_path=sequence_dir / "pointcloud.ply",
    )
    config.set_timestamp()
    config.print_to_terminal()
    config.save_config()

    set_random_seed(config.machine.seed)
    runner = config.setup(local_rank=0, world_size=1)
    runner.setup()
    runner.run()

    # clear writer buffers
    writer.EVENT_WRITERS = []
    writer.EVENT_STORAGE = []
    writer.GLOBAL_BUFFER = {}
    return config.get_base_dir()


def sparse_train_fn(view_fraction: float | int | tuple[int, int] | tuple[float, float]):
    if isinstance(view_fraction, float):
        return view_fraction
    elif isinstance(view_fraction, int):
        return view_fraction
    elif isinstance(view_fraction, tuple):
        if all(isinstance(i, int) for i in view_fraction):  # int, int --> choose random integer in this interval
            return random.randint(view_fraction[0], view_fraction[1])
        elif all(isinstance(i, float) for i in view_fraction):  # float, float --> choose random float in this interval
            return random.uniform(view_fraction[0], view_fraction[1])
        else:
            raise ValueError(f"Invalid view_fraction tuple: {view_fraction}")
    else:
        raise ValueError(f"Invalid view_fraction: {view_fraction}")


def process_sequence(
    download_item,
    work_dir,
    data_dir,
    scale_dir,
    view_fraction: float | int | tuple[int, int] | tuple[float, float],
    include_other: bool = True,
):
    # check if scale factor is available, if not, skip
    if download_item["name"] in TEST_HASHES:
        assert isinstance(view_fraction, int), "view_fraction must be an integer for test sequences"
        scale_path = Path(os.path.join(scale_dir, "140", download_item["name"]))
    else:
        scale_path = Path(os.path.join(scale_dir, f"{download_item['rel_path'].replace('.zip', '')}"))
    if not os.path.exists(scale_path / "scale_factor.txt"):
        CONSOLE.log(f"Scale factor for {download_item['name']} not found. Skipping.")
        return

    sequence_dir = Path(os.path.join(work_dir, "data", download_item["rel_path"].replace(".zip", "")))
    os.makedirs(sequence_dir, exist_ok=True)

    CONSOLE.log("Downloading...")
    download([download_item], os.path.join(work_dir, "download"), CLEAN_CACHE)
    download_dir = Path(os.path.join(work_dir, "download", download_item["rel_path"].replace(".zip", "")))
    if os.path.exists(download_dir / "nerfstudio"):
        download_dir = download_dir / "nerfstudio"

    CONSOLE.log("Preparing...")
    prepare_data(download_dir, sequence_dir, scale_path, lambda: sparse_train_fn(view_fraction), include_other=include_other)
    CONSOLE.log("Reconstructing...")
    begin = time.perf_counter()
    model_dir = reconstruct(Path(work_dir), sequence_dir)
    CONSOLE.log(f"Reconstruction took {time.perf_counter() - begin:.2f} seconds.")
    torch.cuda.empty_cache()

    CONSOLE.log("Rendering...")
    begin = time.perf_counter()
    render(model_dir / "config.yml", sequence_dir, include_other=include_other)
    CONSOLE.log(f"Rendering took {time.perf_counter() - begin:.2f} seconds.")
    torch.cuda.empty_cache()

    # zip the result
    CONSOLE.log("Zipping...")
    if download_item["name"] in TEST_HASHES:
        assert isinstance(view_fraction, int), "view_fraction must be an integer for test sequences"
        relative_path = f"140/{view_fraction}views/{download_item['name']}"
    else:
        relative_path = download_item["rel_path"].replace(".zip", "")

    shutil.make_archive(os.path.join(data_dir, "data", relative_path), "zip", sequence_dir)

    # Save uncompressed model if in dl3dv140, else save only compressed version
    if download_item["name"] in TEST_HASHES:
        ExportGaussianSplat(model_dir / "config.yml", model_dir / "export").main()
    else:
        ExportGaussianSplatCompressed(model_dir / "config.yml", model_dir / "export").main()

    # copy latest metrics_{step}.json to export
    latest_metrics_file = max(model_dir.glob("metrics_*.json"), key=os.path.getctime)
    shutil.copy(latest_metrics_file, model_dir / "export/metrics.json")
    shutil.make_archive(os.path.join(data_dir, "models", relative_path), "zip", model_dir / "export")

    # delete the downloaded files to free up space
    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(sequence_dir, ignore_errors=True)
    shutil.rmtree(model_dir, ignore_errors=True)
    CONSOLE.log(f"Done with {sequence_dir.name}.")
