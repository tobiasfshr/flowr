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
from PIL import Image
from tqdm import tqdm

from flowr.common.distributed import dict_to_cpu
from flowr.common.random import set_random_seed
from flowr.config.workflow import instant_splatfacto
from flowr.data.dataset import InputDatasetConfig
from flowr.data.parser.image import ImageDataParserConfig
from flowr.data.parser.scannetpp import ScanNetppDataParserConfig
from flowr.data.transforms import DataKeys, Resize
from flowr.scripts.export import ExportGaussianSplat, ExportGaussianSplatCompressed
from flowr.struct.cameras import Cameras
from flowr.util.eval import eval_setup
from flowr.util.undistort import parallel_undistort_images

RESOLUTION = 960  # use 960p resolution
VIEW_FRAC_MIN = 0.25
VIEW_FRAC_MAX = 0.5


def collate_fn(x):
    return x[0]


def _resize(data):
    im_file, camera = data
    resize_op = Resize(RESOLUTION, short_edge=False)
    data = {DataKeys.IMAGE: torch.from_numpy(np.array(Image.open(im_file))).float() / 255, DataKeys.CAM: camera}
    resize_op(data)
    Image.fromarray((data[DataKeys.IMAGE].cpu().numpy() * 255).astype(np.uint8)).save(im_file)
    return im_file, data[DataKeys.CAM]


def parallel_resize_images(dpo, nprocs=0):
    input_data = list(zip(dpo.image_filenames, dpo.cameras))
    nprocs = min(nprocs, len(input_data))
    if nprocs > 0:
        with torch.multiprocessing.Pool(processes=nprocs) as pool:
            outputs = []
            with tqdm(total=len(input_data)) as progress_bar:
                for result in pool.imap(_resize, input_data):
                    outputs.append(result)
                    progress_bar.update(1)
            pool.close()
            pool.join()
    else:
        outputs = [_resize(inp) for inp in tqdm(input_data)]

    dpo.image_filenames = [out[0] for out in outputs]
    dpo.cameras = Cameras.cat([out[1].unsqueeze(0) for out in outputs])


def render(config_path: Path, sequence_dir: Path, subset: str, include_other: bool = True):
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


def prepare_data(original_path: Path, sequence_dir: Path, subset: str, include_other: bool = True):
    dataparser = ScanNetppDataParserConfig(
        data=original_path,
        auto_scale_poses=False,
        sparse_training_fraction=1.0 if subset == "val" else random.uniform(VIEW_FRAC_MIN, VIEW_FRAC_MAX),
    ).setup()
    splits = ["train", "test"]
    if include_other:
        splits.insert(1, "other")

    for split in splits:
        dpo = dataparser._generate_dataparser_outputs(split=split)
        if len(dpo.image_filenames) == 0:
            continue

        mask_filenames = dpo.mask_filenames
        depth_filenames = (
            dpo.metadata["depth_filenames"] if dpo.metadata is not None and "depth_filenames" in dpo.metadata else None
        )
        if mask_filenames is not None:
            CONSOLE.log(f"NOTE: Found masks in {original_path}. These will not be converted.")
            dpo.mask_filenames = None

        if depth_filenames is not None:
            CONSOLE.log(f"NOTE: Found depth images in {original_path}. These will not be converted.")
            del dpo.metadata["depth_filenames"]

        # run image undistortion and save if cameras have distortion parameters
        CONSOLE.log(f"Undistorting {split} images...")
        parallel_undistort_images(dpo, save_path=sequence_dir / split, nprocs=4)

        # resize outputs to desired resolution
        CONSOLE.log(f"Resizing {split} images...")
        parallel_resize_images(dpo, nprocs=4)

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
        window_size=2,
        sparse_matching=True,
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


def process_sequence(scene, root, work_dir, data_dir, subset, save_compressed=True, include_other: bool = True):
    sequence_dir = Path(os.path.join(work_dir, "data", scene))
    os.makedirs(sequence_dir, exist_ok=True)

    CONSOLE.log("Prepare data...")
    original_dir = Path(os.path.join(root, "data", scene))
    prepare_data(original_dir, sequence_dir, subset, include_other=include_other)

    CONSOLE.log("Reconstructing...")
    begin = time.perf_counter()
    model_dir = reconstruct(Path(work_dir), sequence_dir)
    CONSOLE.log(f"Reconstruction took {time.perf_counter() - begin:.2f} seconds.")
    torch.cuda.empty_cache()

    CONSOLE.log("Rendering...")
    begin = time.perf_counter()
    render(model_dir / "config.yml", sequence_dir, subset, include_other=include_other)
    CONSOLE.log(f"Rendering took {time.perf_counter() - begin:.2f} seconds.")
    torch.cuda.empty_cache()

    # zip the result
    CONSOLE.log("Zipping...")
    shutil.make_archive(os.path.join(data_dir, "data", scene), "zip", sequence_dir)

    # Save uncompressed model if in dl3dv140, else save only compressed version
    if not save_compressed:
        ExportGaussianSplat(model_dir / "config.yml", model_dir / "export").main()
    else:
        ExportGaussianSplatCompressed(model_dir / "config.yml", model_dir / "export").main()

    # copy latest metrics_{step}.json to export
    latest_metrics_file = max(model_dir.glob("metrics_*.json"), key=os.path.getctime)
    shutil.copy(latest_metrics_file, model_dir / "export/metrics.json")
    shutil.make_archive(os.path.join(data_dir, "models", scene), "zip", model_dir / "export")

    # delete files to free up space
    shutil.rmtree(sequence_dir, ignore_errors=True)
    shutil.rmtree(model_dir, ignore_errors=True)
    CONSOLE.log(f"Done with {sequence_dir.name}.")
