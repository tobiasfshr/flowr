import argparse
import json
import math
import os
import time

import torch
import yaml
from accelerate.logging import get_logger
from torch import Tensor

from flowr.common.io import save_image
from flowr.training.config import Config
from flowr.training.dataset import Dataset, collate_fn
from flowr.training.util import add_config_args, get_checkpoint_dir, override_config_args, setup

logger = get_logger(__name__)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sequence_path, split, max_inference_images, min_chunks=4):
        self.dataset = dataset
        self.sequence_path = sequence_path
        self.split = split
        assert (
            dataset.args.num_max_ref_images < max_inference_images
        ), f"{dataset.args.num_max_ref_images} (num refs) > {max_inference_images} (max images), please lower num_max_ref_images"
        # get number of target images in sequence
        num_targets = len(dataset._get_cameras(sequence_path, split))
        num_refs = min(dataset.args.num_max_ref_images, len(dataset._get_cameras(sequence_path, "train")))

        if num_targets + num_refs <= max_inference_images:
            self.chunks = [(0, -1)]
        else:
            # split into equal chunks
            num_chunks = math.ceil(num_targets / (max_inference_images - num_refs))
            chunk_size = math.ceil(num_targets / num_chunks)
            self.chunks = []
            for i in range(num_chunks):
                start_id, end_id = i * chunk_size, i * chunk_size + chunk_size
                self.chunks.append((start_id, end_id))

        logger.info("Number of target images: %d", num_targets)
        logger.info("Number of reference images: %d", num_refs)
        logger.info("Number of chunks: %d", len(self.chunks))
        logger.info("Chunk sizes: %s", [end_id - start_id for start_id, end_id in self.chunks])
        logger.info("Sequence path: %s", sequence_path)

        if len(self.chunks) < min_chunks:
            missing_chunks = min_chunks - len(self.chunks)
            for _ in range(missing_chunks):
                self.chunks.append(None)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        if self.chunks[index] is None:
            return None
        start_id, end_id = self.chunks[index]
        return self.dataset.prepare_sequence(self.sequence_path, self.split, start_id, end_id, load_gt=False)


def my_collate(data):
    if data[0] is None:
        return None
    else:
        return collate_fn(data)


@torch.no_grad()
def generate_views(
    args: Config,
    config_dir: str,
    input_dir: str,
    max_inference_images: int = 0,
):
    """Generate views for an augmented training dataset.

    Args:
        args (Config): Configuration.
        model (BaseModel): Model.
        accelerator (Accelerator): Accelerator for multi-gpu.
        input_dir str: Input sequence directory that defines reference views and target
            renders/cameras. This is where the generated views are stored.
        max_inference_images (int): Maximum number of images to generate in a single fwd pass.
            If 0, no limit.
    """
    # Setup model
    accelerator, model, _ = setup(args, config_dir)

    # Load checkpoint
    ckpt_dir = get_checkpoint_dir(config_dir)
    logger.info(f"Loading checkpoint from {ckpt_dir} ...")
    model.load_pretrained(ckpt_dir)

    # setup data
    dataset = SequenceDataset(Dataset(args, training=False), input_dir, "other", max_inference_images)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
        collate_fn=my_collate,
    )
    dataloader = accelerator.prepare(dataloader)

    model.setup_test(accelerator)
    cam_dict = {}
    begin = time.perf_counter()
    for chunk in dataloader:
        if chunk is None:
            continue
        with torch.autocast(accelerator.device.type):
            for k, v in chunk.items():
                if not isinstance(v, Tensor):
                    continue
                chunk[k] = v.to(accelerator.device)
            images = model.test_step(chunk)

        split = chunk["image_path"][0][0].split("/")[-3]
        for i, (im_path, cam) in enumerate(zip(chunk["image_path"][0], chunk["camera"][0])):
            assert split == im_path.split("/")[-3], "Inconsistent split"
            image_name = im_path.split("/")[-1]
            save_image(images[i], os.path.join(input_dir, split, "images", image_name))
            if "is_generated" in cam.metadata:
                del cam.metadata["is_generated"]
            cam_dict[image_name] = cam.to_dict()

    # gather cam dict across processes
    proc_rank = accelerator.process_index
    world_size = accelerator.num_processes
    with open(os.path.join(input_dir, f"_other_cameras_new_{proc_rank}.json"), "w") as f:
        json.dump(cam_dict, f)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # load all cam dicts
        all_cam_dict = {}
        for proc_rank in range(world_size):
            with open(os.path.join(input_dir, f"_other_cameras_new_{proc_rank}.json"), "r") as f:
                all_cam_dict.update(json.load(f))
            os.remove(os.path.join(input_dir, f"_other_cameras_new_{proc_rank}.json"))

        with open(os.path.join(input_dir, "other_cameras_new.json"), "w") as f:
            json.dump(all_cam_dict, f)

    logger.info(f"Time: {time.perf_counter() - begin:.2f}s")

    logger.info(f"Saved generated views to {input_dir}/other/images")
    model.on_test_end()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate views for an augmented training dataset.")
    parser.add_argument("--config", required=True, type=str, help="Path to generative model training configuration")
    parser.add_argument("--input_dir", required=True, type=str, help="Dataset input dir.")
    parser.add_argument("--num_views", type=int, default=64, help="Number of views to generate in a single fwd pass.")
    add_config_args(Config, parser)

    parser_args = parser.parse_args()
    with open(parser_args.config, "r") as f:
        yaml_content = yaml.load(f, yaml.Loader)
    config = Config(**yaml_content)
    override_config_args(config, parser_args)

    generate_views(
        config,
        os.path.dirname(parser_args.config),
        parser_args.input_dir,
        parser_args.num_views,
    )
