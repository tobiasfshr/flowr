import math
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from nerfstudio.utils.poses import to4x4
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm

from flowr.common.geometry import inverse_rigid_transform
from flowr.common.io import close_zipfile, load_image, load_json, zip_exists, zip_listdir, zip_valid
from flowr.data.transforms import Crop, DataKeys, HWCtoCHW, Resize
from flowr.struct.cameras import Cameras
from flowr.training.config import Config
from flowr.util.cameras import rays_to_plucker, select_distance


def print_statistics(metrics_dict, metric_key="psnr"):
    all_metrics = {}
    for _, metric in metrics_dict.items():
        for key, value in metric.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(value)

    # make histogram
    hist, bins = np.histogram(all_metrics[metric_key], bins=10)
    CONSOLE.log(f"{metric_key} histogram:")
    for i in range(len(hist)):
        CONSOLE.log(f"{bins[i]:.2f}-{bins[i+1]:.2f}: {hist[i]}")

    # print stddev, max min etc statistical information
    CONSOLE.log(f"mean: {np.mean(all_metrics[metric_key])}")
    CONSOLE.log(f"std: {np.std(all_metrics[metric_key])}")
    CONSOLE.log(f"max: {np.max(all_metrics[metric_key])}")
    CONSOLE.log(f"min: {np.min(all_metrics[metric_key])}")
    CONSOLE.log(f"median: {np.median(all_metrics[metric_key])}")
    CONSOLE.log(f"percentile 25: {np.percentile(all_metrics[metric_key], 25)}")
    CONSOLE.log(f"percentile 75: {np.percentile(all_metrics[metric_key], 75)}")
    CONSOLE.log("\n")


def read_split_file(path) -> List[str]:
    if os.path.basename(path).endswith("dl3dv140.txt"):
        data = pd.read_csv(path, header=None, names=None)
        return data[1].tolist()

    with open(path, "r") as f:
        return f.read().splitlines()


def filter_samples(samples, metrics, bounds):
    filtered_samples = []
    filtered_metrics = {}
    for sample in samples:
        seqpath, split, filename = sample
        key = os.path.basename(seqpath) + "_" + split + "_" + filename
        if bounds[0] <= metrics[key]["psnr"] <= bounds[1]:
            filtered_samples.append(sample)
            filtered_metrics[key] = metrics[key]
    return filtered_samples, filtered_metrics


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: Config, training=False):
        self.args = args
        self.training = training
        self.samples = []
        self.buckets = {}
        self.test_samples = []
        assert len(self.args.max_inference_images) == len(
            self.args.data_dir
        ), "Please provide max_inference_images per dataset"

        if len(args.test_split_paths) == 0:
            split_files = [None] * len(args.data_dir)
        else:
            assert len(args.test_split_paths) == len(args.data_dir)
            split_files = args.test_split_paths

        if len(args.invalid_sequences_files) > 0:
            assert len(args.invalid_sequences_files) == len(args.data_dir)
            invalid_sequences_files = [f if f else None for f in args.invalid_sequences_files]
        else:
            invalid_sequences_files = [None] * len(args.data_dir)

        if self.training:
            total_samples = 0
            for path, split_file, invalid_file in zip(args.data_dir, split_files, invalid_sequences_files):
                samples, metrics = self._get_samples(path, split_file, invalid_file)
                assert len(metrics) == len(samples), "Number of samples and metrics do not match"
                if len(samples) == 0:
                    continue

                # print out statistics
                CONSOLE.log(f"Showing {'train' if self.training else 'test'} dataset statistics")
                print_statistics(metrics)

                # filter by psnr bounds
                if args.psnr_bounds is not None:
                    num_samples = len(samples)
                    samples, metrics = filter_samples(samples, metrics, args.psnr_bounds)
                    CONSOLE.log(f"Number of filtered samples by applying PSNR bounds: {num_samples - len(samples)}")

                self.buckets[path] = [i + total_samples for i in range(len(samples))]
                self.samples.extend(samples)
                total_samples += len(samples)

            CONSOLE.log(f"Total training samples: {total_samples}")

            assert self.args.max_train_samples is None, "Not yet implemented."
        else:
            samples = []
            sequences = []
            for path, split_file, max_inf_images in zip(args.data_dir, split_files, args.max_inference_images):
                samples.extend(self._get_samples(path, split_file)[0])
                # compute samples per sequence
                samples_per_seq = {}
                for sample in samples:
                    seqpath = sample[0]
                    if seqpath not in samples_per_seq:
                        samples_per_seq[seqpath] = 0
                    samples_per_seq[seqpath] += 1

                # chunk all sequences above max_inf_images
                for seqpath, num_samples in samples_per_seq.items():
                    if num_samples <= max_inf_images:
                        sequences.append((seqpath, path, 0, num_samples))
                        continue
                    # split into equal chunks
                    num_chunks = math.ceil(num_samples / max_inf_images)
                    num_per_chunk = math.ceil(num_samples / num_chunks)
                    for i in range(num_chunks):
                        sequences.append(
                            (seqpath, path, i * num_per_chunk, min(i * num_per_chunk + num_per_chunk, num_samples))
                        )

            self.test_samples = sequences

            if self.args.sample_based_inference:
                self.test_samples = samples

            if self.args.num_validation_images > 0 and len(self.test_samples) > self.args.num_validation_images:
                self.test_samples = random.sample(self.test_samples, self.args.num_validation_images)

        assert not self.args.single_view_overfit, "Not yet implemented."

        self.transforms = [
            Resize(args.resolution, short_edge=False),
            Crop(size_divisor=16, method="random" if training else "center"),
            HWCtoCHW(),
        ]
        self.num_ref_images = self.args.num_ref_images
        self.num_tgt_images = self.args.num_tgt_images
        # if to normalize tgt <-> first ref pose to unit norm
        self.scale_poses = False

    def _get_sequences(self, root, test_split_file=None, invalid_sequence_file=None):
        # find all zip files in root (recursive), or all directories that contain a 'train' and 'test' folder
        sequences = []
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                if f.endswith(".zip"):
                    sequences.append(os.path.join(dirpath, f))
            if "train" in dirnames and "test" in dirnames:
                sequences.append(dirpath)

        if test_split_file is None:
            max_ind = max(1, int(len(sequences) * self.args.train_split_fraction))
            train_sequences = sequences[:max_ind]
            test_sequences = sequences[max_ind:]
        else:
            test_split = read_split_file(test_split_file)
            test_sequences = [s for s in sequences if os.path.basename(s).replace(".zip", "") in test_split]
            train_sequences = [s for s in sequences if s not in test_sequences]

        if invalid_sequence_file is not None:
            with open(invalid_sequence_file, "r") as f:
                invalid_sequences = f.read().splitlines()
            train_sequences = [
                s for s in train_sequences if os.path.basename(s).replace(".zip", "") not in invalid_sequences
            ]

        if self.training:
            return train_sequences
        else:
            return test_sequences

    def _get_samples_per_sequence(self, seq: str):
        # get all samples in a sequence:
        splits = ["other", "test"] if self.training else ["test"]
        seqname = os.path.basename(seq).replace(".zip", "")
        if seq.endswith(".zip") and not zip_valid(seq, close=False):
            close_zipfile(seq)
            return [], {}, seqname
        all_metrics = {}
        seq_samples = []
        for split in splits:
            listdir = lambda x: zip_listdir(x, close=False) if seq.endswith(".zip") else os.listdir(x)
            exists = lambda x: zip_exists(x, close=False) if seq.endswith(".zip") else os.path.exists(x)

            if (
                not exists(os.path.join(seq, split))
                or not exists(os.path.join(seq, split, "images"))
                or not exists(os.path.join(seq, split, "renders"))
            ):
                close_zipfile(seq)
                return [], {}, seqname

            if exists(os.path.join(seq, split, "metrics.json")):
                metrics = load_json(os.path.join(seq, split, "metrics.json"), close=False)
                metrics = {os.path.basename(seq) + "_" + split + "_" + k: v for k, v in metrics.items()}
                all_metrics.update(metrics)
            for i, im in enumerate(listdir(os.path.join(seq, split, "images"))):
                if i == 0:  # only test once per split
                    x = load_image(os.path.join(seq, split, "images", im), close=False)
                    width, height = x.size
                    if width <= height:
                        CONSOLE.log(
                            f"wrong aspect ratio {width}x{height} in processing {os.path.join(seq, split, 'images', im)}"
                        )
                        close_zipfile(seq)
                        return [], {}, seqname

                if not exists(os.path.join(seq, split, "renders", im)):
                    close_zipfile(seq)
                    return [], {}, seqname
                seq_samples.append((seq, split, im))

        close_zipfile(seq)
        return seq_samples, all_metrics, seqname

    def _get_samples(self, root: str, split_file=None, invalid_sequences_file=None, nthreads: int = 64):
        seq_paths = self._get_sequences(root, split_file, invalid_sequences_file)
        if not len(seq_paths):
            return [], {}

        cache_filename = "train_cache.pkl" if self.training else "test_cache.pkl"
        if os.path.exists(os.path.join(root, cache_filename)):
            with open(os.path.join(root, cache_filename), "rb") as f:
                seq_paths_cache, samples, metrics = pickle.loads(f.read())

            if set(seq_paths) == set(seq_paths_cache):
                return samples, metrics
            else:
                CONSOLE.log("WARNING: cache file does not match current data directory, recomputing samples")

        # sequence processing
        samples, invalid_seqs, results, metrics = [], [], [], {}
        with ThreadPoolExecutor(max_workers=min(len(seq_paths), nthreads)) as executor:
            future_to_zip = {executor.submit(self._get_samples_per_sequence, seq): seq for seq in seq_paths}
            for future in tqdm(as_completed(future_to_zip), total=len(future_to_zip), desc="Generating dataset index"):
                result = future.result()
                results.append(result)
        for result in results:
            if len(result[0]) == 0:
                invalid_seqs.append(result[-1])
            else:
                samples.extend(result[0])
                metrics.update(result[1])

        if len(invalid_seqs) > 0:
            CONSOLE.log(f"WARNING: found invalid sequences!\n{invalid_seqs}")
        assert len(seq_paths) - len(invalid_seqs) > 0, "No valid sequences in data directory"

        # save to cache
        with open(os.path.join(root, cache_filename), "wb") as f:
            f.write(pickle.dumps((seq_paths, samples, metrics)))

        return samples, metrics

    def _get_cameras(self, path: str, split: str) -> dict:
        loaded_dict = load_json(os.path.join(path, f"{split}_cameras.json"))
        # order this dict by image name
        return dict(sorted(loaded_dict.items()))

    def _get_closest_train_views(self, sequence_path: str, target_cameras: list[Cameras], num_views: int):
        im_filenames, train_cameras = [], []
        for k, v in self._get_cameras(sequence_path, "train").items():
            im_filenames.append(k)
            train_cameras.append(Cameras.from_dict(v).unsqueeze())

        # load all images if num_views is -1 or num_views greater than number of training images
        if num_views < 0 or num_views >= len(train_cameras):
            return [
                load_image(os.path.join(sequence_path, "train", "images", im)) for im in im_filenames
            ], train_cameras

        # cluster target cameras into num_views clusters
        if num_views < len(target_cameras):
            c2ws = torch.cat([c.camera_to_worlds for c in target_cameras])
            pos_fwd = torch.cat([c2ws[:, :3, 3], c2ws[:, :3, 2]], axis=1)
            centers = KMeans(n_clusters=num_views, random_state=0, n_init="auto").fit(pos_fwd).cluster_centers_
            tgt_cams = []
            for center in centers:
                closest_id = torch.argmin(torch.norm(pos_fwd - center, dim=1))
                tgt_cams.append(target_cameras[closest_id])
            target_cameras = tgt_cams
        elif num_views > len(target_cameras):
            # duplicate target cameras if not enough
            indices = np.linspace(0, len(target_cameras) - 0.5, num_views, dtype=int)
            target_cameras = [target_cameras[i] for i in indices]

        assert len(target_cameras) == num_views

        closest_cams, images = [], []
        for tgt_cam in target_cameras:
            closest_id = get_closest_cameras(tgt_cam, train_cameras, 1)[0]
            images.append(load_image(os.path.join(sequence_path, "train", "images", im_filenames.pop(closest_id))))
            closest_cams.append(train_cameras.pop(closest_id))

        assert len(images) == len(closest_cams) == num_views
        return images, closest_cams

    def _preprocess_image(self, image: Optional[Image.Image] = None, camera: Optional[Cameras] = None):
        data = {}
        if image is not None:
            data[DataKeys.IMAGE] = torch.from_numpy(np.array(image, dtype=np.float32)) / 255.0
        if camera is not None:
            data[DataKeys.CAM] = camera
        assert len(data) > 0, "No data to preprocess"
        for t in self.transforms:
            t(data)
        if image is not None:
            # normalize to [-1, 1] interval
            image = (data[DataKeys.IMAGE] - 0.5) * 2
        if camera is not None:
            camera = data[DataKeys.CAM]
        return image, camera

    def get_levels(self):
        if self.training:
            return []

        fracs = set()
        for sample in self.test_samples:
            frac = os.path.basename(os.path.dirname(sample[0]))
            fracs.add(frac)
        return list(fracs)

    def _load_rays(self, target_cameras, ref_cameras):
        scale_factor = 1.0
        if self.scale_poses:
            scale_factor = (
                ref_cameras[0].camera_to_worlds[0, :3, 3] - ref_cameras[1].camera_to_worlds[0, :3, 3]
            ).norm()
        rays = []
        if self.training:
            idx = random.randint(0, len(ref_cameras) - 1)
        else:
            # find ref cam closest to centroid
            centroid = torch.mean(torch.cat([c.camera_to_worlds[:, :3, 3] for c in target_cameras + ref_cameras], 0), 0)
            idx = torch.argmin(
                torch.norm(torch.cat([c.camera_to_worlds[:, :3, 3] for c in ref_cameras], 0) - centroid, dim=1)
            )

        world_to_cam = to4x4(inverse_rigid_transform(ref_cameras[idx].camera_to_worlds[0]))
        for cam in [*target_cameras, *ref_cameras]:
            # OpenGL rays at 1/8 (latent) resolution, coord frame of the ref cam that is most center
            cam.rescale_output_resolution(1 / 8)
            cam.camera_to_worlds = (world_to_cam @ to4x4(cam.camera_to_worlds[0]))[None, :3]
            cam.camera_to_worlds[..., 3] *= scale_factor
            rays_cam = cam.generate_rays(0)
            rays.append(rays_to_plucker(rays_cam).permute(2, 0, 1))

        return torch.stack(rays).contiguous()

    def _test_get_item(self, idx: int):
        sequence_path, _, start_idx, end_idx = self.test_samples[idx]
        return self.prepare_sequence(sequence_path, "test", start_idx, end_idx)

    def prepare_sequence(self, sequence_path: str, split: str, start_idx: int = 0, end_idx: int = -1, load_gt=True):
        # get all test images/cameras as target within [start_idx, end_idx]
        cam_dict = self._get_cameras(sequence_path, split)

        if end_idx == -1:
            end_idx = len(cam_dict)
        cam_dict = {k: v for i, (k, v) in enumerate(cam_dict.items()) if start_idx <= i < end_idx}

        gt_paths, target_pred, target_gt, image_whs, target_cameras = [], [], [], [], []
        for im, cam in cam_dict.items():
            rend_path, gt_path = os.path.join(sequence_path, split, "renders", im), os.path.join(
                sequence_path, split, "images", im
            )
            image, cam = self._preprocess_image(
                load_image(gt_path) if load_gt else None, Cameras.from_dict(cam).unsqueeze()
            )
            render, _ = self._preprocess_image(load_image(rend_path), None)

            rend_shape = (render.shape[-1], render.shape[-2])
            image_whs.append(rend_shape)
            assert (
                max(render.shape[-2:]) == self.args.resolution
            ), f"wrong shape {render.shape} in processing {rend_path}"
            if image is not None:
                assert rend_shape == (image.shape[-1], image.shape[-2]), f"shape mismatch {rend_shape} vs {image.shape}"

            target_gt.append(image)
            target_pred.append(render)
            gt_paths.append(gt_path)
            target_cameras.append(cam)

        target_gt = torch.stack(target_gt) if load_gt else None
        conditioning_inputs = torch.stack(target_pred)

        # get and load reference images (if any)
        ref_images, ref_cameras = self._get_closest_train_views(
            sequence_path, target_cameras, self.args.num_max_ref_images
        )
        ref_info = [self._preprocess_image(im, cam) for im, cam in zip(ref_images, ref_cameras)]
        ref_images, ref_cameras = torch.stack([r[0] for r in ref_info]), [r[1] for r in ref_info]
        del ref_info

        conditioning_inputs = torch.cat([conditioning_inputs, ref_images])
        target_gt = torch.cat([target_gt, torch.zeros_like(ref_images)]) if load_gt else None

        rays = None
        if self.args.load_rays:
            rays = self._load_rays([c.clone() for c in target_cameras], [c.clone() for c in ref_cameras])

        mask = torch.cat([torch.ones((len(gt_paths))), torch.zeros((ref_images.shape[0]))]).bool()

        data = {
            "pixel_values": target_gt,
            "conditioning_pixel_values": conditioning_inputs,
            "conditioning_ray_values": rays,
            "camera": target_cameras,
            "ref_camera": ref_cameras,
            "is_target": mask,
            "image_path": gt_paths,
            "image_wh": image_whs,
        }
        return data

    def __len__(self):
        if self.training:
            return len(self.samples)
        else:
            return len(self.test_samples)

    def __getitem__(self, idx: int):
        if not self.training and not self.args.sample_based_inference:
            return self._test_get_item(idx)

        sequence_path, split, filename = self.samples[idx] if self.training else self.test_samples[idx]

        # get cameras in other/test set
        splits = ["other", "test"] if self.training else ["test"]
        im_paths, target_cameras = [], []
        for s in splits:
            for k, v in self._get_cameras(sequence_path, s).items():
                im_paths.append(os.path.join(sequence_path, s, "images", k))
                target_cameras.append(Cameras.from_dict(v).unsqueeze())

        target_idx = im_paths.index(os.path.join(sequence_path, split, "images", filename))
        camera = target_cameras.pop(target_idx)
        im_path = im_paths.pop(target_idx)

        # select number of target and reference images
        num_tgt_images = self.num_tgt_images
        num_ref_images = self.num_ref_images
        tgt_ref_sum = self.num_tgt_images + self.num_ref_images
        if self.args.num_total_images > tgt_ref_sum:
            add_images = self.args.num_total_images - tgt_ref_sum
            rand_vec = np.random.randint(0, 2, size=add_images)
            add_refs = np.sum(rand_vec == 0)
            add_tgts = np.sum(rand_vec == 1)
            num_tgt_images += add_tgts
            num_ref_images += add_refs
            assert num_ref_images + num_tgt_images == self.args.num_total_images

        # get target images
        if num_tgt_images > 1 and len(target_cameras) > 0:
            # train: get candidate close views, randomly select k
            if self.training:
                # the more reference views we have, the farther apart target views can be
                num_candidates = max(
                    self.args.min_candidates,
                    int(len(target_cameras) * self.args.candidate_views_fraction * num_ref_images),
                )
                num_candidates = min(num_candidates, len(target_cameras))
                closest_ids = get_closest_cameras(camera, target_cameras, num_candidates)
                closest_ids = random.sample(closest_ids, k=min(len(closest_ids), num_tgt_images - 1))
            else:
                closest_ids = get_closest_cameras(camera, target_cameras, min(len(target_cameras), num_tgt_images - 1))
        else:
            closest_ids = []

        target_cameras = [camera] + [target_cameras[i] for i in closest_ids]
        target_images = [im_path] + [im_paths[i] for i in closest_ids]

        # get and load reference images (if any)
        ref_images, ref_cameras = None, None
        if num_ref_images > 0:
            ref_images, ref_cameras = self._get_closest_train_views(sequence_path, target_cameras, num_ref_images)
            ref_info = [self._preprocess_image(im, cam) for im, cam in zip(ref_images, ref_cameras)]
            ref_images, ref_cameras = torch.stack([r[0] for r in ref_info]), [r[1] for r in ref_info]
            del ref_info

        # load target GT/renderings
        target_pred, target_gt, image_whs = [], [], []
        for i, (gt_path, cam) in enumerate(zip(target_images, target_cameras)):
            split_path = os.path.dirname(os.path.dirname(gt_path))
            rend_path = os.path.join(split_path, "renders", os.path.basename(gt_path))
            image, target_cameras[i] = self._preprocess_image(load_image(gt_path), cam)
            image_whs.append((image.shape[-1], image.shape[-2]))
            assert max(image.shape[-2:]) == self.args.resolution, f"wrong shape {image.shape} in processing {gt_path}"
            render, _ = self._preprocess_image(load_image(rend_path), None)
            target_gt.append(image)
            target_pred.append(render)

        # stack target / reference images
        target_gt = torch.stack(target_gt)
        conditioning_inputs = torch.stack(target_pred)
        if ref_images is not None:
            conditioning_inputs = torch.cat([conditioning_inputs, ref_images])
            target_gt = torch.cat([target_gt, torch.zeros_like(ref_images)])

        # load rays
        rays = None
        if self.args.load_rays:
            assert (
                num_ref_images > 1 or not self.scale_poses
            ), "ray conditioning is scale ambiguous if only one ref view is given."
            rays = self._load_rays([c.clone() for c in target_cameras], [c.clone() for c in ref_cameras])

        if self.args.gt_overfit:
            assert num_ref_images == 0
            conditioning_inputs = target_gt.clone()

        mask = torch.cat([torch.ones((len(target_images))), torch.zeros((ref_images.shape[0]))]).bool()

        # randomize view order
        if self.training:
            random_permutation = torch.randperm(len(conditioning_inputs))
            tgt_order = [i.item() for i in random_permutation if mask[i]]
            ref_order = [i.item() - len(target_images) for i in random_permutation if not mask[i]]

            target_gt = target_gt[random_permutation]
            conditioning_inputs = conditioning_inputs[random_permutation]
            mask = mask[random_permutation]
            if rays is not None:
                rays = rays[random_permutation]

            # adjust info order accordingly
            target_images = [target_images[i] for i in tgt_order]
            image_whs = [image_whs[i] for i in tgt_order]
            target_cameras = [target_cameras[i] for i in tgt_order]
            ref_cameras = [ref_cameras[i] for i in ref_order] if ref_cameras is not None else None

        data = {
            "pixel_values": target_gt,
            "conditioning_pixel_values": conditioning_inputs,
            "conditioning_ray_values": rays,
            "camera": target_cameras,
            "ref_camera": ref_cameras,
            "is_target": mask,
            "image_path": target_images,
            "image_wh": image_whs,
        }
        return data


def get_closest_cameras(
    reference_cameras: Cameras | list[Cameras], other_cameras: list[Cameras], num_views: int = 1
) -> List[int]:
    other_cameras = Cameras.cat([c.unsqueeze() for c in other_cameras])
    if isinstance(reference_cameras, list):
        reference_cameras = Cameras.cat([c.unsqueeze() for c in reference_cameras])
    return select_distance(reference_cameras, other_cameras, num_views)


def collate_fn(examples):
    if examples[0]["pixel_values"] is not None:
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        pixel_values = None

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    if examples[0]["conditioning_ray_values"] is not None:
        conditioning_ray_values = torch.stack([example["conditioning_ray_values"] for example in examples])
        conditioning_ray_values = conditioning_ray_values.to(memory_format=torch.contiguous_format).float()
    else:
        conditioning_ray_values = None

    mask = torch.stack([example["is_target"] for example in examples])
    mask = mask.to(memory_format=torch.contiguous_format).bool()

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "conditioning_ray_values": conditioning_ray_values,
        "is_target": mask,
        "ref_camera": [example["ref_camera"] for example in examples],
        "camera": [example["camera"] for example in examples],
        "image_path": [example["image_path"] for example in examples],
        "image_wh": [example["image_wh"] for example in examples],
    }
