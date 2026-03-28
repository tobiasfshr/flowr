import argparse
import os
import zipfile

from nerfstudio.utils.rich_utils import CONSOLE

from flowr.scripts.datasets.scannetpp.util import process_sequence

ALL_SUBSETS = ["train", "val"]


def generate_data(root: str, subset_or_hash: str, work_dir: str, data_dir: str, force: bool, include_other: bool = True):
    # load the scene list:
    if subset_or_hash in ALL_SUBSETS:
        # load the scene list:
        with open(f"{root}/splits/nvs_sem_{subset_or_hash}.txt", "r") as f:
            scenes = f.read().splitlines()
        subset = subset_or_hash
    else:
        with open(f"{root}/splits/nvs_sem_val.txt", "r") as f:
            scenes = f.read().splitlines()
        subset = "val" if subset_or_hash in scenes else "train"
        scenes = [subset_or_hash]

    save_compressed = subset != "val"
    # filter existing items if not force
    if not force and subset_or_hash in ALL_SUBSETS:
        scenes = filter_scenes(scenes, data_dir, subset)

    if subset_or_hash not in ALL_SUBSETS:
        for i, task in enumerate(scenes):
            CONSOLE.log(f"Processing {task}...")
            process_sequence(task, root, work_dir, data_dir, subset, save_compressed, include_other=include_other)
        return

    CONSOLE.log(f"Starting {len(scenes)} jobs...")
    for i, task in enumerate(scenes):
        submit_job(task, root, work_dir, data_dir, include_other)


def submit_job(task, root, work_dir, data_dir, include_other):
    add_opts = "" if include_other else " --skip_other"
    cmd = (
        f"python -m flowr.scripts.datasets.scannetpp.generate_data generate {root} {work_dir} {data_dir} --hash {task}{add_opts}"
    )
    submit_cmd = (
        f'python submit.py "{cmd}"  --cores_per_gpu 4 --ram_per_gpu 32000 --num_gpus 1 --scratch 16000 --time 3:59:00'
    )
    os.system(submit_cmd)


def filter_scenes(scenes, data_dir, subset, return_idcs=False):
    """Only keep scenes that do not have a valid zip file yet."""
    filtered_items, idcs = [], []
    for i, scene in enumerate(scenes):
        zip_path = os.path.join(data_dir, "data", f"{scene}.zip")
        if not os.path.exists(zip_path):
            filtered_items.append(scene)
            idcs.append(i)
            continue
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                result = zip_ref.testzip()
        except Exception:
            result = "exception"
        if result is not None:
            CONSOLE.log(f"Reconstruction Zipfile for {scene} in {subset} is invalid.")
            filtered_items.append(scene)
            idcs.append(i)

    if return_idcs:
        return filtered_items, idcs
    return filtered_items


def check_data(root: str, data_dir: str, subset=None):
    if subset is None:
        total_items, success_items = [0 for _ in range(len(ALL_SUBSETS))], [0 for _ in range(len(ALL_SUBSETS))]
        for i, subset in enumerate(ALL_SUBSETS):
            with open(f"{root}/splits/nvs_sem_{subset}.txt", "r") as f:
                scenes = f.read().splitlines()
            filtered_scenes = filter_scenes(scenes, data_dir, subset)
            success_items[i], total_items[i] = len(scenes) - len(filtered_scenes), len(scenes)
        for i, subset in enumerate(ALL_SUBSETS):
            CONSOLE.log(f"Summary for {subset}: {success_items[i]}/{total_items[i]} scale factors found and valid.")
        success_items, total_items = sum(success_items), sum(total_items)
        CONSOLE.log(f"Total: {success_items}/{total_items} scale factors found and valid.")
    else:
        with open(f"{root}/splits/nvs_sem_{subset}.txt", "r") as f:
            scenes = f.read().splitlines()
        filtered_scenes = filter_scenes(scenes, data_dir, subset)
        success_items, total_items = len(scenes) - len(filtered_scenes), len(scenes)
        CONSOLE.log(f"Summary for {subset}: {success_items}/{total_items} scale factors found and valid.")
    return success_items == total_items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Do Stage 1 reconstruction, generate image pairs, create Zip at target location."
    )
    subparsers = parser.add_subparsers(dest="action", required=True, help="Action to perform: generate or check")

    # Subcommand: compute
    compute_parser = subparsers.add_parser("generate", help="Generate data")
    compute_parser.add_argument("scannet_root", type=str, help="Path to scannet dataset root.")
    compute_parser.add_argument(
        "work_dir", type=str, help="Path to the working directory. Will save reconstruction and renderings temporarily."
    )
    compute_parser.add_argument("data_dir", type=str, help="Path to the final data directory.")
    compute_parser.add_argument("--hash", type=str, default=None, help="Specific hash to process.")
    compute_parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val"], help="Split to process (uses file)."
    )
    compute_parser.add_argument("--force", action="store_true", help="Force compute even if the file already exists.")
    compute_parser.add_argument(
        "--skip_other",
        action="store_true",
        help="Skip preparing and rendering the optional other split. Useful for evaluation subsets such as ScanNet++ val.",
    )

    # Subcommand: check
    check_parser = subparsers.add_parser("check", help="Check data")
    check_parser.add_argument("scannet_root", type=str, help="Path to scannet dataset root.")
    check_parser.add_argument("data_dir", type=str, help="Path to the final data directory.")
    check_parser.add_argument(
        "--split", type=str, default=None, choices=["train", "val"], help="Split to check (uses file)."
    )

    args = parser.parse_args()
    if args.action == "generate":
        generate_data(
            args.scannet_root,
            args.hash if args.hash is not None else args.split,
            args.work_dir,
            args.data_dir,
            args.force,
            include_other=not args.skip_other,
        )
    elif args.action == "check":
        assert check_data(args.scannet_root, args.data_dir, args.split), "Some reconstructions are missing or invalid."
