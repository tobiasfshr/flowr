import argparse
import os
import zipfile

from nerfstudio.utils.rich_utils import CONSOLE

from flowr.scripts.datasets.dl3dv.hf_util import ALL_SUBSETS, TEST_HASHES, get_download_list
from flowr.scripts.datasets.dl3dv.util import process_sequence


def generate_data(
    subset_or_hash: str,
    work_dir: str,
    data_dir: str,
    scale_dir: str,
    view_fraction,
    force: bool,
    resolution: str = "960P",
    use_benchmark_data: bool = False,
    include_other: bool = True,
):
    if subset_or_hash in ALL_SUBSETS + ["140"]:
        subset = subset_or_hash
        hash_name = ""
    else:
        subset = ""
        hash_name = subset_or_hash

    # get all hashes
    download_items = get_download_list(
        subset, hash_name, resolution, "images+poses", os.path.join(work_dir, "download"), use_benchmark_data
    )

    if subset == "":
        for i, task in enumerate(download_items):
            CONSOLE.log(f"Processing {task['name']}...")
            process_sequence(task, work_dir, data_dir, scale_dir, view_fraction, include_other=include_other)
        return

    # filter existing items if not force
    if not force and subset != "":
        download_items = filter_download_items(
            download_items, os.path.join(data_dir, "data"), subset, view_frac=view_fraction
        )

    # start jobs
    CONSOLE.log(f"Starting {len(download_items)} jobs...")
    for i, task in enumerate(download_items):
        submit_job(task, work_dir, data_dir, scale_dir, view_fraction, use_benchmark_data, include_other)


def submit_job(task, work_dir, data_dir, scale_dir, view_fraction, use_benchmark_data, include_other):
    view_fraction = [view_fraction] if isinstance(view_fraction, (int, float)) else view_fraction
    view_frac_str = " ".join(map(str, view_fraction))
    add_opts = "--use_benchmark" if use_benchmark_data else ""
    if not include_other:
        add_opts = f"{add_opts} --skip_other".strip()
    cmd = f"python -m flowr.scripts.datasets.dl3dv.generate_data generate {work_dir} {data_dir} {scale_dir} --hash {task['name']} --views {view_frac_str} {add_opts}"
    submit_cmd = (
        f'python submit.py "{cmd}"  --cores_per_gpu 4 --ram_per_gpu 32000 --num_gpus 1 --scratch 16000 --time 3:59:00'
    )
    os.system(submit_cmd)


def filter_download_items(download_items, data_dir, subset, return_idcs=False, view_frac=None):
    """Only keep download items that do not have a valid zip file yet."""
    filtered_items, idcs = [], []
    for i, item in enumerate(download_items):
        if item["name"] in TEST_HASHES:
            assert view_frac is not None and isinstance(view_frac, int), "view_frac must be provided for test sequences"
            relative_path = f"140/{view_frac}views/{item['name']}.zip"
        else:
            relative_path = item["rel_path"]
        zip_path = os.path.join(data_dir, relative_path)
        if not os.path.exists(zip_path):
            filtered_items.append(item)
            idcs.append(i)
            continue
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                result = zip_ref.testzip()
        except Exception:
            result = "exception"
        if result is not None:
            CONSOLE.log(f"Reconstruction Zipfile for {item['name']} in {subset} is invalid.")
            filtered_items.append(item)
            idcs.append(i)
        CONSOLE.log(f"Data for {item['name']} in {subset} exists. Skipping.")

    if return_idcs:
        return filtered_items, idcs
    return filtered_items


def check_data(data_dir: str, subset=None, view_frac=None):
    if subset is None:
        total_items, success_items = [0 for _ in range(len(ALL_SUBSETS))], [0 for _ in range(len(ALL_SUBSETS))]
        for i, subset in enumerate(ALL_SUBSETS):
            download_items = get_download_list(subset, "", "960P", "images+poses", "")
            filtered_items = filter_download_items(download_items, data_dir, subset, view_frac=view_frac)
            success_items[i], total_items[i] = len(download_items) - len(filtered_items), len(download_items)
        for i, subset in enumerate(ALL_SUBSETS):
            CONSOLE.log(f"Summary for {subset}: {success_items[i]}/{total_items[i]} scale factors found and valid.")
        success_items, total_items = sum(success_items), sum(total_items)
        CONSOLE.log(f"Total: {success_items}/{total_items} scale factors found and valid.")
    else:
        # test set
        if subset == "140":
            download_items = []
            for hash_name in TEST_HASHES:
                download_items.extend(get_download_list("", hash_name, "960P", "images+poses", ""))
        else:
            download_items = get_download_list(subset, "", "960P", "images+poses", "")

        filtered_items = filter_download_items(download_items, data_dir, subset, view_frac=view_frac)
        success_items, total_items = len(download_items) - len(filtered_items), len(download_items)
        CONSOLE.log(f"Summary for {subset}: {success_items}/{total_items} scale factors found and valid.")
    return success_items == total_items


def view_fraction_type(value: str) -> float | int:
    try:
        # Try to convert to int
        return int(value)
    except ValueError:
        pass

    try:
        # Try to convert to float
        return float(value)
    except ValueError:
        pass

    raise argparse.ArgumentTypeError(f"Invalid argument: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Do Stage 1 reconstruction, generate image pairs, create Zip at target location."
    )
    subparsers = parser.add_subparsers(dest="action", required=True, help="Action to perform: generate or check")

    # Subcommand: compute
    compute_parser = subparsers.add_parser("generate", help="Generate data")
    compute_parser.add_argument(
        "work_dir",
        type=str,
        help="Path to the working directory. Will save download, reconstruction, renderings temporarily.",
    )
    compute_parser.add_argument("data_dir", type=str, help="Path to the final data directory.")
    compute_parser.add_argument("scale_dir", type=str, help="Path to the scale directory.")
    compute_parser.add_argument(
        "--views",
        type=view_fraction_type,
        nargs="+",
        default=[0.05],
        help="Fraction of views to use for training. Could be a float, int, or 2-tuple of ints or floats.",
    )
    compute_parser.add_argument(
        "--hash", type=str, default=None, help="Hash of the specific reconstruction to download."
    )
    compute_parser.add_argument(
        "--subset", choices=ALL_SUBSETS + ["140"], default=None, help="The subset of the benchmark to download."
    )
    compute_parser.add_argument("--force", action="store_true", help="Force compute even if the file already exists.")
    compute_parser.add_argument(
        "--use_benchmark",
        action="store_true",
        help="Use benchmark data ('DL3DV-Benchmark') for download, which is slower but complete.",
    )
    compute_parser.add_argument(
        "--skip_other",
        action="store_true",
        help="Skip preparing and rendering the optional other split. Useful for evaluation subsets such as DL3DV-140.",
    )

    # Subcommand: check
    check_parser = subparsers.add_parser("check", help="Check data")
    check_parser.add_argument("data_dir", type=str, help="Path to the data directory.")
    check_parser.add_argument(
        "--subset",
        choices=ALL_SUBSETS + ["140"],
        default=None,
        help="The subset of the benchmark to download. If None, check all subsets.",
    )
    check_parser.add_argument(
        "--views", type=int, default=None, help="View fraction to check for test set (int only!)."
    )

    args = parser.parse_args()

    if args.action == "generate":
        if len(args.views) == 1:
            args.views = args.views[0]
        elif len(args.views) == 2:
            args.views = tuple(args.views)
        else:
            raise ValueError("Invalid arguments for view fraction.")
        assert not (args.hash is None and args.subset is None), "Please provide a hash or subset."
        generate_data(
            args.subset if args.subset is not None else args.hash,
            args.work_dir,
            args.data_dir,
            args.scale_dir,
            args.views,
            args.force,
            use_benchmark_data=args.use_benchmark,
            include_other=not args.skip_other,
        )
    elif args.action == "check":
        assert check_data(args.data_dir, args.subset, args.views), "Some reconstructions are missing or invalid."
