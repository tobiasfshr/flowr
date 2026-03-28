import os
import pickle
import shutil
import sys
import traceback
import urllib.request
import zipfile

import pandas as pd
from huggingface_hub import HfApi
from tqdm import tqdm

CLEAN_CACHE = False
ALL_SUBSETS = ["1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K", "11K"]


# load from assets/dl3dv140.txt (each line contains one hash)
with open("assets/dl3dv140.txt", "r") as f:
    TEST_HASHES = f.read().splitlines()
    # only keep after the first comma and before the second
    TEST_HASHES = [h.split(",")[1] for h in TEST_HASHES]

api = HfApi()
resolution2repo = {
    "480P": "DL3DV/DL3DV-ALL-480P",
    "960P": "DL3DV/DL3DV-ALL-960P",
    "2K": "DL3DV/DL3DV-ALL-2K",
    "4K": "DL3DV/DL3DV-ALL-4K",
}


def hf_download_path(repo: str, rel_path: str, odir: str, max_try: int = 5):
    """hf api is not reliable, retry when failed with max tries

    :param repo: The huggingface dataset repo
    :param rel_path: The relative path in the repo
    :param odir: output path
    :param max_try: As the downloading is not a reliable process, we will retry for max_try times
    """
    counter = 0
    while True:
        if counter >= max_try:
            tqdm.write(f"ERROR: Download {repo}/{rel_path} failed.")
            return False
        try:
            api.hf_hub_download(
                repo_id=repo,
                filename=rel_path,
                repo_type="dataset",
                local_dir=odir,
                cache_dir=os.path.join(odir, ".cache"),
            )
            return True

        except KeyboardInterrupt:
            tqdm.write("Keyboard Interrupt. Exit.")
            sys.exit()
        except BaseException:
            traceback.print_exc()
            counter += 1


def download_from_url(url: str, ofile: str):
    """Download a file from the url to ofile

    :param url: The url link
    :param ofile: The output path
    :return: True if download success, False otherwise
    """
    try:
        # Use urllib.request.urlretrieve to download the file from `url` and save it locally at `local_file_path`
        urllib.request.urlretrieve(url, ofile)
        return True
    except Exception as e:
        tqdm.write(f"An error occurred while downloading the file: {e}")
        return False


def clean_huggingface_cache(output_dir: str, repo: str):
    """Huggingface cache may take too much space, we clean the cache to save space if necessary

        Current huggingface hub does not provide good practice to clean the space.
        We manually clean the cache directory if necessary.

    :param output_dir: the current output directory
    :param output_dir: the huggingface repo
    """
    # cur_cache_dir = join(output_dir, '.cache', f'datasets--{repo_cache_dir}')
    cur_cache_dir = os.path.join(output_dir, ".cache")

    if os.path.exists(cur_cache_dir):
        shutil.rmtree(cur_cache_dir)


def get_download_list(
    subset_opt: str, hash_name: str, reso_opt: str, file_type: str, output_dir: str, use_benchmark_data: bool = False
):
    """Get the download list based on the subset and hash name

        1. Get the meta file
        2. Select the subset. Based on reso_opt, get the downloading list prepared.
        3. Return the download list.

    :param subset_opt: Subset of the 10K, e.g. 1K(0~1K), 2K(1K~2K), 3K(2K~3K), etc
    :param hash_name: If provided a non-empty string, ignore the subset_opt and only download the specific hash
    :param reso_opt: The resolution to download.
    :param file_type: The file type to download: video | images+poses | colmap_cache
    :param output_dir: The output directory.
    """

    def to_download_item(hash_name, reso, batch, file_type):
        if file_type == "images+poses":
            repo = resolution2repo[reso]
            rel_path = f"{batch}/{hash_name}.zip"
        elif file_type == "video":
            repo = "DL3DV/DL3DV-ALL-video"
            rel_path = f"{batch}/{hash_name}/video.mp4"
        elif file_type == "colmap_cache":
            repo = "DL3DV/DL3DV-ALL-ColmapCache"
            rel_path = f"{batch}/{hash_name}.zip"

        # return f'{repo}/{batch}/{hash_name}'
        return {"repo": repo, "rel_path": rel_path, "name": hash_name}

    ret = []
    meta_link = "https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv"
    cache_folder = os.path.join(output_dir, ".cache")
    meta_file = os.path.join(cache_folder, "DL3DV-valid.csv")
    os.makedirs(cache_folder, exist_ok=True)
    if not os.path.exists(meta_file):
        assert download_from_url(meta_link, meta_file), "Download meta file failed."

    df = pd.read_csv(meta_file)

    # if hash is set, ignore the subset_opt
    if hash_name != "":
        assert (
            hash_name in df["hash"].values or hash_name in TEST_HASHES
        ), f"Hash {hash_name} not found in the meta file."

        if hash_name not in df["hash"].values or (use_benchmark_data and hash_name in TEST_HASHES):
            assert hf_download_path("DL3DV/DL3DV-10K-Benchmark", "benchmark-meta.csv", cache_folder)
            assert hf_download_path("DL3DV/DL3DV-10K-Benchmark", ".cache/filelist.bin", cache_folder)
            df = pd.read_csv(os.path.join(cache_folder, "benchmark-meta.csv"))
            if reso_opt == "480P":
                reso_num = "_8"
            elif reso_opt == "960P":
                reso_num = "_4"
            elif reso_opt == "2K":
                reso_num = "_2"
            else:
                reso_num = ""
            filelist = []
            for f in pickle.load(open(os.path.join(cache_folder, ".cache/filelist.bin"), "rb"))[hash_name]:
                if "nerfstudio" not in f:
                    continue
                if "images" in os.path.dirname(f) and f"images{reso_num}" not in f:
                    continue
                filelist.append(f)
            link = {"repo": "DL3DV/DL3DV-10K-Benchmark", "rel_path": hash_name, "files": filelist, "name": hash_name}
            return [link]
        else:
            batch = df[df["hash"] == hash_name]["batch"].values[0]
            link = to_download_item(hash_name, reso_opt, batch, file_type)
            ret = [link]
            return ret

    # if hash not set, we download the whole subset
    if subset_opt != "140":
        subdf = df[df["batch"] == subset_opt]
        for i, r in subdf.iterrows():
            hash_name = r["hash"]
            if hash_name in TEST_HASHES:
                continue
            ret.append(to_download_item(hash_name, reso_opt, subset_opt, file_type))
    else:
        for hash_name in TEST_HASHES:
            ret.extend(get_download_list("", hash_name, reso_opt, file_type, output_dir, use_benchmark_data))

    return ret


def download(download_list: list, output_dir: str, is_clean_cache: bool):
    """Download the dataset based on the download_list and user options.

    :param download_list: the list of files to download, [{'repo', 'rel_path'}]
    :param output_dir: the output directory
    :param reso_opt: the resolution option
    :param is_clean_cache: if set, will clean the huggingface cache to save space
    """
    succ_count = 0

    for item in tqdm(download_list, desc="Downloading"):
        repo = item["repo"]
        rel_path = item["rel_path"]

        output_path = os.path.join(output_dir, rel_path)
        output_path = output_path.replace(".zip", "")
        # skip if already exists locally
        if os.path.exists(output_path):
            succ_count += 1
            continue

        if "files" in item:
            succs = []
            for f in item["files"]:
                succ = hf_download_path(repo, f, output_dir)
                succs.append(succ)
            succ = all(succs)
        else:
            succ = hf_download_path(repo, rel_path, output_dir)

        if succ:
            succ_count += 1
            if is_clean_cache:
                clean_huggingface_cache(output_dir, repo)

            # unzip the file
            if rel_path.endswith(".zip"):
                zip_file = os.path.join(output_dir, rel_path)
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    ofile = os.path.join(output_dir, os.path.dirname(rel_path))
                    # check if the zip file contains a single parent folder
                    if not all(name.startswith(item["name"]) for name in zip_ref.namelist()):
                        ofile = os.path.join(output_dir, rel_path.replace(".zip", ""))
                    zip_ref.extractall(ofile)
                os.remove(zip_file)
        else:
            tqdm.write(f"Download {rel_path} failed")

    tqdm.write(f"Summary: {succ_count}/{len(download_list)} files downloaded successfully")
    return succ_count == len(download_list)
