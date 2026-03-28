import argparse
import json
import os

import matplotlib.pyplot as plt


def get_statistics(input_dir: str):
    os.makedirs(os.path.join(input_dir, "plots"), exist_ok=True)
    # sequence-level metrics statistics
    seq_metrics_all = []
    for seq in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, seq)) and seq != "plots":
            with open(os.path.join(input_dir, seq, "metrics.json"), "r") as f:
                seq_metrics = json.load(f)
            seq_metrics_all.append(seq_metrics)

    keys = ["psnr", "ssim", "lpips"]
    diff_keys = ["psnr_init", "ssim_init", "lpips_init"]
    for key, diff_key in zip(keys, diff_keys):
        values = [m[key] for m in seq_metrics_all]
        values_init = [m[diff_key] for m in seq_metrics_all]
        plt.scatter(values_init, values)
        max_val = max(max(values_init), max(values))  # noqa: PLW3301
        plt.plot([0, max_val], [0, max_val], "k--", lw=1)
        plt.xlabel(diff_key)
        plt.ylabel(key)
        plt.savefig(os.path.join(input_dir, "plots", f"sequence_{key}.png"), bbox_inches="tight", dpi=300)
        plt.clf()

    # image-level-metrics statistics
    with open(os.path.join(input_dir, "metrics_list.json"), "r") as f:
        metrics_list = json.load(f)
    for key, diff_key in zip(keys, diff_keys):
        values = [m[key] for m in metrics_list]
        values_init = [m[diff_key] for m in metrics_list]
        plt.boxplot([values, values_init], labels=[key, diff_key])
        plt.savefig(os.path.join(input_dir, "plots", f"{key}.png"), bbox_inches="tight", dpi=300)
        plt.clf()
        diffs = [m[key] - m[diff_key] for m in metrics_list]
        plt.boxplot([diffs], labels=[key])
        plt.savefig(os.path.join(input_dir, "plots", f"{key}_diff.png"), bbox_inches="tight", dpi=300)
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, help="input dir")
    args = parser.parse_args()
    get_statistics(args.input_dir)
