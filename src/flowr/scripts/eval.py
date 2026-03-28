"""Evaluation entry point."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import tyro
from nerfstudio.utils.rich_utils import CONSOLE

from flowr.util.eval import eval_setup


@dataclass
class EvaluateModel:
    """Load a model, compute metrics, and save it to a JSON file."""

    load_config: Path
    """Path to config YAML file."""
    output_path: Path = Path("output.json")
    """ Name of the output file."""
    render_output_path: Optional[Path] = None
    """Optional path to save rendered outputs to."""
    split: Literal["train", "test"] = "test"
    """Which split should be evaluated."""

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        if self.split == "train":
            # Evaluate on training set when explicitly requested
            pipeline.datamanager.eval_dataset = pipeline.datamanager.train_dataset

        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = pipeline.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(EvaluateModel).main()


if __name__ == "__main__":
    entrypoint()
