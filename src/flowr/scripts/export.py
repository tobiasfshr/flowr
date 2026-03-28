"""Export 3DGS to PLY or compressed PNG format."""


from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Dict, Literal, Union

import numpy as np
import torch
import tyro
from gsplat import PngCompression
from nerfstudio.scripts.exporter import Exporter
from nerfstudio.utils.rich_utils import CONSOLE

from flowr.model.splatfacto import SplatfactoModel
from flowr.util.eval import eval_setup


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    output_filename: str = "splat.ply"
    """Name of the output file."""
    ply_color_mode: Literal["sh_coeffs", "rgb"] = "sh_coeffs"
    """If "rgb", export colors as red/green/blue fields. Otherwise, export colors as
    spherical harmonics coefficients."""

    @staticmethod
    def write_file(
        filename: str,
        count: int,
        map_to_tensors: OrderedDict[str, np.ndarray],
    ):
        """
        Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
        Note: All float values will be converted to float32 for writing.

        Parameters:
        filename (str): The name of the file to write.
        count (int): The number of vertices to write.
        map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
            Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
        """

        # Ensure count matches the length of all tensors
        if not all(tensor.size == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        # Type check for numpy arrays of type float or uint8 and non-empty
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

        with open(filename, "wb") as ply_file:
            # Write PLY header
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")
            ply_file.write(b"comment Vertical Axis: z\n")
            ply_file.write(f"element vertex {count}\n".encode())

            # Write properties, in order due to OrderedDict
            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            # Write binary data
            # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config, test_mode="inference")

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model

        filename = self.output_dir / self.output_filename

        map_to_tensors = OrderedDict()

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            count = positions.shape[0]
            n = count
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

            if self.ply_color_mode == "rgb":
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                colors = (colors * 255).astype(np.uint8)
                map_to_tensors["red"] = colors[:, 0]
                map_to_tensors["green"] = colors[:, 1]
                map_to_tensors["blue"] = colors[:, 2]
            elif self.ply_color_mode == "sh_coeffs":
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

            if model.config.sh_degree > 0:
                if self.ply_color_mode == "rgb":
                    CONSOLE.print(
                        "Warning: model has higher level of spherical harmonics, ignoring them and only export rgb."
                    )
                elif self.ply_color_mode == "sh_coeffs":
                    # transpose(1, 2) was needed to match the sh order in Inria version
                    shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                    shs_rest = shs_rest.reshape((n, -1))
                    for i in range(shs_rest.shape[-1]):
                        map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")
        nan_count = np.sum(select) - n

        # filter gaussians that have opacities < 1/255, because they are skipped in cuda rasterization
        low_opacity_gaussians = (map_to_tensors["opacity"]).squeeze(axis=-1) < -5.5373  # logit(1/255)
        lowopa_count = np.sum(low_opacity_gaussians)
        select[low_opacity_gaussians] = 0

        if np.sum(select) < n:
            CONSOLE.print(
                f"{nan_count} Gaussians have NaN/Inf and {lowopa_count} have low opacity, only export {np.sum(select)}/{n}"
            )
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)

        ExportGaussianSplat.write_file(str(filename), count, map_to_tensors)


@dataclass
class ExportGaussianSplatCompressed(Exporter):
    @staticmethod
    def write_file(
        export_dir: Path,
        splats: Dict[str, torch.Tensor],
    ):
        """Write splats to compressed PNG files.

        Args:
            export_dir (str): The directory to save the compressed PNG files.
            splats (Dict[str, torch.Tensor]): A dictionary mapping property names to tensors.
        """
        compression_method = PngCompression()
        compression_method.compress(str(export_dir), splats)

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config, test_mode="inference")

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model

        assert not model.gauss_params.means.device == torch.device(
            "cpu"
        ), "Model is on CPU, but torchpq KMeans w/ manhattan distance requires GPU"
        with torch.no_grad():
            splats = {
                "means": model.gauss_params.means,
                "scales": model.gauss_params.scales.data,
                "quats": model.gauss_params.quats.data,
                "opacities": model.gauss_params.opacities.data.squeeze(-1),
                "sh0": model.features_dc.data,
            }
            if model.config.sh_degree > 0:
                splats["shN"] = model.features_rest.data

        ExportGaussianSplatCompressed.write_file(self.output_dir, splats)


@dataclass
class ExportGaussianSplatDecompress:
    """Decompresses a PNG-compressed 3DGS to a PLY file."""

    input_dir: Path
    output_filename: str = "splat.ply"

    def main(self) -> None:
        compression_method = PngCompression()
        result_dict = compression_method.decompress(str(self.input_dir))
        count = result_dict["means"].shape[0]

        map_to_tensors = OrderedDict()
        map_to_tensors["x"] = result_dict["means"][:, 0].cpu().numpy()
        map_to_tensors["y"] = result_dict["means"][:, 1].cpu().numpy()
        map_to_tensors["z"] = result_dict["means"][:, 2].cpu().numpy()

        map_to_tensors["nx"] = np.zeros(count, dtype=np.float32)
        map_to_tensors["ny"] = np.zeros(count, dtype=np.float32)
        map_to_tensors["nz"] = np.zeros(count, dtype=np.float32)

        scales = result_dict["scales"].cpu().numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None]

        quats = result_dict["quats"].cpu().numpy()
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        map_to_tensors["opacity"] = result_dict["opacities"].unsqueeze(-1).cpu().numpy()

        # SH coeffs
        shs_0 = result_dict["sh0"].cpu().numpy()
        for i in range(shs_0.shape[1]):
            map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

        if "shN" in result_dict:
            shs_rest = result_dict["shN"].transpose(1, 2).contiguous().cpu().numpy()
            shs_rest = shs_rest.reshape((count, -1))
            for i in range(shs_rest.shape[-1]):
                map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

        ExportGaussianSplat.write_file(str(self.input_dir / self.output_filename), count, map_to_tensors)


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="ply")],
        Annotated[ExportGaussianSplatCompressed, tyro.conf.subcommand(name="png")],
        Annotated[ExportGaussianSplatDecompress, tyro.conf.subcommand(name="decompress")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()
