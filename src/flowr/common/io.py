import json
import os
import zipfile
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from torch import Tensor

ZIP_CACHE = OrderedDict()
NAMELIST_CACHE = {}
MAX_CACHE_SIZE = 128


def save_image(img: Union[np.ndarray, Tensor, Image.Image], path: str):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if not isinstance(img, Image.Image):
        if isinstance(img, Tensor):
            assert len(img.shape) == 3, "Too many input channels!"
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = img.detach().cpu().numpy()
        elif not isinstance(img, np.ndarray):
            raise ValueError(f"Invalid type for argument img: {type(img)}")
        if img.dtype in [np.float16, np.float32, np.float64]:
            img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
    with open(path, "wb") as fid:
        img.save(fid)


def open_zipfile(path: str | Path) -> zipfile.ZipFile:
    path = str(path)
    if path in ZIP_CACHE:
        ZIP_CACHE.move_to_end(path)
        return ZIP_CACHE[path]

    if len(ZIP_CACHE) >= MAX_CACHE_SIZE:
        zip_key, oldest_zip = ZIP_CACHE.popitem(last=False)
        NAMELIST_CACHE.pop(zip_key, None)
        oldest_zip.close()

    ZIP_CACHE[path] = zipfile.ZipFile(path, "r")
    return ZIP_CACHE[path]


def close_zipfile(path: str | Path):
    path = str(path)
    if path in ZIP_CACHE:
        ZIP_CACHE[path].close()
        del ZIP_CACHE[path]


def zip_exists(path: str | Path, close: bool = True) -> bool:
    """
    Check whether a file or directory exists inside a zip archive,
    or if a zip file exists on disk.

    The path may be provided in one of two forms:
      - "archive.zip"              -> checks if the zip file exists on disk.
      - "archive.zip/some/path"    -> checks if "some/path" exists inside archive.zip.

    For an internal zip path, existence means:
      - There is an entry exactly matching the given path, or
      - For a directory, either the archive contains an explicit directory entry
        (with a trailing slash) or some file exists whose path begins with that directory.
    """
    path = str(path)

    # If the path does not reference a zip archive at all, delegate to os.path.exists.
    if ".zip" not in path:
        return os.path.exists(path)

    # If the path exactly ends with ".zip", check that file on disk.
    if path.endswith(".zip"):
        return os.path.exists(path)

    # Otherwise, assume the path is for an entry inside a zip.
    # We look for the first occurrence of ".zip/" as the separator.
    sep = ".zip/"
    zip_index = path.find(sep)
    if zip_index == -1:
        raise ValueError("Internal path must be specified as 'archive.zip/some/path'")

    # Extract the zip file path and the internal relative path.
    zip_path = path[: zip_index + len(".zip")]
    rel_path = path[zip_index + len(sep) :]

    # Open the zip file and get its list of names.
    try:
        z = open_zipfile(zip_path)
        # An empty relative path corresponds to the top-level,
        # which we consider as existing if the zip is openable.
        if not rel_path:
            return True

        if zip_path not in NAMELIST_CACHE:
            names = z.namelist()
            NAMELIST_CACHE[zip_path] = names
        else:
            names = NAMELIST_CACHE[zip_path]

        if close:
            close_zipfile(zip_path)

        # Direct file or explicit directory match.
        if rel_path in names:
            return True

        # Check for directory existence. Normalize to ensure a trailing slash.
        prefix = rel_path.rstrip("/") + "/"
        for name in names:
            if name.startswith(prefix):
                return True
    except FileNotFoundError:
        # The zip file itself does not exist.
        return False

    return False


def zip_valid(path: str | Path, close: bool = True) -> bool:
    """Return True if the file at `path` is a valid ZIP archive."""
    try:
        zf = open_zipfile(path)
        # testzip() returns None if no errors are found
        if zf.testzip() is None:
            if close:
                close_zipfile(path)
            return True
    except Exception:
        pass
    return False


def zip_listdir(path: str | Path, close: bool = True) -> list[str]:
    """
    List the contents of a directory inside a zip file.

    The path should be either:
      - "archive.zip"            -> top-level listing
      - "archive.zip/some/dir"   -> listing for "some/dir" inside the zip

    Behaves like os.listdir: returns only the immediate (unique) names,
    and raises FileNotFoundError if the directory does not exist.
    """
    path = str(path)
    if ".zip/" in path:
        zip_path, rel_path = path.split(".zip/", 1)
        zip_path += ".zip"
        # Normalize the relative path so it ends with a slash
        rel_path = rel_path.rstrip("/") + "/"
    elif path.endswith(".zip"):
        zip_path = path
        rel_path = ""
    else:
        raise ValueError("Path must contain '.zip/' or end with '.zip'")

    z = open_zipfile(zip_path)

    if NAMELIST_CACHE.get(zip_path) is None:
        names = z.namelist()
        NAMELIST_CACHE[zip_path] = names
    else:
        names = NAMELIST_CACHE[zip_path]

    if close:
        close_zipfile(zip_path)

    # If we're listing a subdirectory, ensure it exists.
    if rel_path:
        if not any(name == rel_path or name.startswith(rel_path) for name in names):
            raise FileNotFoundError(f"No such directory in zip: {rel_path}")

        result = []
        for name in names:
            if name.startswith(rel_path):
                # Remove the directory prefix
                remainder = name[len(rel_path) :]
                if remainder:
                    # Get the first part (immediate child)
                    entry = remainder.split("/", 1)[0]
                    if entry not in result:
                        result.append(entry)
    else:
        # Top-level: get the first part of every name.
        result = []
        for name in names:
            entry = name.split("/", 1)[0]
            if entry and entry not in result:
                result.append(entry)
    return result


def load_bytes(path: str | Path, close: bool = True) -> bytes:
    path = str(path)
    # if zip file + relative path
    if ".zip/" in path:
        zip_path, rel_path = path.split(".zip/")
        z = open_zipfile(zip_path + ".zip")
        with z.open(rel_path) as f:
            content = f.read()
        if close:
            close_zipfile(zip_path + ".zip")
        return content
    else:
        with open(path, "rb") as f:
            return f.read()


def load_json(path: str | Path, close: bool = True) -> dict:
    path = str(path)
    # if zip file + relative path
    if ".zip/" in path:
        zip_path, rel_path = path.split(".zip/")
        z = open_zipfile(zip_path + ".zip")
        with z.open(rel_path) as f:
            content = json.load(f)
        if close:
            close_zipfile(zip_path + ".zip")
    else:
        with open(path, "r") as f:
            content = json.load(f)
    return content


def load_image(path: str | Path, close: bool = True) -> Image.Image:
    path = str(path)
    # if zip file + relative path
    if ".zip/" in path:
        zip_path, rel_path = path.split(".zip/")
        z = open_zipfile(zip_path + ".zip")
        with z.open(rel_path) as f:
            image = Image.open(BytesIO(f.read())).convert("RGB")
        if close:
            close_zipfile(zip_path + ".zip")
        return image
    return Image.open(path)


def get_image_mask_from_path(filepath: Path, scale_factor: float = 1.0) -> np.ndarray:
    """
    Utility function to read a mask image from the given path and return a boolean array
    """
    pil_mask = load_image(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.Resampling.NEAREST)
    mask_array = np.array(pil_mask)[:, :, np.newaxis].astype(bool)
    if len(mask_array.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return mask_array


def get_depth_image_from_path(
    filepath: Path | str,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """Loads, rescales and resizes depth images.

    Filepath points to a 16-bit or 32-bit depth image, a numpy array `*.npy` or a `*.ptz` file.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image numpy array (float32) with shape [height, width, 1].
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    file_bytes = load_bytes(filepath)
    if filepath.suffix == ".npy":
        image = np.load(BytesIO(file_bytes))
    else:
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_ANYDEPTH)

    image = image.astype(np.float64) * scale_factor
    image = cv2.resize(image, (width, height), interpolation=interpolation)
    return image[:, :, np.newaxis].astype(np.float32)


def from_ply(filename: str | Path, close: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Load point cloud from PLY file.

    Args:
        filename (str | Path): Path to the PLY file.
        close (bool): Close the zip file after reading.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Point cloud and color array.
    """
    # Load the PLY file
    filename = str(filename)
    if ".zip/" in filename:
        zip_path, rel_path = filename.split(".zip/")
        z = open_zipfile(zip_path + ".zip")
        with z.open(rel_path) as f:
            ply_data = PlyData.read(BytesIO(f.read()))
        if close:
            close_zipfile(zip_path + ".zip")
    else:
        ply_data = PlyData.read(filename)

    # Extract X, Y, and Z coordinates into a NumPy array
    x_coordinates = ply_data["vertex"]["x"]
    y_coordinates = ply_data["vertex"]["y"]
    z_coordinates = ply_data["vertex"]["z"]

    # Extract R, G, and B values into a NumPy array
    if "red" in ply_data["vertex"]:
        assert (
            "green" in ply_data["vertex"] and "blue" in ply_data["vertex"]
        ), "Color channels must be present together!"
        red = ply_data["vertex"]["red"]
        green = ply_data["vertex"]["green"]
        blue = ply_data["vertex"]["blue"]
        color_array = np.column_stack((red, green, blue))
    else:
        color_array = None

    # Combine X, Y, and Z coordinates, colors into a Nx3 NumPy array
    point_cloud_array = np.column_stack((x_coordinates, y_coordinates, z_coordinates))
    return point_cloud_array, color_array


def to_ply(
    filename: str,
    points: Tensor | np.ndarray,
    colors: Tensor | np.ndarray | None = None,
):
    """N,3 points / colors to ply file."""
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if colors is not None and not isinstance(colors, np.ndarray):
        colors = colors.cpu().numpy()

    vertexs = np.array(
        [tuple(v) for v in points],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    if colors is not None:
        vertex_colors = np.array(
            [tuple(v) for v in colors.astype(np.uint8)],
            dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
        for prop in vertexs.dtype.names:
            vertex_all[prop] = vertexs[prop]
        for prop in vertex_colors.dtype.names:
            vertex_all[prop] = vertex_colors[prop]
        vertexs = vertex_all

    data = [PlyElement.describe(vertexs, "vertex")]
    PlyData(data).write(filename)
