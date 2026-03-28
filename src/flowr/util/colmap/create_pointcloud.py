"""COLMAP utility for creating a pointcloud from posed images."""
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Literal, Optional

from nerfstudio.cameras.camera_utils import quaternion_from_matrix
from nerfstudio.process_data.colmap_utils import create_ply_from_colmap
from nerfstudio.utils.poses import to4x4
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command

from flowr.common.geometry import inverse_rigid_transform, opencv_to_opengl
from flowr.common.io import load_bytes
from flowr.common.pointcloud import sample_farthest_points
from flowr.struct.cameras import Cameras, CameraType
from flowr.util.colmap.database import COLMAPDatabase, pair_id_to_image_ids


def create_ply_from_posed_images(
    base_dir: Path,
    image_filenames: List[Path],
    cameras: Cameras,
    camera_mask_path: Optional[str] = None,
    use_gpu: bool = True,
    verbose: bool = False,
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "exhaustive",
    save_path: Optional[Path] = None,
    use_guided_matching: bool = False,
    vocab_tree_path: Optional[str] = None,
    max_images: int = 0,
) -> None:
    """Creates a sparse 3D pointcloud from given posed images using COLMAP."""
    # initialize colmap
    colmap_cmd = "colmap"
    colmap_dir = base_dir / "colmap"
    image_dir = colmap_dir / "images"
    if colmap_dir.exists():
        shutil.rmtree(colmap_dir)
    image_dir.mkdir(parents=True, exist_ok=False)

    if max_images > 0 and len(image_filenames) > max_images:
        indices = sample_farthest_points(cameras.camera_to_worlds[:, :3, 3], max_images)
        image_filenames = [image_filenames[i] for i in indices]
        cameras = Cameras.cat([cameras[i].unsqueeze() for i in indices])

    for im in image_filenames:
        im_str = str(im)
        if ".zip/" in im_str:
            base_name = im_str.split("/")[-1]
            target_path = Path(image_dir) / base_name
            data = load_bytes(im_str)
            with open(target_path, "wb") as f:
                f.write(data)
        else:
            shutil.copy(im_str, image_dir)

    camera_type = CameraType._value2member_map_[cameras.camera_type[0].item()]
    assert len(cameras.camera_type.unique()) == 1, "All camera types must be the same."
    assert camera_type == CameraType.PERSPECTIVE, "Only perspective cameras supported for now."
    if cameras.distortion_params is not None and not (cameras.distortion_params == 0).all():
        camera_model = "FULL_OPENCV"
    else:
        camera_model = "PINHOLE"

    # Feature extraction
    if not os.path.exists(colmap_dir / "database.db"):
        feature_extractor_cmd = [
            f"{colmap_cmd} feature_extractor",
            f"--database_path {colmap_dir / 'database.db'}",
            f"--image_path {image_dir}",
            f"--ImageReader.camera_model {camera_model}",
            f"--SiftExtraction.use_gpu {int(use_gpu)}",
        ]
        if camera_mask_path is not None:
            feature_extractor_cmd.append(f"--ImageReader.camera_mask_path {camera_mask_path}")
        feature_extractor_cmd = " ".join(feature_extractor_cmd)
        with status(msg="[bold yellow]Running COLMAP feature extractor...", spinner="moon", verbose=verbose):
            run_command(feature_extractor_cmd, verbose=verbose)

        CONSOLE.log("[bold green]:tada: Done extracting COLMAP features.")

        # Feature matching
        assert os.path.exists(colmap_dir / "database.db")
        feature_matcher_cmd = [
            f"{colmap_cmd} {matching_method}_matcher",
            f"--database_path {colmap_dir / 'database.db'}",
            f"--SiftMatching.use_gpu {int(use_gpu)}",
            f"--SiftMatching.guided_matching {int(use_guided_matching)}",
        ]
        if matching_method == "vocab_tree":
            feature_matcher_cmd.append(f'--VocabTreeMatching.vocab_tree_path "{vocab_tree_path}"')
        feature_matcher_cmd = " ".join(feature_matcher_cmd)
        with status(msg="[bold yellow]Running COLMAP feature matcher...", spinner="runner", verbose=verbose):
            run_command(feature_matcher_cmd, verbose=verbose)
        CONSOLE.log("[bold green]:tada: Done matching COLMAP features.")
    else:
        CONSOLE.log(
            f"Found existing database at {colmap_dir / 'database.db'}. Skipping feature extraction and matching."
        )

    # load database to get camera / image ids
    assert os.path.exists(colmap_dir / "database.db")
    db = COLMAPDatabase.connect(colmap_dir / "database.db")

    # filter images without sufficient matches
    images_with_matches = defaultdict(list)
    for pair_id, data in db.execute("SELECT pair_id, data FROM matches"):
        if data is None:
            continue
        image1, image2 = pair_id_to_image_ids(pair_id)
        images_with_matches[int(image1)] += [1]
        images_with_matches[int(image2)] += [1]
    images_with_matches = set((k for k, v in images_with_matches.items() if sum(v) > 1))

    image_map = {}
    for data in db.execute("SELECT * FROM images"):
        image_id, image_name, camera_id = data[:3]
        image_map[image_name.lower()] = (image_id, camera_id)
    db.close()

    # create initial sparse model from cameras only
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_file = ""
    cameras_file = ""
    for im, cam in zip(image_filenames, cameras):
        # opengl to opencv
        c2w = to4x4(cam.camera_to_worlds) @ opencv_to_opengl
        # colmap uses w2c format instead of c2w
        w2c = inverse_rigid_transform(c2w)
        tx, ty, tz = w2c[:3, 3].cpu().numpy().tolist()
        qw, qx, qy, qz = quaternion_from_matrix(w2c[:3, :3].cpu().numpy()).tolist()
        image_id, camera_id = image_map[os.path.basename(im).lower()]

        # skip images without matches (these will not end up in the correspondence graph)
        # workaround for https://github.com/colmap/colmap/issues/2297
        if image_id not in images_with_matches:
            continue

        images_file += f"{image_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {camera_id} {os.path.basename(im)}\n\n"
        fx, fy, cx, cy = cam.fx.item(), cam.fy.item(), cam.cx.item(), cam.cy.item()
        cameras_file += (
            f"{camera_id} {camera_model} {cam.width.item()} {cam.height.item()} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}"
        )
        if camera_model != "PINHOLE":
            assert cam.distortion_params is not None
            k1, k2, k3, k4, p1, p2 = cam.distortion_params.cpu().numpy().tolist()
            # colmap uses k1, k2, p1, p2, k3, k4, k5, k6 for FULL_OPENCV
            # See https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
            cameras_file += f" {k1:.6f} {k2:.6f} {p1:.6f} {p2:.6f} {k3:.6f} {k4:.6f} 0.000000 0.000000"
        cameras_file += "\n"

    with open(sparse_dir / "images.txt", "w") as f:
        f.write(images_file)
    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write(cameras_file)
    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("")

    # 3D Point triangulation
    triangulator_cmd = [
        f"{colmap_cmd} point_triangulator",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        f"--input_path {sparse_dir}",
        f"--output_path {sparse_dir}",
    ]
    triangulator_cmd = " ".join(triangulator_cmd)

    with status(
        msg="[bold yellow]Running COLMAP 3D point triangulation...",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(triangulator_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done COLMAP 3D point triangulation.")

    # create ply
    if save_path is not None:
        filename, save_dir = os.path.basename(save_path), Path(os.path.dirname(save_path))
    else:
        filename, save_dir = "pointcloud.ply", base_dir
    create_ply_from_colmap(
        filename=filename,
        recon_dir=sparse_dir,
        output_dir=save_dir,
        applied_transform=None,
    )
    CONSOLE.log(f"Pointcloud saved to {save_dir / filename}")
