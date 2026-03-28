# ruff: noqa: E402
"""Utility for creating a pointcloud from (un-)posed images."""
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import subprocess
import pycolmap
import torch
from kapture.converter.colmap.database import COLMAPDatabase
from kapture.converter.colmap.database_extra import kapture_to_colmap
from nerfstudio.cameras.camera_utils import quaternion_from_matrix
from nerfstudio.process_data.colmap_utils import create_ply_from_colmap
from nerfstudio.utils.poses import to4x4
from nerfstudio.utils.rich_utils import CONSOLE
from pytorch3d.renderer import PerspectiveCameras

from flowr.common.geometry import inverse_rigid_transform, opencv_to_opengl
from flowr.common.io import to_ply
from flowr.struct.cameras import Cameras
from flowr.util.colmap.scale_to_metric import calculate_scale_factor, scale_reconstruction
from flowr.util.metric3d_util import get_prediction, transform_test_data_scalecano

sys.path.append("./extern/")
sys.path.append("./extern/dust3r")

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.cloud_opt.optimizer import PointCloudOptimizer
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images
from mast3r.colmap.mapping import kapture_import_image_folder_or_list, run_mast3r_matching
from mast3r.image_pairs import make_pairs
from mast3r.model import AsymmetricMASt3R


def create_ply_from_posed_images(
    image_filenames: List[Path],
    cameras: Cameras,
    save_path: Path,
    device: torch.device = torch.device("cuda"),
    mast3r_ckpt: str = "./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    mast3r_retrieval_ckpt: Optional[
        str
    ] = "./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
    use_mast3r_GA: bool = False,
    niter: int = 300,
    schedule: str = "linear",
    lr: float = 0.01,
    min_conf_thr: float = 3.0,  # minimum=1.0, maximum=20
    bg_trd: float = 0.2,
    image_size: int = 512,
    min_views_for_retrieval: int = 12,
    winsize: int = 5,
    dense_matching: bool = True,
    scale_path: Optional[Path] = None,
) -> None:
    """Creates a sparse 3D pointcloud from given posed images using MASt3R."""
    assert len(image_filenames) > 1

    # load model, inputs
    mast3r = AsymmetricMASt3R.from_pretrained(mast3r_ckpt).to(device)
    input_images = load_images([str(im) for im in image_filenames], size=image_size)

    # mast3r sfm sparse view graph construction
    if mast3r_retrieval_ckpt is not None and len(image_filenames) > min_views_for_retrieval:
        if cameras is None:
            from mast3r.retrieval.processor import Retriever

            retriever = Retriever(mast3r_retrieval_ckpt, backbone=mast3r)
            with torch.no_grad():
                sim_matrix = retriever([str(im) for im in image_filenames])
            del retriever
            torch.cuda.empty_cache()
        else:
            # use translation / rotation difference as similarity
            c2w = cameras.camera_to_worlds
            t_distance = (c2w[:, None, :3, 3] - c2w[None, :, :3, 3]).norm(dim=-1)
            r_distance = (c2w[:, None, :3, :3] - c2w[None, :, :3, :3]).pow_(2).sum((-1, -2))
            distance = 0.6 * r_distance + 0.1 * t_distance
            distance = distance / distance.max()
            sim_matrix = 1.0 - distance.cpu().numpy()

        num_keyframes = round(math.sqrt(len(image_filenames)))

        # if images are from two folders ('train' and 'other'), image indices in train should be keyframes
        if any(["/train/" in str(im) for im in image_filenames]) and any(
            ["/other/" in str(im) for im in image_filenames]
        ):
            keyframes = [i for i in range(len(image_filenames)) if "/train/" in str(image_filenames[i])]
            # subsample to num_keyframes if necessary
            if len(keyframes) > num_keyframes:
                indices = np.linspace(0, len(keyframes) - 1, num_keyframes, dtype=int)
                keyframes = [keyframes[i] for i in indices]
            num_keyframes = len(keyframes)
        else:
            keyframes = None

        pairs = make_pairs(
            input_images,
            f"retrieval-{num_keyframes}-{winsize}",
            prefilter=None,
            symmetrize=True,
            sim_mat=sim_matrix,
            keyframes=keyframes,
        )
    else:
        pairs = make_pairs(input_images, scene_graph="complete", prefilter=None, symmetrize=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_mast3r_GA:
        output = inference(pairs, mast3r, device, batch_size=1)
        scene = global_alignment(output, cameras, device, niter, schedule, lr, min_conf_thr, bg_trd, save_path)
        if scale_path is not None and not (scale_path / "scale_factor.txt").exists():
            metric3d_model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_large", pretrain=True).to(device)
            mast3r_depths, metric3d_depths, confidences = [], [], []
            for im, depth, focal, pp in zip(
                scene.imgs, scene.get_depthmaps(), scene.get_focals(), scene.get_principal_points()
            ):
                intrinsic = [focal.item(), focal.item(), pp[0].item(), pp[1].item()]
                height, width = im.shape[:2]
                rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(
                    im * 255, intrinsic
                )
                pred_depth, confidence = get_prediction(
                    rgb_input, cam_models_stacks, pad, metric3d_model, height, width, label_scale_factor
                )
                depth = depth.detach().cpu().numpy()
                mask = (depth > 0) & (pred_depth > 0)
                mast3r_depths.append(depth[mask])
                metric3d_depths.append(pred_depth[mask])
                confidences.append(confidence[mask])

            mast3r_depths = np.concatenate(mast3r_depths)
            metric3d_depths = np.concatenate(metric3d_depths)
            confidences = np.concatenate(confidences)
            CONSOLE.log("Calculating scale factor...")
            scale_factor = calculate_scale_factor(mast3r_depths, metric3d_depths, confidences)
            CONSOLE.log(f"Scale factor: {scale_factor}")
            with open(scale_path / "scale_factor.txt", "w") as f:
                f.write(str(scale_factor))
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            colmap_alignment(
                Path(temp_dir),
                image_filenames,
                mast3r,
                pairs,
                cameras,
                device,
                image_size,
                save_path,
                dense_matching=dense_matching,
            )
            if scale_path is not None and not (scale_path / "scale_factor.txt").exists():
                scale_factor = scale_reconstruction(Path(temp_dir) / "sparse", Path(temp_dir) / "images")
                with open(scale_path / "scale_factor.txt", "w") as f:
                    f.write(str(scale_factor))


def colmap_alignment(
    colmap_dir: Path,
    filelist: List[Path],
    model: AsymmetricMASt3R,
    pairs: List[Tuple[Dict, Dict]],
    cameras: Cameras,
    device: torch.device,
    image_size: int,
    save_path: Path,
    shared_intrinsics: bool = False,
    colmap_cmd: str = "colmap",
    verbose: bool = False,
    camera_model: str = "PINHOLE",
    dense_matching: bool = True,
):
    sparse_dir = colmap_dir / "sparse"
    images_dir = colmap_dir / "images"

    # copy filelist to images_dir with new names
    tmp_filelist = []
    os.makedirs(images_dir, exist_ok=True)
    for i, file in enumerate(filelist):
        suffix = file.suffix.lower()
        framename = f"{i:08d}{suffix}"
        shutil.copyfile(file, images_dir / framename)
        tmp_filelist.append(images_dir / framename)
    filelist = tmp_filelist

    root_path = str(images_dir)

    root_path = os.path.commonpath(filelist)
    filelist_relpath = [os.path.relpath(filename, root_path).replace("\\", "/") for filename in filelist]
    kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), shared_intrinsics)
    image_pairs = [(filelist_relpath[img1["idx"]], filelist_relpath[img2["idx"]]) for img1, img2 in pairs]

    colmap_db_path = os.path.join(colmap_dir, "colmap.db")
    if os.path.isfile(colmap_db_path):
        os.remove(colmap_db_path)

    os.makedirs(os.path.dirname(colmap_db_path), exist_ok=True)
    colmap_db = COLMAPDatabase.connect(colmap_db_path)
    kapture_to_colmap(
        kdata,
        root_path,
        tar_handler=None,
        database=colmap_db,
        keypoints_type=None,
        descriptors_type=None,
        export_two_view_geometry=False,
    )
    colmap_image_pairs = run_mast3r_matching(
        model, image_size, 16, device, kdata, root_path, image_pairs, colmap_db, dense_matching, 5, 1.001, True, 3
    )
    image_map = {}
    for data in colmap_db.execute("SELECT * FROM images"):
        image_id, image_name, camera_id = data[:3]
        image_map[image_name.lower()] = (image_id, camera_id)

    colmap_db.close()

    # colmap db is now full, run colmap
    CONSOLE.log("verify_matches")
    f = open(colmap_dir / "pairs.txt", "w")
    for image_path1, image_path2 in colmap_image_pairs:
        f.write("{} {}\n".format(image_path1, image_path2))
    f.close()
    pycolmap.verify_matches(colmap_db_path, colmap_dir / "pairs.txt")

    # create initial sparse model from cameras only
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_file = ""
    cameras_file = ""
    for im, cam in zip(filelist, cameras):
        # opengl to opencv
        c2w = to4x4(cam.camera_to_worlds) @ opencv_to_opengl
        # colmap uses w2c format instead of c2w
        w2c = inverse_rigid_transform(c2w)
        tx, ty, tz = w2c[:3, 3].cpu().numpy().tolist()
        qw, qx, qy, qz = quaternion_from_matrix(w2c[:3, :3].cpu().numpy()).tolist()
        image_id, camera_id = image_map[os.path.basename(im).lower()]

        images_file += f"{image_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {camera_id} {os.path.basename(im)}\n\n"
        fx, fy, cx, cy = cam.fx.item(), cam.fy.item(), cam.cx.item(), cam.cy.item()
        cameras_file += (
            f"{camera_id} {camera_model} {cam.width.item()} {cam.height.item()} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}"
        )
        cameras_file += "\n"

    with open(sparse_dir / "images.txt", "w") as f:
        f.write(images_file)
    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write(cameras_file)
    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("")

    # 3D Point triangulation
    CONSOLE.log("triangulate points")
    triangulator_cmd = [
        f"{colmap_cmd} point_triangulator",
        f"--database_path {colmap_db_path}",
        f"--image_path {images_dir}",
        f"--input_path {sparse_dir}",
        f"--output_path {sparse_dir}",
    ]
    triangulator_cmd = " ".join(triangulator_cmd)


    result = subprocess.run(triangulator_cmd, shell=True, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"COLMAP point_triangulator failed (code {result.returncode}): {stderr[:500]}")

    # create ply
    filename, save_dir = os.path.basename(save_path), Path(os.path.dirname(save_path))
    create_ply_from_colmap(
        filename=filename,
        recon_dir=sparse_dir,
        output_dir=save_dir,
        applied_transform=None,
    )


def global_alignment(output, cameras, device, niter, schedule, lr, min_conf_thr, bg_trd, save_path):
    # global alignment
    scene: PointCloudOptimizer = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

    # set poses, intrinics and freeze (no grad)
    cameras = cameras_nerfstudio2pytorch3d(cameras)
    scene.preset_pose(torch.cat([cameras.R, cameras.T[..., None]], -1))
    assert (cameras.focal_length == cameras.focal_length[0]).all(), "all fx/fy must be the same"
    assert (cameras.principal_point == cameras.principal_point[0]).all(), "all cx/cy must be the same"
    scale_factor = torch.tensor(scene.imshapes) / cameras.image_size
    # mean because dust3r assumes fx==fy
    focals = (cameras.focal_length * scale_factor).mean(dim=-1)
    pps = cameras.principal_point * scale_factor
    scene.preset_focal(focals)
    scene.preset_principal_point(pps)

    # run global alignment
    scene.compute_global_alignment(init="known_poses", niter=niter, schedule=schedule, lr=lr)
    scene.clean_pointcloud()

    # mask for cleaner pointcloud
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    masks = scene.get_masks()
    depth = scene.get_depthmaps()
    bgs_mask = [dpt > bg_trd * (torch.max(dpt[40:-40, :]) + torch.min(dpt[40:-40, :])) for dpt in depth]
    masks = [m + mb for m, mb in zip(masks, bgs_mask)]

    # save out pointcloud in nerfstudio format
    pcd = [i.detach() for i in scene.get_pts3d(clip_thred=1.0)]  # a list of points of size whc

    device, dtype = pcd[0].device, pcd[0].dtype
    transform = torch.tensor([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], device=device, dtype=dtype)
    # RDF dust3r to nerfstudio, save to ply
    save_pointcloud(np.array(scene.imgs), [(transform.T @ p[..., None]).squeeze(-1) for p in pcd], save_path, masks)
    return scene


def cameras_nerfstudio2pytorch3d(cameras: Cameras, convention="RDF"):
    device = cameras.device
    R = cameras.camera_to_worlds[:, :3, :3]
    T = cameras.camera_to_worlds[:, :3, 3]
    fs = torch.cat([cameras.fx, cameras.fy], -1)
    c = torch.cat([cameras.cx, cameras.cy], -1)
    image_size = torch.cat([cameras.height, cameras.width], -1)
    dtype = R.dtype
    # BRU (nerfstudio) + opengl to ... + opencv
    opengl_to_opencv = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=dtype)
    if convention == "RDF":
        transform = torch.tensor([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], device=device, dtype=dtype)
    else:
        raise ValueError(f"unknown convention {convention}")
    R = transform @ R @ opengl_to_opencv
    T = (transform @ T[..., None]).squeeze(-1)
    cameras = PerspectiveCameras(
        focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R, T=T, device=device
    )
    return cameras


def save_pointcloud(imgs, pts3d, save_path, mask=None):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)

    if mask is not None:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    else:
        pts = np.concatenate([p for p in pts3d])
        col = np.concatenate([p for p in imgs])

    pts = pts.reshape(-1, 3)
    col = col.reshape(-1, 3) * 255
    to_ply(save_path, pts, col)
