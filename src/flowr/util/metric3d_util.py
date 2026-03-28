import cv2
import numpy as np
import torch
from PIL import Image

DATA_BASIC = dict(
    canonical_space=dict(
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(0.1, 200),
    crop_size=(616, 1064),  # %28 = 0
    clip_depth_range=(0.1, 200),
    vit_size=(616, 1064),
)


def build_camera_model(H: int, W: int, intrinsics: list) -> np.array:
    """
    Encode the camera intrinsic parameters (focal length and principle point) to a 4-channel map.
    """
    fx, fy, u0, v0 = intrinsics
    f = (fx + fy) / 2.0
    # principle point location
    x_row = np.arange(0, W).astype(np.float32)
    x_row_center_norm = (x_row - u0) / W
    x_center = np.tile(x_row_center_norm, (H, 1))  # [H, W]

    y_col = np.arange(0, H).astype(np.float32)
    y_col_center_norm = (y_col - v0) / H
    y_center = np.tile(y_col_center_norm, (W, 1)).T  # [H, W]

    # FoV
    fov_x = np.arctan(x_center / (f / W))
    fov_y = np.arctan(y_center / (f / H))

    cam_model = np.stack([x_center, y_center, fov_x, fov_y], axis=2)
    return cam_model


def resize_for_input(image, output_shape, intrinsic, canonical_shape, to_canonical_ratio):
    """
    Resize the input.
    Resizing consists of two processed, i.e. 1) to the canonical space (adjust the camera model); 2) resize the image while the camera model holds. Thus the
    label will be scaled with the resize factor.
    """
    padding = [123.675, 116.28, 103.53]
    h, w, _ = image.shape
    resize_ratio_h = output_shape[0] / canonical_shape[0]
    resize_ratio_w = output_shape[1] / canonical_shape[1]
    to_scale_ratio = min(resize_ratio_h, resize_ratio_w)

    resize_ratio = to_canonical_ratio * to_scale_ratio

    reshape_h = int(resize_ratio * h)
    reshape_w = int(resize_ratio * w)

    pad_h = max(output_shape[0] - reshape_h, 0)
    pad_w = max(output_shape[1] - reshape_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    # resize
    image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
    # padding
    image = cv2.copyMakeBorder(
        image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding
    )

    # Resize, adjust principle point
    intrinsic[2] = intrinsic[2] * to_scale_ratio
    intrinsic[3] = intrinsic[3] * to_scale_ratio

    cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
    cam_model = cv2.copyMakeBorder(
        cam_model, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=-1
    )

    pad = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    label_scale_factor = 1 / to_scale_ratio
    return image, cam_model, pad, label_scale_factor


def transform_test_data_scalecano(rgb, intrinsic):
    """
    Pre-process the input for forwarding. Employ `label scale canonical transformation.'
        Args:
            rgb: input rgb image. [H, W, 3]
            intrinsic: camera intrinsic parameter, [fx, fy, u0, v0]
            data_basic: predefined canonical space in configs.
    """
    canonical_space = DATA_BASIC["canonical_space"]
    forward_size = DATA_BASIC["crop_size"]
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    ori_h, ori_w, _ = rgb.shape
    ori_focal = (intrinsic[0] + intrinsic[1]) / 2
    canonical_focal = canonical_space["focal_length"]

    cano_label_scale_ratio = canonical_focal / ori_focal

    canonical_intrinsic = [
        intrinsic[0] * cano_label_scale_ratio,
        intrinsic[1] * cano_label_scale_ratio,
        intrinsic[2],
        intrinsic[3],
    ]

    # resize
    rgb, cam_model, pad, resize_label_scale_ratio = resize_for_input(
        rgb, forward_size, canonical_intrinsic, [ori_h, ori_w], 1.0
    )

    # label scale factor
    label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio

    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb.unsqueeze(0).cuda()

    cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
    cam_model = cam_model[None, :, :, :].cuda()
    cam_model_stacks = [
        torch.nn.functional.interpolate(
            cam_model, size=(cam_model.shape[2] // i, cam_model.shape[3] // i), mode="bilinear", align_corners=False
        )
        for i in [2, 4, 8, 16, 32]
    ]
    return rgb, cam_model_stacks, pad, label_scale_factor


@torch.no_grad()
def get_prediction(rgb_input, cam_models_stacks, pad, model, height, width, label_scale_factor):
    normalize_scale = DATA_BASIC["depth_range"][1]
    pred_depth, confidence, output_dict = model.inference(
        {
            "input": rgb_input,
            "cam_model": cam_models_stacks,
        }
    )
    pred_depth = pred_depth.squeeze()
    confidence = confidence.squeeze()
    pred_depth = pred_depth[pad[0] : pred_depth.shape[0] - pad[1], pad[2] : pred_depth.shape[1] - pad[3]]
    confidence = confidence[pad[0] : confidence.shape[0] - pad[1], pad[2] : confidence.shape[1] - pad[3]]
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None, :, :], [height, width], mode="bilinear"
    ).squeeze()  # to original size
    confidence = torch.nn.functional.interpolate(
        confidence[None, None, :, :], [height, width], mode="bilinear"
    ).squeeze()  # to original size
    pred_depth = pred_depth * normalize_scale / label_scale_factor
    return pred_depth.cpu().numpy(), confidence.cpu().numpy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--u0", type=float, default=None)
    parser.add_argument("--v0", type=float, default=1.0)
    args = parser.parse_args()
    img = np.array(Image.open(args.image_path))
    height, width = img.shape[:2]
    model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_large", pretrain=True).cuda()
    intrinsic = [
        args.fx,
        args.fy if args.fy is not None else args.fx,
        args.u0 if args.u0 is not None else width / 2.0,
        args.v0 if args.v0 is not None else height / 2.0,
    ]
    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(img, intrinsic)
    pred_depth, confidence = get_prediction(rgb_input, cam_models_stacks, pad, model, height, width, label_scale_factor)

    print("depth median", np.median(pred_depth))
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    output_path = "side_by_side_visualization.png"
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    depth_map = axes[1].imshow(pred_depth, cmap="viridis")
    axes[1].set_title("Predicted Depth")
    axes[1].axis("off")
    fig.colorbar(depth_map, ax=axes[1], orientation="vertical", fraction=0.046, pad=0.04)
    confidence_map = axes[2].imshow(confidence, cmap="magma")
    axes[2].set_title("Confidence")
    axes[2].axis("off")
    fig.colorbar(confidence_map, ax=axes[2], orientation="vertical", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path)

    # make pointcloud, save to file
    from flowr.common.geometry import depth_to_points
    from flowr.common.io import to_ply

    # image/depth prediction to points/colors
    intrinsic_matrix = torch.eye(3)
    intrinsic_matrix[0, 0] = intrinsic[0]
    intrinsic_matrix[1, 1] = intrinsic[1]
    intrinsic_matrix[0, 2] = intrinsic[2]
    intrinsic_matrix[1, 2] = intrinsic[3]
    points = depth_to_points(torch.from_numpy(pred_depth), intrinsic_matrix)
    colors = img.reshape(-1, 3)
    mask = confidence.flatten() > 0.9
    points = points[mask]
    colors = colors[mask]
    to_ply("pointcloud.ply", points, colors)
