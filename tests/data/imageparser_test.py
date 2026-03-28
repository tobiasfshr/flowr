import json
import os

import numpy as np
import torch

from flowr.common.io import load_image, save_image
from flowr.data.parser.image import ImageDataParser, ImageDataParserConfig
from flowr.struct.cameras import Cameras


def test_imageparser(tmp_path):
    num_images = 5
    poses = torch.rand((num_images, 3, 4))
    cameras = Cameras(camera_to_worlds=poses, fx=300.0, fy=300.0, cx=64.0, cy=64.0, width=128, height=128)
    images = [torch.randint(0, 255, (3, 128, 128)).float() / 255 for _ in range(num_images)]
    cam_dict = {}
    for i, (im, cam) in enumerate(zip(images, cameras)):
        save_image(im, tmp_path / f"{i:04d}.png")
        cam_dict[f"{i:04d}.png"] = cam.to_dict()
    with open(tmp_path / "cameras.json", "w") as f:
        json.dump(cam_dict, f)

    dp: ImageDataParser = ImageDataParserConfig(data=tmp_path, load_3D_points=False).setup()
    dpo = dp._generate_dataparser_outputs(split="train")
    for i, (im, cam) in enumerate(zip(dpo.image_filenames, dpo.cameras)):
        loaded_im = (torch.from_numpy(np.array(load_image(im))).float() / 255).permute(2, 0, 1)
        assert os.path.basename(im) == f"{i:04d}.png"
        assert torch.allclose(cam.camera_to_worlds, poses[i])
        assert torch.allclose(loaded_im, images[i], atol=1e-4)
