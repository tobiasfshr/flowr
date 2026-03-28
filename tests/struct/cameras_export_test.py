import torch

from flowr.struct.cameras import Cameras, CameraType


def test_json_export_import(tmp_path):
    num_images = 5
    poses = torch.rand((num_images, 3, 4))
    dist_params = torch.rand(num_images, 6)
    times = torch.rand(num_images, 1)
    metadata = {
        "something1": torch.rand((num_images, 1)),
        "something4": torch.rand((num_images, 4)),
        "somethingbool": torch.rand((num_images, 1)).bool(),
        "somethingint": torch.rand((num_images, 1)).long(),
    }
    fx, fy, cx, cy = 300.0, 300.0, 64.0, 64.0
    width, height = 128, 128
    cameras = Cameras(
        camera_to_worlds=poses,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        distortion_params=dist_params,
        camera_type=CameraType.PERSPECTIVE,
        times=times,
        metadata=metadata,
    )

    cameras.to_json(tmp_path / "cameras.json")
    exported_cams = Cameras.from_json(tmp_path / "cameras.json")

    assert torch.allclose(exported_cams.camera_to_worlds, poses)
    assert (exported_cams.fx == 300.0).all()
    assert (exported_cams.fy == 300.0).all()
    assert (exported_cams.cx == 64.0).all()
    assert (exported_cams.cy == 64.0).all()
    assert (exported_cams.width == 128).all()
    assert (exported_cams.height == 128).all()
    assert (exported_cams.camera_type == 1).all()
    assert torch.allclose(exported_cams.times, times)
    assert set(metadata.keys()) == set(exported_cams.metadata.keys())
    for key, value in metadata.items():
        assert value.dtype == exported_cams.metadata[key].dtype
        assert value.shape == exported_cams.metadata[key].shape
        assert torch.allclose(value, exported_cams.metadata[key])
