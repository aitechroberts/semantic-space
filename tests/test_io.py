"""Unit tests for stages/io.py — serialization, deserialization, file I/O."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestSerializeDeserializeRoundtrip:
    """The critical test: serialize a live detection, deserialize it, verify types."""

    def test_axis_aligned_bbox_roundtrip(self):
        import open3d as o3d
        import torch
        from conceptgraph.stages.io import serialize_detection, deserialize_detection

        pcd = o3d.geometry.PointCloud()
        pts = np.random.randn(100, 3)
        colors = np.random.rand(100, 3)
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        bbox = pcd.get_axis_aligned_bounding_box()

        det = {
            "pcd": pcd,
            "bbox": bbox,
            "clip_ft": torch.randn(512),
            "text_ft": torch.randn(512),
            "vlm_vit_ft": torch.randn(256),
            "vlm_proj_ft": None,
            "class_name": "chair",
            "class_id": [3],
            "curr_obj_num": 7,
        }

        serialized = serialize_detection(det, spatial_sim_type="iou")

        assert isinstance(serialized["pcd_points"], np.ndarray)
        assert isinstance(serialized["pcd_colors"], np.ndarray)
        assert serialized["bbox_type"] == "axis_aligned"
        assert serialized["class_name"] == "chair"
        assert serialized["n_points"] == 100

        restored = deserialize_detection(serialized, device="cpu")

        assert isinstance(restored["pcd"], o3d.geometry.PointCloud)
        assert isinstance(restored["bbox"], o3d.geometry.AxisAlignedBoundingBox)
        assert isinstance(restored["clip_ft"], torch.Tensor)
        assert restored["vlm_proj_ft"] is None
        assert restored["class_name"] == "chair"
        assert len(np.asarray(restored["pcd"].points)) == 100

        np.testing.assert_allclose(
            np.asarray(restored["pcd"].points), pts, atol=1e-10,
        )
        np.testing.assert_allclose(
            restored["clip_ft"].numpy(), det["clip_ft"].numpy(), atol=1e-6,
        )

    def test_oriented_bbox_roundtrip(self):
        import open3d as o3d
        import torch
        from conceptgraph.stages.io import serialize_detection, deserialize_detection

        pcd = o3d.geometry.PointCloud()
        pts = np.random.randn(50, 3) + [1, 2, 3]
        pcd.points = o3d.utility.Vector3dVector(pts)

        bbox = pcd.get_oriented_bounding_box()

        det = {
            "pcd": pcd,
            "bbox": bbox,
            "clip_ft": torch.randn(512),
            "text_ft": torch.randn(512),
            "vlm_vit_ft": None,
            "vlm_proj_ft": None,
            "class_name": "table",
            "class_id": [1],
            "curr_obj_num": 0,
        }

        serialized = serialize_detection(det, spatial_sim_type="oriented")
        assert serialized["bbox_type"] == "oriented"

        restored = deserialize_detection(serialized, device="cpu")
        assert isinstance(restored["bbox"], o3d.geometry.OrientedBoundingBox)


class TestSaveLoadRawDet:
    """Test save/load roundtrip for raw detection metadata."""

    def test_roundtrip(self):
        from conceptgraph.stages.io import save_raw_det, load_raw_det
        from conceptgraph.stages.paths import make_empty_gobs

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            gobs = make_empty_gobs(5, feat_dim=128)
            save_raw_det(path, 42, gobs)

            loaded = load_raw_det(path, 42)
            assert loaded is not None
            assert loaded["xyxy"].shape == (5, 4)
            assert len(loaded["captions"]) == 5

    def test_missing_file_returns_none(self):
        from conceptgraph.stages.io import load_raw_det

        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_raw_det(Path(tmpdir), 999)
            assert result is None


class TestSaveLoadFrameData:
    """Test save/load roundtrip for processed frame data."""

    def test_roundtrip(self):
        from conceptgraph.stages.io import save_frame_data, load_frame_data
        from conceptgraph.stages.paths import FrameDataRecord, SerializedDetection

        record = FrameDataRecord(
            frame_idx=10,
            color_path="/some/path.png",
            skip_matching=False,
            surviving_indices=np.array([0, 2, 5], dtype=np.int32),
            detections=[
                SerializedDetection(
                    pcd_points=np.random.randn(20, 3),
                    pcd_colors=np.random.rand(20, 3),
                    bbox_corners=np.random.randn(8, 3),
                    bbox_type="axis_aligned",
                    clip_ft=np.random.randn(512).astype(np.float32),
                    text_ft=np.random.randn(512).astype(np.float32),
                    vlm_vit_ft=None,
                    vlm_proj_ft=None,
                    class_name="object",
                    class_id=0,
                    inst_id=0,
                    n_points=20,
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            save_frame_data(path, 10, record)

            loaded = load_frame_data(path, 10)
            assert loaded is not None
            assert loaded["frame_idx"] == 10
            assert loaded["color_path"] == "/some/path.png"
            assert len(loaded["detections"]) == 1
            np.testing.assert_array_equal(loaded["surviving_indices"], np.array([0, 2, 5]))


class TestListFrameIndices:
    """Test frame index discovery."""

    def test_discovers_indices(self):
        from conceptgraph.stages.io import save_raw_det, list_frame_indices
        from conceptgraph.stages.paths import make_empty_gobs

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            for idx in [5, 1, 10]:
                save_raw_det(path, idx, make_empty_gobs(1))

            indices = list_frame_indices(path)
            assert indices == [1, 5, 10]

    def test_empty_dir(self):
        from conceptgraph.stages.io import list_frame_indices

        with tempfile.TemporaryDirectory() as tmpdir:
            indices = list_frame_indices(Path(tmpdir))
            assert indices == []
