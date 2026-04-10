"""Unit tests for geometry projection and mesh_io modules."""

import numpy as np
import pytest


class TestProjectPointsToFrame:
    """Tests for projection.project_points_to_frame."""

    def test_point_in_front_of_camera(self):
        from conceptgraph.slam.geometry.projection import project_points_to_frame

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        pose_c2w = np.eye(4)
        points = np.array([[0.0, 0.0, 5.0]])

        coords, valid = project_points_to_frame(points, pose_c2w, K, H=480, W=640)

        assert valid[0], "Point directly in front of camera should be valid"
        np.testing.assert_allclose(coords[0], [320.0, 240.0], atol=1e-6)

    def test_point_behind_camera(self):
        from conceptgraph.slam.geometry.projection import project_points_to_frame

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        pose_c2w = np.eye(4)
        points = np.array([[0.0, 0.0, -5.0]])

        _, valid = project_points_to_frame(points, pose_c2w, K, H=480, W=640)
        assert not valid[0], "Point behind camera should be invalid"

    def test_point_out_of_bounds(self):
        from conceptgraph.slam.geometry.projection import project_points_to_frame

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        pose_c2w = np.eye(4)
        points = np.array([[100.0, 100.0, 1.0]])

        _, valid = project_points_to_frame(points, pose_c2w, K, H=480, W=640)
        assert not valid[0], "Point projecting outside image bounds should be invalid"

    def test_multiple_points(self):
        from conceptgraph.slam.geometry.projection import project_points_to_frame

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        pose_c2w = np.eye(4)
        points = np.array([
            [0.0, 0.0, 5.0],    # center, valid
            [0.0, 0.0, -5.0],   # behind, invalid
            [0.1, 0.0, 5.0],    # slightly off-center, valid
        ])

        coords, valid = project_points_to_frame(points, pose_c2w, K, H=480, W=640)
        assert valid[0] and not valid[1] and valid[2]

    def test_rotated_camera(self):
        from conceptgraph.slam.geometry.projection import project_points_to_frame

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        pose_c2w = np.eye(4)
        pose_c2w[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        points = np.array([[5.0, 0.0, 0.0]])

        coords, valid = project_points_to_frame(points, pose_c2w, K, H=480, W=640)
        assert valid[0], "Point should be visible after camera rotation"


class TestSelectBestViews:
    """Tests for projection.select_best_views."""

    def test_basic_ranking(self):
        from conceptgraph.slam.geometry.projection import select_best_views

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        points = np.random.randn(200, 3)
        points[:, 2] = np.abs(points[:, 2]) + 2.0

        frames = [
            {"pose": np.eye(4), "intrinsics": K, "H": 480, "W": 640},
            {"pose": np.eye(4) * 0.001, "intrinsics": K, "H": 480, "W": 640},
        ]
        frames[1]["pose"][3, 3] = 1.0
        frames[1]["pose"][:3, 3] = [100, 100, 100]

        result = select_best_views(points, frames, top_k=2, min_visible=10)
        assert len(result) >= 1, "Should find at least one view with visible points"

    def test_no_views_pass_threshold(self):
        from conceptgraph.slam.geometry.projection import select_best_views

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        points = np.array([[0.0, 0.0, 5.0]])

        frames = [{"pose": np.eye(4), "intrinsics": K, "H": 480, "W": 640}]
        result = select_best_views(points, frames, top_k=5, min_visible=100)
        assert len(result) == 0, "Single point should not meet min_visible=100"


class TestMakeEmptyGobs:
    """Tests for paths.make_empty_gobs."""

    def test_has_all_keys(self):
        from conceptgraph.stages.paths import RawGobs, make_empty_gobs

        gobs = make_empty_gobs(5)
        expected_keys = set(RawGobs.__annotations__.keys())
        assert set(gobs.keys()) == expected_keys, f"Missing keys: {expected_keys - set(gobs.keys())}"

    def test_shapes_consistent(self):
        from conceptgraph.stages.paths import make_empty_gobs

        n = 3
        gobs = make_empty_gobs(n, feat_dim=256)

        assert gobs["xyxy"].shape == (n, 4)
        assert gobs["confidence"].shape == (n,)
        assert gobs["class_id"].shape == (n,)
        assert gobs["image_feats"].shape == (n, 256)
        assert gobs["text_feats"].shape == (n, 256)
        assert len(gobs["detection_class_labels"]) == n
        assert len(gobs["labels"]) == n
        assert len(gobs["captions"]) == n

    def test_zero_detections(self):
        from conceptgraph.stages.paths import make_empty_gobs

        gobs = make_empty_gobs(0)
        assert gobs["xyxy"].shape == (0, 4)
        assert len(gobs["captions"]) == 0
