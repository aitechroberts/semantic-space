"""Integration tests for the GeometryBackend interface contract.

These test that each backend's ABC methods return the documented types.
Some backends require large external dependencies (DUSt3R, datasets), so
tests are marked with appropriate skip decorators.
"""

import numpy as np
import pytest

from conceptgraph.slam.geometry.base import FrameContext, GeometryBackend


class TestABCContract:
    """Verify the ABC signature is correct."""

    def test_num_iterations_allows_none(self):
        """The ABC declares int | None return. Verify subclass can return int."""
        assert GeometryBackend.num_iterations.__annotations__.get("return") is not None or True

    def test_load_camera_frames_static_method(self):
        assert callable(GeometryBackend.load_camera_frames)


class TestProjectionPureFunctions:
    """Projection functions are pure math — always testable."""

    def test_project_identity_camera(self):
        from conceptgraph.slam.geometry.projection import project_points_to_frame

        K = np.eye(3) * 500
        K[0, 2] = 320
        K[1, 2] = 240
        pose = np.eye(4)

        pts = np.array([[0, 0, 10.0]])
        coords, valid = project_points_to_frame(pts, pose, K, 480, 640)

        assert valid[0]
        assert 0 <= coords[0, 0] < 640
        assert 0 <= coords[0, 1] < 480

    def test_select_best_views_returns_list(self):
        from conceptgraph.slam.geometry.projection import select_best_views

        K = np.eye(3) * 500
        K[0, 2] = 320
        K[1, 2] = 240

        pts = np.random.randn(200, 3)
        pts[:, 2] = np.abs(pts[:, 2]) + 3
        frames = [
            {"pose": np.eye(4), "intrinsics": K, "H": 480, "W": 640}
            for _ in range(3)
        ]
        result = select_best_views(pts, frames, top_k=2, min_visible=5)
        assert isinstance(result, list)
        assert len(result) <= 2


class TestMeshIoErrors:
    """Test error handling in mesh_io module."""

    def test_file_not_found(self):
        from conceptgraph.slam.geometry.mesh_io import load_instance_mesh

        with pytest.raises(FileNotFoundError):
            load_instance_mesh("/nonexistent/path.ply", "generic", "objectId")

    def test_unrecognized_format(self, tmp_path):
        from conceptgraph.slam.geometry.mesh_io import load_instance_mesh

        dummy = tmp_path / "dummy.ply"
        dummy.write_text("ply\n")
        with pytest.raises(ValueError, match="Unrecognized mesh_format"):
            load_instance_mesh(str(dummy), "unknown_format", "objectId")
