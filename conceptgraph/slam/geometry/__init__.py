"""
Geometry backend abstraction for the staged pipeline.

Provides three backends that control how 2D detections are lifted to 3D:
  - trajectory: RGBD depth unprojection (current default)
  - gt_mesh:    Ground-truth mesh vertex lookup
  - sparse:     DUSt3R RGB-only point map lifting
"""

from conceptgraph.slam.geometry.base import GeometryBackend, FrameContext


def get_geometry_backend(mode: str) -> GeometryBackend:
    """Factory that returns the appropriate GeometryBackend for *mode*."""
    if mode == "trajectory":
        from conceptgraph.slam.geometry.trajectory import TrajectoryBackend
        return TrajectoryBackend()
    elif mode == "gt_mesh":
        from conceptgraph.slam.geometry.gt_mesh import GTMeshBackend
        return GTMeshBackend()
    elif mode == "sparse":
        from conceptgraph.slam.geometry.sparse import SparseBackend
        return SparseBackend()
    else:
        raise ValueError(
            f"Unknown pipeline_mode '{mode}'. "
            f"Valid options: 'trajectory', 'gt_mesh', 'sparse'"
        )


__all__ = ["GeometryBackend", "FrameContext", "get_geometry_backend"]
