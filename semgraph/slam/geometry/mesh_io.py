"""
Mesh I/O — format-specific PLY loading and vertex/instance extraction.

Supports:
  - ``replica``: Replica's ``mesh_semantic.ply`` with per-face ``object_id``.
    Falls back from ``plyfile`` to ``trimesh`` if plyfile is not installed.
  - Generic PLY meshes with per-vertex instance labels accessible via a
    configurable ``label_key``.
"""

from __future__ import annotations

import numpy as np


class MeshLoadError(Exception):
    """Raised when PLY parsing fails for reasons other than missing files/labels."""


def load_instance_mesh(
    ply_path: str,
    mesh_format: str,
    label_key: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Read a PLY and return per-vertex geometry + instance labels.

    Parameters
    ----------
    ply_path : str
        Path to the ``.ply`` mesh file.
    mesh_format : str
        One of ``"replica"`` or ``"generic"``.
    label_key : str
        Vertex (or face, for Replica) attribute name holding instance IDs.

    Returns
    -------
    vertices : (N, 3) float64
    colors : (N, 3) float64 in [0, 1], or None if the mesh has no color
    instance_ids : (N,) int32

    Raises
    ------
    FileNotFoundError
        If *ply_path* does not exist.
    ValueError
        If *mesh_format* is unrecognized, or *label_key* is not present in the
        mesh data.
    MeshLoadError
        If the underlying PLY parser (plyfile / trimesh) fails.
    """
    import os

    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Mesh file not found: {ply_path}")

    supported_formats = ("replica", "generic")
    if mesh_format not in supported_formats:
        raise ValueError(
            f"Unrecognized mesh_format '{mesh_format}'. "
            f"Supported formats: {supported_formats}"
        )

    if mesh_format == "replica":
        return _load_replica_semantic_mesh(ply_path)

    return _load_generic_mesh(ply_path, label_key)


def _load_generic_mesh(
    ply_path: str, label_key: str
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Load a PLY with per-vertex instance labels via trimesh."""
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for gt_mesh mode. Install with: pip install trimesh"
        )

    try:
        mesh = trimesh.load(ply_path, process=False)
    except Exception as exc:
        raise MeshLoadError(f"trimesh failed to load '{ply_path}': {exc}") from exc

    vertices = np.asarray(mesh.vertices)
    colors = None
    if hasattr(mesh.visual, "vertex_colors"):
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0

    raw_props = (
        mesh.metadata.get("ply_raw", {}).get("vertex", {}).get("properties", {})
    )
    if label_key in raw_props:
        instance_ids = np.asarray(raw_props[label_key])
    elif hasattr(mesh, "vertex_attributes") and label_key in mesh.vertex_attributes:
        instance_ids = np.asarray(mesh.vertex_attributes[label_key])
    else:
        available = list(getattr(mesh, "vertex_attributes", {}).keys())
        available += list(raw_props.keys())
        raise ValueError(
            f"Cannot find instance label key '{label_key}' in mesh. "
            f"Available attributes: {sorted(set(available))}"
        )

    return vertices, colors, instance_ids


def _load_replica_semantic_mesh(
    ply_path: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Load Replica's ``mesh_semantic.ply`` which stores *object_id* per face.

    Replica encodes instance identity on faces, not vertices.  We convert to
    per-vertex labels by assigning each vertex the instance of its first face.

    Falls back from ``plyfile`` to ``trimesh`` if plyfile is unavailable.
    """
    try:
        return _load_replica_via_plyfile(ply_path)
    except ImportError:
        pass

    try:
        return _load_replica_via_trimesh(ply_path)
    except ImportError:
        raise ImportError(
            "Either plyfile or trimesh is required for Replica mesh loading."
        )


def _load_replica_via_plyfile(
    ply_path: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Primary path: use plyfile for Replica meshes."""
    from plyfile import PlyData

    try:
        ply = PlyData.read(ply_path)
    except Exception as exc:
        raise MeshLoadError(
            f"plyfile failed to load '{ply_path}': {exc}"
        ) from exc

    vdata = ply["vertex"]
    vertices = np.column_stack([vdata["x"], vdata["y"], vdata["z"]])
    colors = None
    if "red" in vdata.data.dtype.names:
        colors = (
            np.column_stack([vdata["red"], vdata["green"], vdata["blue"]]) / 255.0
        )

    fdata = ply["face"]
    if "object_id" in fdata.data.dtype.names:
        face_ids = np.asarray(fdata["object_id"])
    else:
        face_ids = np.zeros(len(fdata.data), dtype=np.int32)

    faces = np.vstack(fdata["vertex_indices"])
    vertex_ids = _face_ids_to_vertex_ids(faces, face_ids, len(vertices))
    return vertices, colors, vertex_ids


def _load_replica_via_trimesh(
    ply_path: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Fallback path: use trimesh for Replica meshes when plyfile is absent."""
    import trimesh

    try:
        mesh = trimesh.load(ply_path, process=False)
    except Exception as exc:
        raise MeshLoadError(
            f"trimesh failed to load '{ply_path}': {exc}"
        ) from exc

    vertices = np.asarray(mesh.vertices)
    colors = None
    if hasattr(mesh.visual, "vertex_colors"):
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0

    if hasattr(mesh, "face_attributes") and "object_id" in mesh.face_attributes:
        face_ids = np.asarray(mesh.face_attributes["object_id"])
    elif hasattr(mesh.metadata, "get"):
        raw = mesh.metadata.get("ply_raw", {})
        face_props = raw.get("face", {}).get("properties", {})
        if "object_id" in face_props:
            face_ids = np.asarray(face_props["object_id"])
        else:
            face_ids = np.zeros(len(mesh.faces), dtype=np.int32)
    else:
        face_ids = np.zeros(len(mesh.faces), dtype=np.int32)

    faces = np.asarray(mesh.faces)
    vertex_ids = _face_ids_to_vertex_ids(faces, face_ids, len(vertices))
    return vertices, colors, vertex_ids


def _face_ids_to_vertex_ids(
    faces: np.ndarray, face_ids: np.ndarray, n_vertices: int
) -> np.ndarray:
    """Assign each vertex the instance ID of its first incident face."""
    vertex_ids = np.full(n_vertices, -1, dtype=np.int32)
    for fi in range(len(faces)):
        for vi in faces[fi]:
            if vertex_ids[vi] == -1:
                vertex_ids[vi] = face_ids[fi]
    vertex_ids[vertex_ids == -1] = 0
    return vertex_ids
