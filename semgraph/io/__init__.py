"""
semgraph.io — data layer: records, serializers, and save/load glue.

Stage scripts import from here::

    from semgraph.io import save_frame_data, load_oracle_scene, FrameDataRecord
"""

from semgraph.io.records import (
    CaptionsRecord,
    FrameDataRecord,
    OracleSceneRecord,
    RawDetRecord,
    VariantRecord,
)
from semgraph.io.loaders import (
    deserialize_detection,
    list_frame_indices,
    load_captions,
    load_frame_data,
    load_map,
    load_oracle_scene,
    load_raw_det,
    load_variant,
    save_captions,
    save_frame_data,
    save_map,
    save_oracle_scene,
    save_raw_det,
    save_variant,
    serialize_detection,
    write_scene_graph_json,
)
from semgraph.io.serializers import NpzSerializer

__all__ = [
    # records
    "CaptionsRecord",
    "FrameDataRecord",
    "OracleSceneRecord",
    "RawDetRecord",
    "VariantRecord",
    # loaders
    "deserialize_detection",
    "list_frame_indices",
    "load_captions",
    "load_frame_data",
    "load_map",
    "load_oracle_scene",
    "load_raw_det",
    "load_variant",
    "save_captions",
    "save_frame_data",
    "save_map",
    "save_oracle_scene",
    "save_raw_det",
    "save_variant",
    "serialize_detection",
    "write_scene_graph_json",
    # serializer
    "NpzSerializer",
]
