"""
Stage B2 — Per-object VLM captioning.

Phase B stage: loads per_view_records metadata from the oracle scene (object
IDs, crop image paths, n_points for view ranking). Does NOT load point clouds
or geometry.

For each object: sorts views by n_points descending, takes top K, sends 1.5x
crop images to VLM with three prompts (caption, color, material), runs LLM
consolidation to produce canonical_tag, candidate_tags, summary.

Saves as variant keyed by VLM name.

Standalone usage::

    python -m conceptgraph.stages.caption <hydra overrides> caption.vlm_name=...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VLM client
# ---------------------------------------------------------------------------

def init_vlm_client(cfg: Any) -> Any | None:
    """Initialize VLM API client."""
    from conceptgraph.utils.vlms.vlm_api import VLMAPIClient, wait_for_server

    caption_cfg = cfg.get("caption", {}) if hasattr(cfg, "get") else {}
    vlm_name = caption_cfg.get("vlm_name") or cfg.get("vlm_model_name", "Qwen/Qwen3-VL-2B-Instruct")
    vlm_api_url = cfg.get("vlm_api_url", "http://localhost:8000/v1")
    prompts = cfg.get("prompts_standard", None) or cfg.get("prompts_compact", None)

    try:
        wait_for_server(vlm_api_url, timeout=30)
    except Exception as exc:
        logger.error("VLM server not reachable at %s: %s", vlm_api_url, exc)
        return None

    return VLMAPIClient(
        base_url=vlm_api_url,
        model_name=vlm_name,
        prompts=prompts,
    )


# ---------------------------------------------------------------------------
# Per-object captioning
# ---------------------------------------------------------------------------

CAPTION_PROMPT = "Describe this object in one sentence. What is it?"
COLOR_PROMPT = "What is the primary color of this object? Answer with one or two words."
MATERIAL_PROMPT = "What material is this object made of? Answer with one or two words."

CONSOLIDATION_PROMPT = """You are given multiple captions describing the same object from different viewpoints.
Produce:
1. "canonical_tag": a single noun phrase identifying the object (e.g. "office chair", "wooden desk")
2. "candidate_tags": 3-5 alternative noun phrases that could also describe this object
3. "summary": a one-sentence description combining all observations

Captions:
{captions}

Respond in JSON format:
{{"canonical_tag": "...", "candidate_tags": ["...", ...], "summary": "..."}}"""


def _caption_object(
    per_view_records: list[dict],
    vlm_client: Any,
    top_k: int = 10,
) -> dict:
    """Caption a single object from its top-K views. Returns caption dict."""
    sorted_views = sorted(per_view_records, key=lambda r: r.get("n_points", 0), reverse=True)
    selected = sorted_views[:top_k]

    captions = []
    colors = []
    materials = []

    for record in selected:
        crop_path = record.get("crop_path", "")
        if not crop_path or not Path(crop_path).is_file():
            continue

        try:
            crop_img = Image.open(crop_path).convert("RGB")
        except Exception:
            continue

        try:
            caption = vlm_client.generate(image=crop_img, prompt=CAPTION_PROMPT)
            if caption:
                captions.append(caption.strip())
        except Exception as exc:
            logger.debug("Caption failed: %s", exc)

        try:
            color = vlm_client.generate(image=crop_img, prompt=COLOR_PROMPT)
            if color:
                colors.append(color.strip())
        except Exception:
            pass

        try:
            material = vlm_client.generate(image=crop_img, prompt=MATERIAL_PROMPT)
            if material:
                materials.append(material.strip())
        except Exception:
            pass

    # Consolidation
    result = {
        "canonical_tag": "unknown",
        "candidate_tags": [],
        "summary": "",
        "color": _most_common(colors) if colors else "",
        "material": _most_common(materials) if materials else "",
        "raw_captions": captions,
    }

    if captions and vlm_client is not None:
        consolidation_input = CONSOLIDATION_PROMPT.format(captions="\n".join(f"- {c}" for c in captions))
        try:
            resp = vlm_client.generate(prompt=consolidation_input)
            if resp:
                parsed = _parse_consolidation(resp)
                result.update(parsed)
        except Exception as exc:
            logger.warning("Consolidation failed: %s", exc)
            result["canonical_tag"] = captions[0][:50] if captions else "unknown"
            result["summary"] = captions[0] if captions else ""

    return result


def _most_common(items: list[str]) -> str:
    """Return the most common string from a list."""
    from collections import Counter
    if not items:
        return ""
    return Counter(items).most_common(1)[0][0]


def _parse_consolidation(resp: str) -> dict:
    """Best-effort parse of consolidation JSON response."""
    import json
    import re

    json_match = re.search(r'\{[^}]+\}', resp, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "canonical_tag": data.get("canonical_tag", "unknown"),
                "candidate_tags": data.get("candidate_tags", []),
                "summary": data.get("summary", ""),
            }
        except json.JSONDecodeError:
            pass

    return {"canonical_tag": "unknown", "candidate_tags": [], "summary": resp.strip()[:200]}


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main_standalone(cfg):
    """Phase B caption stage — reads oracle scene, captions all objects."""
    from conceptgraph.stages.paths import stage_paths
    from conceptgraph.stages import io as stage_io
    from conceptgraph.slam.utils import process_cfg
    from tqdm import tqdm

    cfg = process_cfg(cfg)
    paths = stage_paths(cfg)
    paths["variants"].mkdir(parents=True, exist_ok=True)

    oracle = stage_io.load_oracle_scene(paths["oracle_scene"])
    if oracle is None:
        raise FileNotFoundError(f"oracle_scene not found at {paths['oracle_scene']}")

    objects = oracle["objects"]
    caption_cfg = cfg.get("caption", {}) if hasattr(cfg, "get") else {}
    vlm_name = caption_cfg.get("vlm_name") or cfg.get("vlm_model_name", "Qwen/Qwen3-VL-2B-Instruct")
    top_k = caption_cfg.get("top_k", 10) if isinstance(caption_cfg, dict) else 10

    print(f"[caption] Phase B: vlm={vlm_name}, top_k={top_k}, {len(objects)} objects")

    vlm_client = init_vlm_client(cfg)
    if vlm_client is None:
        print("[caption] VLM client not available. Exiting.")
        return

    captions_data = {}
    for obj_idx, obj in enumerate(tqdm(list(objects), desc="caption")):
        pvr = obj.get("per_view_records", [])
        result = _caption_object(pvr, vlm_client, top_k=top_k)
        captions_data[obj_idx] = result

    safe_vlm = vlm_name.replace("/", "_")
    variant_data = {
        "vlm": vlm_name,
        "captions": captions_data,
    }
    stage_io.save_variant(paths["variants"], "__caption_only__", safe_vlm, variant_data)
    print(f"[caption] Done. Saved caption variant for vlm={vlm_name}")

    if vlm_client is not None:
        try:
            vlm_client.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
