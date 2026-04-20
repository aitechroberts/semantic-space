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

    python -m semgraph.stages.caption <hydra overrides> caption.vlm_name=...
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
    from semgraph.utils.vlms.vlm_api import VLMAPIClient, wait_for_server

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

DEFAULT_CAPTION_PROMPT = "Describe this object in one sentence. What is it?"
DEFAULT_COLOR_PROMPT = "What is the primary color of this object? Answer with one or two words."
DEFAULT_MATERIAL_PROMPT = "What material is this object made of? Answer with one or two words."

DEFAULT_CONSOLIDATION_PROMPT = """You are given multiple captions describing the same object from different viewpoints.
Produce:
1. "canonical_tag": a single noun phrase identifying the object (e.g. "office chair", "wooden desk")
2. "candidate_tags": 3-5 alternative noun phrases that could also describe this object
3. "summary": a one-sentence description combining all observations

Captions:
{captions}

Respond in JSON format:
{{"canonical_tag": "...", "candidate_tags": ["...", ...], "summary": "..."}}"""

# Back-compat aliases for callers that imported the old constant names.
CAPTION_PROMPT = DEFAULT_CAPTION_PROMPT
COLOR_PROMPT = DEFAULT_COLOR_PROMPT
MATERIAL_PROMPT = DEFAULT_MATERIAL_PROMPT
CONSOLIDATION_PROMPT = DEFAULT_CONSOLIDATION_PROMPT


def _resolve_prompts(cfg: Any) -> dict[str, str]:
    """Return the active prompt bundle, honoring an optional override bundle.

    Order of precedence (highest first):
      1. ``cfg.caption.prompts.{caption,color,material,consolidation}``
         (e.g. populated by +caption@caption=prompts_rich).
      2. Module-level defaults.

    This lets us sweep prompt variants without monkey-patching constants.
    """
    prompts = {
        "caption": DEFAULT_CAPTION_PROMPT,
        "color": DEFAULT_COLOR_PROMPT,
        "material": DEFAULT_MATERIAL_PROMPT,
        "consolidation": DEFAULT_CONSOLIDATION_PROMPT,
    }
    caption_cfg = cfg.get("caption", {}) if hasattr(cfg, "get") else {}
    override = caption_cfg.get("prompts") if isinstance(caption_cfg, dict) or hasattr(caption_cfg, "get") else None
    if override:
        for key in prompts:
            val = override.get(key) if hasattr(override, "get") else None
            if val:
                prompts[key] = str(val)
    return prompts


def _caption_object(
    per_view_records: list[dict],
    vlm_client: Any,
    top_k: int = 10,
    prompts: dict[str, str] | None = None,
) -> dict:
    """Caption a single object from its top-K views. Returns caption dict."""
    p = prompts or {}
    caption_prompt = p.get("caption", DEFAULT_CAPTION_PROMPT)
    color_prompt = p.get("color", DEFAULT_COLOR_PROMPT)
    material_prompt = p.get("material", DEFAULT_MATERIAL_PROMPT)
    consolidation_prompt = p.get("consolidation", DEFAULT_CONSOLIDATION_PROMPT)

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
            caption = vlm_client.generate(image=crop_img, prompt=caption_prompt)
            if caption:
                captions.append(caption.strip())
        except Exception as exc:
            logger.debug("Caption failed: %s", exc)

        try:
            color = vlm_client.generate(image=crop_img, prompt=color_prompt)
            if color:
                colors.append(color.strip())
        except Exception:
            pass

        try:
            material = vlm_client.generate(image=crop_img, prompt=material_prompt)
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
        consolidation_input = consolidation_prompt.format(captions="\n".join(f"- {c}" for c in captions))
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
    from semgraph.stages.paths import stage_paths
    from semgraph.io import load_oracle_scene, save_captions, CaptionsRecord
    from semgraph.slam.utils import process_cfg
    from tqdm import tqdm

    cfg = process_cfg(cfg)
    paths = stage_paths(cfg)

    oracle = load_oracle_scene(paths["oracle"])
    if oracle is None:
        raise FileNotFoundError(f"oracle_scene not found at {paths['oracle']}")

    n_objects = len(oracle.class_names)
    caption_cfg = cfg.get("caption", {}) if hasattr(cfg, "get") else {}
    vlm_name = caption_cfg.get("vlm_name") or cfg.get("vlm_model_name", "Qwen/Qwen3-VL-2B-Instruct")
    top_k = caption_cfg.get("top_k", 10) if isinstance(caption_cfg, dict) else 10

    prompts = _resolve_prompts(cfg)
    print(
        f"[caption] Phase B: vlm={vlm_name}, top_k={top_k}, {n_objects} objects, "
        f"prompt_bundle={'rich' if prompts['caption'] != DEFAULT_CAPTION_PROMPT else 'default'}"
    )

    vlm_client = init_vlm_client(cfg)
    if vlm_client is None:
        print("[caption] VLM client not available. Exiting.")
        return

    captions_data: dict[int, dict] = {}
    for obj_idx in tqdm(range(n_objects), desc="caption"):
        pv_meta = oracle.per_view_meta[obj_idx] if obj_idx < len(oracle.per_view_meta) else []
        pvr_dicts = [
            {"crop_path": pm.crop_path, "n_points": pm.n_points}
            for pm in pv_meta
        ]
        result = _caption_object(pvr_dicts, vlm_client, top_k=top_k, prompts=prompts)
        captions_data[obj_idx] = result

    safe_vlm = vlm_name.replace("/", "_")
    captions_dir = paths["captions"] / safe_vlm
    captions_dir.mkdir(parents=True, exist_ok=True)
    record = CaptionsRecord(entries=captions_data)
    save_captions(captions_dir, record)
    print(f"[caption] Done. Saved caption record for vlm={vlm_name}")

    if vlm_client is not None:
        try:
            vlm_client.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
