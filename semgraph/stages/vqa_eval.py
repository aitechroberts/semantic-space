"""VQA evaluation for Space3D-Bench (Phase 5).

Four retrieval / reasoning methods:

  1. ``clip_retrieval``          — retrieval-only. Encode the full
     question text with the encoder's text tower, rank objects by
     cosine against the chosen image-tower feature (default
     ``clip_ft_weighted_avg``). Emits the top-1 object's
     ``object_tag`` as the answer. No attribute heuristics.
  2. ``full_sg``                 — feed the entire VLM-built scene
     graph JSON to a VLM via the Sparse3DPR-inspired prompts.
  3. ``task_subgraph_mst_vlm``   — Sparse3DPR-inspired subgraph: embed
     the question with a sentence encoder, seed the ``SEED_K`` nearest
     nodes by caption embedding at ``tau=0.07``, expand 1-hop and
     2-hop along the union of MST + VLM edges, feed the subgraph to
     the VLM with the same prompts as ``full_sg``.
  4. ``task_subgraph_flat_no_planes`` — same as (3) but plane nodes and
     their incident edges are dropped before prompt construction.

Scoring delegates to ``generate_groundtruth/_space3d_layout.py`` which
handles v0.0.2's answer-format taxonomy (binary / object_list /
qualitative are deterministically scored; position / distance / count
are tagged ``format_unsupported`` and excluded from the accuracy
denominator but retained in ``per_question.json``).

Accuracy is reported for four strata:
  * ``all``               — every scored question
  * ``novel``             — ``!language_only_correct`` (from vetted_questions.json)
  * ``by_type``           — {binary, object_list, qualitative}
  * ``by_negation``       — ``with_negation`` / ``without_negation``
  * ``format_unsupported`` — n only (no accuracy; denominator-excluded)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_GGROOT = _REPO_ROOT / "generate_groundtruth"
if str(_GGROOT) not in sys.path:
    sys.path.insert(0, str(_GGROOT))

from _space3d_layout import (  # noqa: E402
    SCENE_MAP,
    TYPE_BINARY,
    TYPE_OBJECT_LIST,
    TYPE_QUALITATIVE,
    ground_truth_path,
    has_scene,
    infer_type,
    is_binary,
    is_supported,
    load_scene_qa,
    normalize_gt_entry,
    scene_dir,
    score_prediction,
)
from semgraph.utils.io_atomic import atomic_write_json, touch_done  # noqa: E402

logger = logging.getLogger("vqa_eval")

DEFAULT_TAU = 0.07
DEFAULT_SEED_K = 5

NEGATION_TOKENS = (
    "not ", " no ", "never", "without", "except",
    "other than", "besides", "isn't", "aren't",
    "doesn't", "don't", "cannot", "can't",
)


def _has_negation(question: str) -> bool:
    q = " " + (question or "").lower() + " "
    return any(tok in q for tok in NEGATION_TOKENS)


# =============================================================================
# Encoder pass: clip_retrieval (retrieval-only; no attribute heuristics)
# =============================================================================

def _load_variant(variants_dir: Path, safe_enc: str) -> Any | None:
    from semgraph.io import load_variant
    return load_variant(variants_dir, f"embed_{safe_enc}")


def _encode_query(encoder_type: str, encoder_name: str, texts: list[str],
                  device: str = "cuda") -> np.ndarray | None:
    from semgraph.encoding import get_encoder
    enc = get_encoder(encoder_type, encoder_name, device=device)
    feats = enc.encode_texts(texts)
    try:
        del enc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return feats


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return an @ bn.T


def run_clip_retrieval(
    scene_graph: dict,
    variant: Any,
    questions: dict[str, str],
    encoder_type: str,
    encoder_name: str,
    feature_key: str = "clip_ft_weighted_avg",
    device: str = "cuda",
) -> dict[str, dict]:
    """Rank objects by cosine similarity to the full question; emit top-1 tag."""
    obj_feats = getattr(variant, feature_key, None)
    if obj_feats is None or len(obj_feats) == 0:
        logger.warning("variant.%s empty — skipping clip_retrieval", feature_key)
        return {qid: {
            "answer": "", "method": "clip_retrieval", "score": 0.0,
            "error": "empty_variant",
        } for qid in questions}

    obj_feats = np.asarray(obj_feats, dtype=np.float32)
    if obj_feats.ndim == 1:
        obj_feats = obj_feats.reshape(1, -1)

    qids = list(questions.keys())
    qtexts = [str(questions[q]) for q in qids]
    q_feats = _encode_query(encoder_type, encoder_name, qtexts, device=device)
    if q_feats is None or q_feats.size == 0:
        return {qid: {"answer": "", "method": "clip_retrieval", "score": 0.0} for qid in qids}
    q_feats = np.asarray(q_feats, dtype=np.float32)
    if q_feats.shape[1] != obj_feats.shape[1]:
        logger.error("dim mismatch: q=%s obj=%s", q_feats.shape, obj_feats.shape)
        return {qid: {"answer": "", "method": "clip_retrieval", "score": 0.0} for qid in qids}

    sims = _cosine_matrix(q_feats, obj_feats)
    objects = scene_graph.get("objects", [])
    out: dict[str, dict] = {}
    for i, qid in enumerate(qids):
        row = sims[i]
        top_idx = int(np.argmax(row))
        top_score = float(row[top_idx])
        obj = objects[top_idx] if top_idx < len(objects) else {}
        tag = str(obj.get("object_tag", ""))
        out[qid] = {
            "answer": tag,
            "method": "clip_retrieval",
            "score": top_score,
            "top_idx": top_idx,
            "feature_key": feature_key,
        }
    return out


# =============================================================================
# VLM pass: full_sg
# =============================================================================

def _classify_prompt(qtype: str, prompts: dict[str, str]) -> str:
    if qtype == TYPE_BINARY:
        return prompts.get("yesno") or prompts.get("descriptive", "")
    if qtype == TYPE_OBJECT_LIST:
        # list answers are best served by the descriptive template
        return prompts.get("descriptive", "")
    # count (format_unsupported for scoring but we still prompt) → counting template
    if qtype == "count":
        return prompts.get("counting") or prompts.get("descriptive", "")
    return prompts.get("descriptive", "")


def _call_vlm_text(client, model_name: str, prompt: str,
                   max_tokens: int = 128, temperature: float = 0.1) -> str:
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=120.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("vlm call failed: %s", exc)
        return ""


def _compact_scene_graph(sg: dict, max_nodes: int = 256) -> dict:
    objs = sg.get("objects", [])[:max_nodes]
    compact_objs = [
        {
            "id": o.get("id"),
            "tag": o.get("object_tag"),
            "caption": (o.get("caption") or "")[:200],
            "color": o.get("color"),
            "material": o.get("material"),
            "center": [round(float(x), 2) for x in (o.get("bbox_center") or [0, 0, 0])],
            "extent": [round(float(x), 2) for x in (o.get("bbox_extent") or [0, 0, 0])],
            "parent_plane_id": o.get("parent_plane_id"),
        }
        for o in objs
    ]
    return {
        "scene_type": sg.get("scene_type"),
        "objects": compact_objs,
        "planes": sg.get("planes", []),
        "edges": sg.get("edges", []),
    }


def run_full_sg(
    scene_graph: dict,
    questions: dict[str, str],
    gt_norm: dict[str, dict],
    client,
    vlm_model: str,
    prompts: dict[str, str],
) -> dict[str, dict]:
    compact = _compact_scene_graph(scene_graph)
    sg_json = json.dumps(compact)
    out: dict[str, dict] = {}
    for qid, q in questions.items():
        norm = gt_norm.get(qid, {"answer_text": ""})
        qtype = infer_type(norm)
        template = _classify_prompt(qtype, prompts)
        prompt = template.replace("{subgraph}", sg_json).replace("{question}", str(q))
        resp = _call_vlm_text(client, vlm_model, prompt)
        out[qid] = {"answer": resp, "method": "full_sg"}
    return out


# =============================================================================
# VLM pass: task_subgraph_mst_vlm / _flat_no_planes
# =============================================================================

def _load_sentence_encoder(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def _extract_task_subgraph(
    scene_graph: dict,
    question: str,
    sentence_encoder,
    seed_k: int = DEFAULT_SEED_K,
    tau: float = DEFAULT_TAU,
    drop_planes: bool = False,
) -> dict:
    """Seed-K + 2-hop expansion over MST ∪ VLM edges."""
    objects = scene_graph.get("objects", [])
    if not objects:
        return {"scene_type": scene_graph.get("scene_type"),
                "objects": [], "planes": [], "edges": []}

    caps = [
        (o.get("object_tag", "") + ". " + (o.get("caption") or ""))[:300]
        for o in objects
    ]
    obj_embs = sentence_encoder.encode(caps, convert_to_numpy=True, normalize_embeddings=True)
    q_emb = sentence_encoder.encode([question], convert_to_numpy=True,
                                    normalize_embeddings=True)[0]
    sims = obj_embs @ q_emb
    # Temperature tau is emitted for diagnostics; selection is top-k.
    order = np.argsort(-sims)
    seed_ids: set[int] = set()
    for idx in order[:seed_k]:
        seed_ids.add(int(objects[int(idx)].get("id", int(idx))))

    edges = scene_graph.get("edges", [])
    adj: dict[int, set[int]] = defaultdict(set)
    plane_ids = {int(p.get("plane_id", -1)) for p in scene_graph.get("planes", [])}
    for e in edges:
        s = int(e.get("source", -1))
        t = int(e.get("target", -1))
        if drop_planes and (s in plane_ids or t in plane_ids):
            continue
        if s >= 0 and t >= 0:
            adj[s].add(t)
            adj[t].add(s)

    keep: set[int] = set(seed_ids)
    frontier = set(seed_ids)
    for _ in range(2):  # 2-hop
        nxt: set[int] = set()
        for n in frontier:
            nxt |= adj.get(n, set())
        nxt -= keep
        keep |= nxt
        frontier = nxt
        if not frontier:
            break

    keep_objs = [o for o in objects if int(o.get("id", -1)) in keep]
    keep_edges = [
        e for e in edges
        if int(e.get("source", -1)) in keep and int(e.get("target", -1)) in keep
    ]

    planes = scene_graph.get("planes", [])
    if drop_planes:
        planes = []
        keep_edges = [
            e for e in keep_edges
            if int(e.get("source", -1)) not in plane_ids
            and int(e.get("target", -1)) not in plane_ids
        ]

    return {
        "scene_type": scene_graph.get("scene_type"),
        "objects": keep_objs,
        "planes": planes,
        "edges": keep_edges,
        "_tau": tau,
    }


def run_task_subgraph(
    scene_graph: dict,
    questions: dict[str, str],
    gt_norm: dict[str, dict],
    client,
    vlm_model: str,
    prompts: dict[str, str],
    sentence_model: str,
    device: str,
    drop_planes: bool,
    seed_k: int,
    tau: float,
    method_name: str,
) -> dict[str, dict]:
    encoder = _load_sentence_encoder(sentence_model, device=device)
    out: dict[str, dict] = {}
    for qid, q in questions.items():
        sub = _extract_task_subgraph(
            scene_graph, str(q), encoder,
            seed_k=seed_k, tau=tau, drop_planes=drop_planes,
        )
        compact = _compact_scene_graph(sub, max_nodes=64)
        norm = gt_norm.get(qid, {"answer_text": ""})
        qtype = infer_type(norm)
        template = _classify_prompt(qtype, prompts)
        prompt = (
            template
            .replace("{subgraph}", json.dumps(compact))
            .replace("{question}", str(q))
        )
        resp = _call_vlm_text(client, vlm_model, prompt)
        out[qid] = {
            "answer": resp,
            "method": method_name,
            "subgraph_size": len(compact["objects"]),
            "n_edges": len(compact["edges"]),
        }
    return out


# =============================================================================
# Accuracy / summary
# =============================================================================

def score_predictions(
    predictions: dict[str, dict],
    gt_norm: dict[str, dict],
    questions: dict[str, str],
    vetted: dict[str, Any] | None,
    method_name: str,
) -> dict:
    """Stratified accuracy for one method.

    Denominators:
      * ``all`` / ``novel`` / ``by_type`` / ``by_negation`` count only
        questions whose format is in ``SUPPORTED_TYPES``.
      * ``format_unsupported`` is a side-bucket reported as a count only
        (no accuracy).
    """
    n_scored = 0
    n_correct = 0
    novel_total = 0
    novel_correct = 0
    by_type: dict[str, dict[str, int]] = {}
    by_neg: dict[str, dict[str, int]] = {
        "with_negation": {"n": 0, "correct": 0},
        "without_negation": {"n": 0, "correct": 0},
    }
    unsupported_by_type: dict[str, int] = {}

    per_q: dict[str, dict] = {}
    for qid, q in questions.items():
        norm = gt_norm.get(qid, {"answer_text": "", "example_answer": None,
                                 "prompt": "", "image_path": None,
                                 "answer_raw": ""})
        qtype = infer_type(norm)
        pred = predictions.get(qid, {}).get("answer", "")
        score = score_prediction(pred, norm, qtype)

        neg = _has_negation(str(q))
        lang_only = False
        if vetted and qid in vetted.get("per_question", {}):
            lang_only = bool(
                vetted["per_question"][qid].get("language_only_correct", False)
            )

        per_q[qid] = {
            "predicted": pred,
            "gt_answer": norm["answer_text"],
            "gt_raw": norm["answer_raw"],
            "type": qtype,
            "scored": score["scored"],
            "correct": score["correct"],
            "reason": score["reason"],
            "has_negation": neg,
            "language_only_correct": lang_only,
        }

        if not score["scored"]:
            unsupported_by_type[qtype] = unsupported_by_type.get(qtype, 0) + 1
            continue

        ok = score["correct"]
        n_scored += 1
        n_correct += int(ok)
        if not lang_only:
            novel_total += 1
            novel_correct += int(ok)

        b = by_type.setdefault(qtype, {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(ok)
        key = "with_negation" if neg else "without_negation"
        by_neg[key]["n"] += 1
        by_neg[key]["correct"] += int(ok)

    def _acc(c: int, n: int) -> float:
        return round(c / n, 4) if n else 0.0

    return {
        "method": method_name,
        "per_question": per_q,
        "summary": {
            "all":   {"n": n_scored, "correct": n_correct,
                      "accuracy": _acc(n_correct, n_scored)},
            "novel": {"n": novel_total, "correct": novel_correct,
                      "accuracy": _acc(novel_correct, novel_total)},
            "by_type": {
                k: {"n": v["n"], "correct": v["correct"],
                    "accuracy": _acc(v["correct"], v["n"])}
                for k, v in by_type.items()
            },
            "by_negation": {
                k: {"n": v["n"], "correct": v["correct"],
                    "accuracy": _acc(v["correct"], v["n"])}
                for k, v in by_neg.items()
            },
            "format_unsupported": {
                "total": sum(unsupported_by_type.values()),
                "by_type": unsupported_by_type,
            },
        },
    }


# =============================================================================
# Orchestration helpers (callable from CLI or wrapper scripts)
# =============================================================================

def _load_scene_inputs(
    space3d_root: Path, scene_out: str,
) -> tuple[dict[str, str], dict[str, dict], dict | None]:
    """Return ``(questions, gt_norm, vetted)`` for one scene.

    ``gt_norm`` is ``dict[qid -> normalized_gt]`` per ``_space3d_layout``.
    """
    if not has_scene(space3d_root, scene_out):
        raise FileNotFoundError(
            f"Space3D-Bench scene data missing at {scene_dir(space3d_root, scene_out)}"
        )
    questions_raw, gt_raw = load_scene_qa(space3d_root, scene_out)
    # force string qids + string question text
    questions = {str(k): str(v) for k, v in questions_raw.items()}
    gt_norm = {str(k): normalize_gt_entry(v) for k, v in gt_raw.items()}
    vetted = None
    v_path = scene_dir(space3d_root, scene_out) / "vetted_questions.json"
    if v_path.is_file():
        vetted = json.loads(v_path.read_text())
    return questions, gt_norm, vetted


def _load_scene_graph(output_root: Path, scene: str, safe_vlm: str) -> dict | None:
    path = output_root / scene / "stages" / "scene_graphs" / safe_vlm / "scene_graph.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _load_prompts_bundle(prompt_dir: Path) -> dict[str, str]:
    bundle = {}
    for name in ("yesno", "counting", "descriptive"):
        p = prompt_dir / f"sparse3dpr_{name}.txt"
        if p.is_file():
            bundle[name] = p.read_text()
    return bundle


def run_encoder_pass(args) -> int:
    output_root = Path(args.output_root)
    space3d_root = Path(args.space3d_root)
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    for scene in scenes:
        questions, gt_norm, vetted = _load_scene_inputs(space3d_root, scene)

        safe_vlm = args.scene_graph_vlm.replace("/", "_") if args.scene_graph_vlm else None
        sg = None
        if safe_vlm:
            sg = _load_scene_graph(output_root, scene, safe_vlm)
        if sg is None:
            sg_root = output_root / scene / "stages" / "scene_graphs"
            if sg_root.is_dir():
                for sub in sorted(sg_root.iterdir()):
                    if (sub / "scene_graph.json").is_file():
                        sg = json.loads((sub / "scene_graph.json").read_text())
                        safe_vlm = sub.name
                        break
        if sg is None:
            logger.error("[%s] no scene_graph.json available", scene)
            continue

        for enc_spec in args.encoders.split(","):
            enc_spec = enc_spec.strip()
            if not enc_spec:
                continue
            enc_type, enc_name = enc_spec.split("|", 1)
            safe_enc = enc_name.replace("/", "_")

            variant = _load_variant(
                output_root / scene / "stages" / "variants", safe_enc
            )
            if variant is None:
                logger.warning("[%s][%s] variant missing — skip", scene, safe_enc)
                continue

            preds = run_clip_retrieval(
                scene_graph=sg,
                variant=variant,
                questions=questions,
                encoder_type=enc_type,
                encoder_name=enc_name,
                feature_key=args.feature_key,
                device=args.device,
            )
            summary = score_predictions(
                preds, gt_norm, questions, vetted,
                method_name="clip_retrieval",
            )
            out_dir = (
                output_root / scene / "stages" / "eval"
                / f"{safe_enc}__sg_{safe_vlm}" / "clip_retrieval"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(out_dir / "predictions.json", preds)
            atomic_write_json(out_dir / "summary.json", summary["summary"])
            atomic_write_json(out_dir / "per_question.json", summary["per_question"])
            touch_done(out_dir / ".done")
            logger.info(
                "[%s][%s] clip_retrieval all=%.3f novel=%.3f unsup=%d",
                scene, safe_enc,
                summary["summary"]["all"]["accuracy"],
                summary["summary"]["novel"]["accuracy"],
                summary["summary"]["format_unsupported"]["total"],
            )
    return 0


def run_vlm_pass(args) -> int:
    from openai import OpenAI
    client = OpenAI(base_url=args.api_url, api_key="not-needed")

    output_root = Path(args.output_root)
    space3d_root = Path(args.space3d_root)
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    prompts = _load_prompts_bundle(Path(args.prompt_dir))
    safe_vlm = args.vlm.replace("/", "_")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    for scene in scenes:
        questions, gt_norm, vetted = _load_scene_inputs(space3d_root, scene)
        sg = _load_scene_graph(output_root, scene, safe_vlm)
        if sg is None:
            logger.error("[%s][%s] no scene_graph.json — skip", scene, safe_vlm)
            continue

        base_eval_dir = output_root / scene / "stages" / "eval" / f"vlm_{safe_vlm}"

        if "full_sg" in methods:
            preds = run_full_sg(sg, questions, gt_norm, client, args.vlm, prompts)
            _emit(base_eval_dir / "full_sg", preds, gt_norm, questions, vetted, "full_sg")

        if "task_subgraph_mst_vlm" in methods:
            preds = run_task_subgraph(
                sg, questions, gt_norm, client, args.vlm, prompts,
                sentence_model=args.sentence_model, device=args.device,
                drop_planes=False, seed_k=args.seed_k, tau=args.tau,
                method_name="task_subgraph_mst_vlm",
            )
            _emit(base_eval_dir / "task_subgraph_mst_vlm", preds, gt_norm, questions, vetted,
                  "task_subgraph_mst_vlm")

        if "task_subgraph_flat_no_planes" in methods:
            preds = run_task_subgraph(
                sg, questions, gt_norm, client, args.vlm, prompts,
                sentence_model=args.sentence_model, device=args.device,
                drop_planes=True, seed_k=args.seed_k, tau=args.tau,
                method_name="task_subgraph_flat_no_planes",
            )
            _emit(base_eval_dir / "task_subgraph_flat_no_planes", preds, gt_norm, questions, vetted,
                  "task_subgraph_flat_no_planes")
    return 0


def _emit(out_dir: Path, preds, gt_norm, questions, vetted, method):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = score_predictions(preds, gt_norm, questions, vetted, method_name=method)
    atomic_write_json(out_dir / "predictions.json", preds)
    atomic_write_json(out_dir / "summary.json", summary["summary"])
    atomic_write_json(out_dir / "per_question.json", summary["per_question"])
    touch_done(out_dir / ".done")
    logger.info(
        "[%s] all=%.3f novel=%.3f unsup=%d",
        method,
        summary["summary"]["all"]["accuracy"],
        summary["summary"]["novel"]["accuracy"],
        summary["summary"]["format_unsupported"]["total"],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[vqa-eval] %(message)s")
    parser = argparse.ArgumentParser(description="Phase-5 VQA evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enc = sub.add_parser("encoder", help="run clip_retrieval for a set of encoders")
    p_enc.add_argument("--output_root", required=True)
    p_enc.add_argument("--space3d_root", required=True)
    p_enc.add_argument("--scenes", default=",".join(SCENE_MAP.keys()))
    p_enc.add_argument("--encoders", required=True,
                       help="comma-separated list of 'type|hf_id'")
    p_enc.add_argument("--scene_graph_vlm", default=None,
                       help="which VLM's scene_graph.json to use for object features/tags")
    p_enc.add_argument("--feature_key", default="clip_ft_weighted_avg")
    p_enc.add_argument("--device", default="cuda")

    p_vlm = sub.add_parser("vlm", help="run full_sg and task_subgraph_* for one VLM")
    p_vlm.add_argument("--output_root", required=True)
    p_vlm.add_argument("--space3d_root", required=True)
    p_vlm.add_argument("--scenes", default=",".join(SCENE_MAP.keys()))
    p_vlm.add_argument("--vlm", required=True)
    p_vlm.add_argument("--api_url", default="http://localhost:8000/v1")
    p_vlm.add_argument("--prompt_dir", default=str(_REPO_ROOT / "config" / "prompts"))
    p_vlm.add_argument(
        "--methods",
        default="full_sg,task_subgraph_mst_vlm,task_subgraph_flat_no_planes",
    )
    p_vlm.add_argument("--sentence_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_vlm.add_argument("--device", default="cuda")
    p_vlm.add_argument("--seed_k", type=int, default=DEFAULT_SEED_K)
    p_vlm.add_argument("--tau", type=float, default=DEFAULT_TAU)

    args = parser.parse_args()

    if args.cmd == "encoder":
        sys.exit(run_encoder_pass(args))
    elif args.cmd == "vlm":
        sys.exit(run_vlm_pass(args))


if __name__ == "__main__":
    main()
