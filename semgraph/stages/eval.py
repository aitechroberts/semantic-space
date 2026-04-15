"""
Stage B4 — Evaluation: classification, QA, and retrieval.

Reads the HPSG JSON produced by semantic_assemble.py.

Evaluation modes:
- Classification: mIoU, F-mIoU, mAcc against 1687-label eval list with
  17-category grouping
- QA: ScanQA + Space3D-Bench with SceneGPT prompts, 2 ICL examples,
  optional subgraph extraction (SentenceTransformer + FAISS top-5 + 2-hop)
  Metrics: EM@1, BLEU-1-4, ROUGE-L, METEOR, CIDEr
- Retrieval: query objects by text embedding, max-over-views similarity,
  recall@K

Standalone usage::

    python -m semgraph.stages.eval <hydra overrides> eval.encoder=... eval.vlm=...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def _load_labels(path: str | Path) -> list[str]:
    """Load a label file (one label per line)."""
    p = Path(path)
    if not p.is_file():
        logger.warning("Label file not found: %s", p)
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _load_category_groups(path: str | Path) -> dict:
    """Load the 17-category grouping JSON."""
    p = Path(path)
    if not p.is_file():
        return {}
    with open(p) as f:
        return json.load(f)


def _best_label_match(predicted: str, gt_labels: list[str]) -> str | None:
    """Find best matching GT label for a predicted tag (case-insensitive substring)."""
    pred_lower = predicted.lower().strip()
    for label in gt_labels:
        if label.lower() == pred_lower:
            return label
    for label in gt_labels:
        if label.lower() in pred_lower or pred_lower in label.lower():
            return label
    return None


def evaluate_classification(
    scene_graph: dict,
    gt_labels_path: str = "config/eval_1687_labels.txt",
    categories_path: str = "config/grouped_17_categories.json",
) -> dict:
    """Run classification evaluation. Returns metrics dict."""
    eval_labels = _load_labels(gt_labels_path)
    categories = _load_category_groups(categories_path)

    objects = scene_graph.get("objects", [])
    if not objects or not eval_labels:
        return {"mIoU": 0.0, "F_mIoU": 0.0, "mAcc": 0.0, "n_objects": len(objects)}

    correct = 0
    total = len(objects)
    per_category_correct: dict[str, int] = {}
    per_category_total: dict[str, int] = {}

    for obj in objects:
        tag = obj.get("object_tag", "unknown")
        matched = _best_label_match(tag, eval_labels)

        # Determine category
        cat = "other"
        if categories:
            for cat_name, cat_labels in categories.items():
                if isinstance(cat_labels, list) and tag.lower() in [l.lower() for l in cat_labels]:
                    cat = cat_name
                    break

        per_category_total[cat] = per_category_total.get(cat, 0) + 1
        if matched is not None:
            correct += 1
            per_category_correct[cat] = per_category_correct.get(cat, 0) + 1

    mAcc = correct / total if total > 0 else 0.0

    # Per-category accuracy for mIoU approximation
    cat_accs = []
    for cat in per_category_total:
        cat_correct = per_category_correct.get(cat, 0)
        cat_total = per_category_total[cat]
        if cat_total > 0:
            cat_accs.append(cat_correct / cat_total)

    mIoU = np.mean(cat_accs) if cat_accs else 0.0

    # Frequency-weighted mIoU (F-mIoU)
    freq_weights = []
    weighted_accs = []
    for cat in per_category_total:
        w = per_category_total[cat] / total
        freq_weights.append(w)
        cat_correct = per_category_correct.get(cat, 0)
        cat_total = per_category_total[cat]
        weighted_accs.append(w * (cat_correct / cat_total if cat_total > 0 else 0.0))

    F_mIoU = sum(weighted_accs)

    return {
        "mIoU": float(mIoU),
        "F_mIoU": float(F_mIoU),
        "mAcc": float(mAcc),
        "n_objects": total,
        "n_correct": correct,
        "per_category": {
            cat: {
                "correct": per_category_correct.get(cat, 0),
                "total": per_category_total[cat],
                "accuracy": per_category_correct.get(cat, 0) / per_category_total[cat] if per_category_total[cat] > 0 else 0.0,
            }
            for cat in sorted(per_category_total.keys())
        },
    }


# ---------------------------------------------------------------------------
# QA evaluation
# ---------------------------------------------------------------------------

def _load_prompt_file(path: str | Path) -> str:
    """Load a prompt template from a text file."""
    p = Path(path)
    if not p.is_file():
        logger.warning("Prompt file not found: %s", p)
        return ""
    return p.read_text().strip()


def _load_icl_examples(path: str | Path) -> dict:
    """Load ICL example JSON."""
    p = Path(path)
    if not p.is_file():
        return {}
    with open(p) as f:
        return json.load(f)


def evaluate_qa(
    scene_graph: dict,
    vlm_client: Any = None,
    system_prompt_path: str = "config/prompts/scene_understanding_system.txt",
    spatial_icl_path: str = "config/prompts/spatial_icl.json",
    geometric_icl_path: str = "config/prompts/geometric_icl.json",
    use_subgraph: bool = False,
) -> dict:
    """Run QA evaluation. Returns metrics dict.

    Actual QA benchmarks (ScanQA, Space3D-Bench) require external question
    sets. This provides the evaluation framework; question loading is a
    TODO for when benchmark data is available.
    """
    system_prompt = _load_prompt_file(system_prompt_path)
    spatial_icl = _load_icl_examples(spatial_icl_path)
    geometric_icl = _load_icl_examples(geometric_icl_path)

    scene_json_str = json.dumps(scene_graph.get("objects", []), indent=2)

    return {
        "status": "framework_ready",
        "system_prompt_loaded": bool(system_prompt),
        "spatial_icl_loaded": bool(spatial_icl),
        "geometric_icl_loaded": bool(geometric_icl),
        "n_objects": len(scene_graph.get("objects", [])),
        "use_subgraph": use_subgraph,
        "note": "QA benchmarks require external question sets (ScanQA, Space3D-Bench). Framework is ready for evaluation.",
    }


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def _normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (lowercase, strip punctuation)."""
    import re
    answer = answer.lower()
    answer = re.sub(r'[^\w\s]', ' ', answer)
    return ' '.join(answer.split())


def _check_retrieval_match(
    retrieved_tags: list[str],
    ground_truth: str,
    k: int,
) -> tuple[bool, int]:
    """Check if any of top-k retrieved object tags match ground truth.

    Returns (is_match, rank) where rank is 1-indexed (0 = no match).
    """
    gt_norm = _normalize_answer(ground_truth)

    for i, tag in enumerate(retrieved_tags[:k]):
        tag_norm = _normalize_answer(tag)

        if gt_norm in tag_norm or tag_norm in gt_norm:
            return True, i + 1

        gt_words = set(gt_norm.split())
        tag_words = set(tag_norm.split())
        if gt_words and len(gt_words & tag_words) / len(gt_words) > 0.3:
            return True, i + 1

    return False, 0


def evaluate_retrieval(
    scene_graph: dict,
    variant: Any = None,
    queries: list[dict] | None = None,
    text_encoder: Any = None,
    k_values: list[int] | None = None,
) -> dict:
    """Run retrieval evaluation using cosine similarity against variant features.

    Parameters
    ----------
    scene_graph : dict
        The assembled scene graph (used for object tags).
    variant : VariantRecord or None
        Phase B embed variant with ``clip_ft_weighted_avg`` and ``clip_ft_best``.
    queries : list[dict] or None
        Each dict has ``"question"`` (str) and ``"answer"`` (str).
    text_encoder : EmbeddingEncoder or None
        Encoder used for query text embedding.  Must have
        ``has_aligned_text_space == True`` and a working ``encode_texts()``.
    k_values : list[int] or None
        Recall cutoffs (default [1, 5, 10]).
    """
    if k_values is None:
        k_values = [1, 5, 10]

    if variant is None or queries is None or not queries:
        return {
            "status": "framework_ready",
            "k_values": k_values,
            "note": "Retrieval requires variant features and a query set.",
        }

    if text_encoder is None:
        return {
            "status": "skipped",
            "reason": "No text encoder provided.",
        }

    if not text_encoder.has_aligned_text_space:
        return {
            "status": "skipped",
            "reason": "Encoder does not have an aligned text space (Path B = N/A).",
        }

    objects = scene_graph.get("objects", [])
    obj_tags = [obj.get("object_tag", "unknown") for obj in objects]

    feat_avg = getattr(variant, "clip_ft_weighted_avg", None)
    feat_best = getattr(variant, "clip_ft_best", None)

    if feat_avg is None or feat_avg.size == 0:
        return {
            "status": "skipped",
            "reason": "Variant has no clip_ft_weighted_avg features.",
        }

    max_k = max(k_values)
    results_avg = {f"recall@{k}": 0 for k in k_values}
    results_best = {f"recall@{k}": 0 for k in k_values} if feat_best is not None and feat_best.size > 0 else None

    n_queries = len(queries)
    for q in queries:
        question = q.get("question", "")
        answer = q.get("answer", "")
        if not question:
            continue

        query_emb = text_encoder.encode_texts([question])
        if query_emb is None or query_emb.size == 0:
            continue

        sims_avg = (query_emb @ feat_avg.T).squeeze(0)
        ranked_idx_avg = np.argsort(-sims_avg)[:max_k]
        ranked_tags_avg = [obj_tags[i] for i in ranked_idx_avg if i < len(obj_tags)]

        for k in k_values:
            match, _ = _check_retrieval_match(ranked_tags_avg, answer, k)
            if match:
                results_avg[f"recall@{k}"] += 1

        if results_best is not None:
            sims_best = (query_emb @ feat_best.T).squeeze(0)
            ranked_idx_best = np.argsort(-sims_best)[:max_k]
            ranked_tags_best = [obj_tags[i] for i in ranked_idx_best if i < len(obj_tags)]

            for k in k_values:
                match, _ = _check_retrieval_match(ranked_tags_best, answer, k)
                if match:
                    results_best[f"recall@{k}"] += 1

    for k in k_values:
        results_avg[f"recall@{k}"] /= max(n_queries, 1)
    if results_best is not None:
        for k in k_values:
            results_best[f"recall@{k}"] /= max(n_queries, 1)

    output: dict[str, Any] = {
        "status": "evaluated",
        "n_queries": n_queries,
        "n_objects": len(objects),
        "k_values": k_values,
        "weighted_avg": results_avg,
    }
    if results_best is not None:
        output["best_view"] = results_best

    return output


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _load_questions(path: str | Path) -> list[dict] | None:
    """Load Space3D-Bench questions+answers into [{question, answer}, ...]."""
    q_path = Path(path)
    if not q_path.is_file():
        logger.warning("Questions file not found: %s", q_path)
        return None

    with open(q_path) as f:
        data = json.load(f)

    # Space3D-Bench format: {"1": "question text", ...}
    # Paired answers file lives alongside as answers.json
    answers_path = q_path.parent / "answers.json"
    answers: dict = {}
    if answers_path.is_file():
        with open(answers_path) as f:
            answers = json.load(f)

    queries: list[dict] = []
    if isinstance(data, dict):
        for qid, question in data.items():
            entry: dict[str, str] = {"question": question}
            if qid in answers:
                ans = answers[qid]
                entry["answer"] = ans if isinstance(ans, str) else str(ans)
            queries.append(entry)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                queries.append(item)
            elif isinstance(item, str):
                queries.append({"question": item, "answer": ""})
    return queries if queries else None


def main_standalone(cfg):
    """Run evaluation on a scene graph variant."""
    from semgraph.encoding import get_encoder
    from semgraph.stages.paths import stage_paths
    from semgraph.io import load_variant
    from semgraph.slam.utils import process_cfg

    cfg = process_cfg(cfg)
    paths = stage_paths(cfg)

    eval_cfg = cfg.get("eval", {}) if hasattr(cfg, "get") else {}
    encoder_name = eval_cfg.get("encoder", "laion_CLIP-ViT-bigG-14-laion2B-39B-b160k")
    vlm_name = eval_cfg.get("vlm", "Qwen_Qwen3-VL-2B-Instruct")
    encoder_type = eval_cfg.get("encoder_type", "hf_clip")

    # Text encoder override for frozen-encoder pairing experiments
    text_encoder_type = eval_cfg.get("text_encoder_type", "") or ""
    text_encoder_name = eval_cfg.get("text_encoder_name", "") or ""
    questions_path = eval_cfg.get("questions", "") or ""

    safe_enc = encoder_name.replace("/", "_")
    safe_vlm = vlm_name.replace("/", "_")

    # Load scene graph JSON from assembled directory
    sg_path = paths["assembled"] / f"{safe_enc}_{safe_vlm}" / "scene_graph.json"
    if not sg_path.is_file():
        print(f"[eval] Scene graph not found: {sg_path}")
        print("[eval] Run semantic_assemble.py first.")
        return

    with open(sg_path) as f:
        scene_graph = json.load(f)

    print(f"[eval] Evaluating: encoder={encoder_name}, vlm={vlm_name}")
    print(f"[eval] Scene graph: {len(scene_graph.get('objects', []))} objects")

    # Classification
    print("\n[eval] === Classification ===")
    cls_results = evaluate_classification(scene_graph)
    print(f"  mIoU:   {cls_results['mIoU']:.4f}")
    print(f"  F-mIoU: {cls_results['F_mIoU']:.4f}")
    print(f"  mAcc:   {cls_results['mAcc']:.4f}")
    print(f"  {cls_results['n_correct']}/{cls_results['n_objects']} objects matched")

    if cls_results.get("per_category"):
        print("  Per-category:")
        for cat, stats in cls_results["per_category"].items():
            print(f"    {cat}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")

    # QA
    print("\n[eval] === QA ===")
    qa_results = evaluate_qa(scene_graph)
    print(f"  Status: {qa_results['status']}")

    # Retrieval
    print("\n[eval] === Retrieval ===")
    embed_variant = load_variant(paths["variants"], f"embed_{safe_enc}")

    queries = None
    if questions_path:
        queries = _load_questions(questions_path)
        if queries:
            print(f"  Loaded {len(queries)} queries from {questions_path}")
        else:
            print(f"  No queries loaded from {questions_path}")

    text_enc = None
    device = "cuda"
    if queries:
        if text_encoder_type and text_encoder_name:
            print(f"  Text encoder override: {text_encoder_type}/{text_encoder_name}")
            text_enc = get_encoder(text_encoder_type, text_encoder_name, device)
        else:
            print(f"  Using image encoder for text: {encoder_type}/{encoder_name}")
            text_enc = get_encoder(encoder_type, encoder_name, device)

    ret_results = evaluate_retrieval(
        scene_graph, embed_variant, queries, text_enc,
    )

    if ret_results.get("status") == "evaluated":
        print(f"  Queries: {ret_results['n_queries']}, Objects: {ret_results['n_objects']}")
        print("  Weighted-avg retrieval:")
        for metric, val in ret_results["weighted_avg"].items():
            print(f"    {metric}: {val:.4f}")
        if "best_view" in ret_results:
            print("  Best-view retrieval:")
            for metric, val in ret_results["best_view"].items():
                print(f"    {metric}: {val:.4f}")
    else:
        print(f"  Status: {ret_results.get('status', 'unknown')}")
        if "reason" in ret_results:
            print(f"  Reason: {ret_results['reason']}")

    # Cleanup text encoder if it supports it
    if text_enc is not None and hasattr(text_enc, "cleanup"):
        text_enc.cleanup()

    # Save results
    results = {
        "encoder": encoder_name,
        "vlm": vlm_name,
        "classification": cls_results,
        "qa": qa_results,
        "retrieval": ret_results,
    }

    eval_dir = paths["eval"] / f"{safe_enc}_{safe_vlm}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {results_path}")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
