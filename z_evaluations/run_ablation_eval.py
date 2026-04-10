#!/usr/bin/env python3
"""
Ablation Evaluation Script
==========================
Compares CLIP embeddings and VLM captions across different configurations:
- Oracle (GPT-4 + ViT-H-14 CLIP)
- Qwen (Qwen3-VL-2B + various CLIP)
- PaliGemma (PaliGemma2-3b + various CLIP)

Outputs nicely formatted Markdown tables for all metrics.
"""

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Optional imports with graceful fallback
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found. CLIP embedding evaluation will be skipped.")

try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
    HAS_COCO_EVAL = True
except ImportError:
    HAS_COCO_EVAL = False
    print("Warning: pycocoevalcap not found. Text evaluation will be skipped.")

# =============================================================================
# Configuration
# =============================================================================

SCENES = ["room0", "room1", "office2", "office3"]

# Mapping of config names to their file suffix patterns
CONFIG_SUFFIXES = {
    "oracle": "batch_vlm_local",       # Oracle uses GPT-4 + ViT-H-14
    "qwen": "batch_qwen",              # Qwen3-VL-2B variants
    "paligemma": "batch_paligemma",    # PaliGemma2-3b variants
}

# CLIP model metadata for reporting
CLIP_MODELS = {
    "oracle": "ViT-H-14 (laion2b_s32b_b79k)",
    "qwen": "MobileCLIP2-S3 (dfndr2b)",
    "paligemma": "MobileCLIP2-S3 (dfndr2b)",
}

VLM_MODELS = {
    "oracle": "GPT-4-mini (OpenAI API)",
    "qwen": "Qwen3-VL-2B-Instruct",
    "paligemma": "PaliGemma2-3b-mix-224",
}

# =============================================================================
# CLIP Embedding Evaluation
# =============================================================================

def load_objects_from_pkl(pkl_path: Path) -> Tuple[List[dict], Optional[Any]]:
    """Load objects and CLIP embeddings from pkl.gz file."""
    if not HAS_TORCH:
        return [], None
    
    if not pkl_path.exists():
        print(f"  Warning: File not found: {pkl_path}")
        return [], None

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        objects_data = data.get('objects', [])
    else:
        objects_data = data

    embeddings = []
    valid_objects = []
    
    for obj in objects_data:
        if 'clip_ft' in obj and obj['clip_ft'] is not None:
            emb = obj['clip_ft']
            if isinstance(emb, list):
                emb = np.array(emb)
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            
            emb = F.normalize(emb.float().view(1, -1), dim=1)
            embeddings.append(emb)
            valid_objects.append(obj)
    
    if len(embeddings) == 0:
        return valid_objects, None
        
    embeddings_tensor = torch.cat(embeddings, dim=0)
    return valid_objects, embeddings_tensor


def compute_chamfer_metrics(ref_emb, cand_emb) -> Dict[str, float]:
    """Compute Chamfer-style semantic recall/precision."""
    if ref_emb is None or cand_emb is None:
        return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0}
    
    if ref_emb.shape[0] == 0 or cand_emb.shape[0] == 0:
        return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0}
    
    sim_matrix = torch.mm(ref_emb, cand_emb.t())
    
    max_sim_ref_to_cand, _ = torch.max(sim_matrix, dim=1)
    recall = torch.mean(max_sim_ref_to_cand).item()
    
    max_sim_cand_to_ref, _ = torch.max(sim_matrix, dim=0)
    precision = torch.mean(max_sim_cand_to_ref).item()
    
    f1 = 0.0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return {
        'recall': recall,
        'precision': precision, 
        'f1': f1,
        'n_oracle': ref_emb.shape[0],
        'n_candidate': cand_emb.shape[0],
    }


# =============================================================================
# VLM Text Evaluation
# =============================================================================

def load_captions_from_json(json_path: Path) -> Tuple[List[str], Dict]:
    """Load object captions from obj_json file."""
    if not json_path.exists():
        print(f"  Warning: File not found: {json_path}")
        return [], {}

    with open(json_path, "r") as f:
        data = json.load(f)
        
    captions = []
    obj_id_to_caption = {}
    
    if isinstance(data, dict):
        for key, obj_data in data.items():
            cap = obj_data.get("object_caption") or obj_data.get("consolidated_caption")
            class_name = obj_data.get("class_name", "object")
            
            if cap and isinstance(cap, str) and len(cap.strip()) > 0:
                captions.append(cap)
                obj_id_to_caption[key] = {"caption": cap, "class_name": class_name}
            elif cap and isinstance(cap, list) and len(cap) > 0:
                captions.append(str(cap[0]))
                obj_id_to_caption[key] = {"caption": str(cap[0]), "class_name": class_name}
    
    return captions, obj_id_to_caption


def load_edges_from_json(json_path: Path) -> List[dict]:
    """Load edges from edge_json file."""
    if not json_path.exists():
        return []

    with open(json_path, "r") as f:
        data = json.load(f)

    edges = []
    if isinstance(data, dict):
        for edge_key, edge_data in data.items():
            edges.append({
                "obj1_idx": edge_data.get("obj1_idx"),
                "obj2_idx": edge_data.get("obj2_idx"),
                "obj1_class": edge_data.get("obj1_class", ""),
                "obj2_class": edge_data.get("obj2_class", ""),
                "relation": edge_data.get("relation", "related_to"),
            })
    elif isinstance(data, list):
        for edge_data in data:
            edges.append({
                "obj1_idx": edge_data.get("obj1_idx"),
                "obj2_idx": edge_data.get("obj2_idx"),
                "obj1_class": edge_data.get("obj1_class", ""),
                "obj2_class": edge_data.get("obj2_class", ""),
                "relation": edge_data.get("relation", "related_to"),
            })

    return edges


def build_triplets(obj_map: Dict, edges: List[dict]) -> List[str]:
    """Build triplet strings for graph evaluation."""
    triplets = []
    for edge in edges:
        obj1_key = str(edge.get("obj1_idx"))
        obj2_key = str(edge.get("obj2_idx"))
        relation = edge.get("relation", "related_to")
        
        obj1_info = obj_map.get(obj1_key, {})
        obj2_info = obj_map.get(obj2_key, {})
        
        obj1_class = obj1_info.get("class_name", edge.get("obj1_class", "object"))
        obj2_class = obj2_info.get("class_name", edge.get("obj2_class", "object"))
        obj1_cap = obj1_info.get("caption", obj1_class)
        obj2_cap = obj2_info.get("caption", obj2_class)
        
        triplet = f"[{obj1_class}] {obj1_cap} | {relation} | [{obj2_class}] {obj2_cap}"
        triplets.append(triplet)
    
    return triplets


def compute_pairwise_matrix(scorer, row_caps: List[str], col_caps: List[str]) -> np.ndarray:
    """Compute NxM pairwise score matrix."""
    if len(row_caps) == 0 or len(col_caps) == 0:
        return np.zeros((max(1, len(row_caps)), max(1, len(col_caps))))
    
    gts = {}
    res = {}
    idx_map = []
    
    count = 0
    for i, r_cap in enumerate(row_caps):
        for j, c_cap in enumerate(col_caps):
            key = str(count)
            gts[key] = [r_cap]
            res[key] = [c_cap]
            idx_map.append((i, j))
            count += 1
    
    if count == 0:
        return np.zeros((len(row_caps), len(col_caps)))
    
    _, scores = scorer.compute_score(gts, res)
    
    matrix = np.zeros((len(row_caps), len(col_caps)))
    
    if isinstance(scores, (list, np.ndarray)):
        scores_list = list(scores) if isinstance(scores, np.ndarray) else scores
        for k, score_val in enumerate(scores_list):
            if k < len(idx_map):
                i, j = idx_map[k]
                if isinstance(score_val, dict):
                    matrix[i, j] = score_val.get('All', {}).get('f', 0.0)
                else:
                    matrix[i, j] = float(score_val)
    
    return matrix


def compute_text_metrics(oracle_caps: List[str], cand_caps: List[str], 
                         scorers: Dict, label: str = "") -> Dict:
    """Compute text-based recall/precision metrics."""
    results = {}
    
    if len(oracle_caps) == 0 or len(cand_caps) == 0:
        return {name: {"recall": 0.0, "precision": 0.0, "f1": 0.0} for name in scorers}
    
    for name, scorer in scorers.items():
        sim_matrix = compute_pairwise_matrix(scorer, oracle_caps, cand_caps)
        
        max_row_to_col = np.max(sim_matrix, axis=1)
        recall = float(np.mean(max_row_to_col))
        
        max_col_to_row = np.max(sim_matrix, axis=0)
        precision = float(np.mean(max_col_to_row))
        
        f1 = 0.0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        results[name] = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "n_oracle": len(oracle_caps),
            "n_candidate": len(cand_caps),
        }
    
    return results


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def find_files(base_path: Path, config_name: str, scene: str) -> Dict[str, Path]:
    """Find all relevant files for a given configuration and scene."""
    suffix = CONFIG_SUFFIXES.get(config_name, config_name)
    scene_path = base_path / config_name / scene
    
    files = {
        "obj_json": None,
        "edge_json": None,
        "pcd_pkl": None,
    }
    
    if not scene_path.exists():
        return files
    
    # Search for files with various naming patterns
    for f in scene_path.glob("*.json"):
        fname = f.name.lower()
        if "obj_json" in fname or "obj-json" in fname:
            files["obj_json"] = f
        elif "edge_json" in fname or "edge-json" in fname:
            files["edge_json"] = f
    
    for f in scene_path.glob("*.pkl.gz"):
        fname = f.name.lower()
        if "pcd_" in fname or "pcd-" in fname:
            files["pcd_pkl"] = f
    
    # Fallback: search with suffix pattern
    if files["obj_json"] is None:
        pattern_files = list(scene_path.glob(f"obj_json_{suffix}.json"))
        if pattern_files:
            files["obj_json"] = pattern_files[0]
    
    if files["edge_json"] is None:
        pattern_files = list(scene_path.glob(f"edge_json_{suffix}.json"))
        if pattern_files:
            files["edge_json"] = pattern_files[0]
    
    if files["pcd_pkl"] is None:
        pattern_files = list(scene_path.glob(f"pcd_{suffix}.pkl.gz"))
        if pattern_files:
            files["pcd_pkl"] = pattern_files[0]
    
    return files


def evaluate_config_vs_oracle(
    base_path: Path,
    config_name: str,
    scenes: List[str],
    skip_spice: bool = False
) -> Dict:
    """Evaluate a configuration against Oracle across all scenes."""
    
    results = {
        "config": config_name,
        "vlm_model": VLM_MODELS.get(config_name, "Unknown"),
        "clip_model": CLIP_MODELS.get(config_name, "Unknown"),
        "scenes": {},
        "aggregate": {},
    }
    
    # Accumulators for aggregation
    clip_metrics_all = []
    cider_obj_all = []
    spice_obj_all = []
    cider_triplet_all = []
    spice_triplet_all = []
    
    for scene in scenes:
        print(f"\n  Evaluating {config_name} on {scene}...")
        scene_results = {"clip": None, "text_objects": None, "text_triplets": None}
        
        # Find files
        oracle_files = find_files(base_path, "oracle", scene)
        config_files = find_files(base_path, config_name, scene)
        
        # === CLIP Embedding Evaluation ===
        if HAS_TORCH and oracle_files["pcd_pkl"] and config_files["pcd_pkl"]:
            _, oracle_emb = load_objects_from_pkl(oracle_files["pcd_pkl"])
            _, config_emb = load_objects_from_pkl(config_files["pcd_pkl"])
            
            clip_metrics = compute_chamfer_metrics(oracle_emb, config_emb)
            scene_results["clip"] = clip_metrics
            clip_metrics_all.append(clip_metrics)
            print(f"    CLIP - R: {clip_metrics['recall']:.4f}, P: {clip_metrics['precision']:.4f}, F1: {clip_metrics['f1']:.4f}")
        
        # === Text Evaluation ===
        if HAS_COCO_EVAL:
            scorers = {"CIDEr": Cider()}
            if not skip_spice:
                scorers["SPICE"] = Spice()
            
            # Object captions
            if oracle_files["obj_json"] and config_files["obj_json"]:
                oracle_caps, oracle_map = load_captions_from_json(oracle_files["obj_json"])
                config_caps, config_map = load_captions_from_json(config_files["obj_json"])
                
                if oracle_caps and config_caps:
                    text_metrics = compute_text_metrics(oracle_caps, config_caps, scorers, "Objects")
                    scene_results["text_objects"] = text_metrics
                    
                    if "CIDEr" in text_metrics:
                        cider_obj_all.append(text_metrics["CIDEr"])
                    if "SPICE" in text_metrics:
                        spice_obj_all.append(text_metrics["SPICE"])
                    
                    for metric_name, vals in text_metrics.items():
                        print(f"    {metric_name} Objects - R: {vals['recall']:.4f}, P: {vals['precision']:.4f}")
            
            # Triplets (edges)
            if oracle_files["edge_json"] and config_files["edge_json"]:
                oracle_edges = load_edges_from_json(oracle_files["edge_json"])
                config_edges = load_edges_from_json(config_files["edge_json"])
                
                if oracle_edges and config_edges:
                    oracle_triplets = build_triplets(oracle_map if 'oracle_map' in dir() else {}, oracle_edges)
                    config_triplets = build_triplets(config_map if 'config_map' in dir() else {}, config_edges)
                    
                    if oracle_triplets and config_triplets:
                        triplet_metrics = compute_text_metrics(oracle_triplets, config_triplets, scorers, "Triplets")
                        scene_results["text_triplets"] = triplet_metrics
                        
                        if "CIDEr" in triplet_metrics:
                            cider_triplet_all.append(triplet_metrics["CIDEr"])
                        if "SPICE" in triplet_metrics:
                            spice_triplet_all.append(triplet_metrics["SPICE"])
        
        results["scenes"][scene] = scene_results
    
    # === Aggregate Metrics ===
    if clip_metrics_all:
        results["aggregate"]["clip"] = {
            "recall": np.mean([m["recall"] for m in clip_metrics_all]),
            "precision": np.mean([m["precision"] for m in clip_metrics_all]),
            "f1": np.mean([m["f1"] for m in clip_metrics_all]),
        }
    
    if cider_obj_all:
        results["aggregate"]["cider_objects"] = {
            "recall": np.mean([m["recall"] for m in cider_obj_all]),
            "precision": np.mean([m["precision"] for m in cider_obj_all]),
            "f1": np.mean([m["f1"] for m in cider_obj_all]),
        }
    
    if spice_obj_all:
        results["aggregate"]["spice_objects"] = {
            "recall": np.mean([m["recall"] for m in spice_obj_all]),
            "precision": np.mean([m["precision"] for m in spice_obj_all]),
            "f1": np.mean([m["f1"] for m in spice_obj_all]),
        }
    
    if cider_triplet_all:
        results["aggregate"]["cider_triplets"] = {
            "recall": np.mean([m["recall"] for m in cider_triplet_all]),
            "precision": np.mean([m["precision"] for m in cider_triplet_all]),
            "f1": np.mean([m["f1"] for m in cider_triplet_all]),
        }
    
    return results


# =============================================================================
# Table Generation
# =============================================================================

def generate_markdown_tables(all_results: Dict, output_path: Path):
    """Generate nicely formatted Markdown tables."""
    
    md_content = """# Ablation Study: Scene Graph Evaluation Results

## Overview

This report compares different VLM and CLIP configurations against the Oracle baseline.

| Configuration | VLM Model | CLIP Model |
|--------------|-----------|------------|
"""
    
    for config_name, results in all_results.items():
        md_content += f"| {config_name} | {results['vlm_model']} | {results['clip_model']} |\n"
    
    # === CLIP Embedding Table ===
    md_content += """
---

## 1. CLIP Embedding Evaluation (Semantic Similarity)

Chamfer-style semantic recall measures how much of the Oracle's semantic content is captured.

### Per-Scene Results

| Config | Scene | Recall ↑ | Precision ↑ | F1 ↑ |
|--------|-------|----------|-------------|------|
"""
    
    for config_name, results in all_results.items():
        for scene, scene_data in results.get("scenes", {}).items():
            clip = scene_data.get("clip", {})
            if clip:
                md_content += f"| {config_name} | {scene} | {clip.get('recall', 0):.4f} | {clip.get('precision', 0):.4f} | {clip.get('f1', 0):.4f} |\n"
    
    md_content += """
### Aggregate Results

| Configuration | Avg Recall ↑ | Avg Precision ↑ | Avg F1 ↑ |
|--------------|--------------|-----------------|----------|
"""
    
    for config_name, results in all_results.items():
        agg = results.get("aggregate", {}).get("clip", {})
        if agg:
            md_content += f"| {config_name} | {agg.get('recall', 0):.4f} | {agg.get('precision', 0):.4f} | {agg.get('f1', 0):.4f} |\n"
    
    # === CIDEr Object Captions Table ===
    md_content += """
---

## 2. Object Caption Evaluation (CIDEr)

Measures caption quality using CIDEr metric.

### Per-Scene Results

| Config | Scene | Recall ↑ | Precision ↑ | F1 ↑ |
|--------|-------|----------|-------------|------|
"""
    
    for config_name, results in all_results.items():
        for scene, scene_data in results.get("scenes", {}).items():
            text_obj = scene_data.get("text_objects", {}).get("CIDEr", {})
            if text_obj:
                md_content += f"| {config_name} | {scene} | {text_obj.get('recall', 0):.4f} | {text_obj.get('precision', 0):.4f} | {text_obj.get('f1', 0):.4f} |\n"
    
    md_content += """
### Aggregate Results

| Configuration | Avg Recall ↑ | Avg Precision ↑ | Avg F1 ↑ |
|--------------|--------------|-----------------|----------|
"""
    
    for config_name, results in all_results.items():
        agg = results.get("aggregate", {}).get("cider_objects", {})
        if agg:
            md_content += f"| {config_name} | {agg.get('recall', 0):.4f} | {agg.get('precision', 0):.4f} | {agg.get('f1', 0):.4f} |\n"
    
    # === SPICE Object Captions Table ===
    md_content += """
---

## 3. Object Caption Evaluation (SPICE)

Measures semantic proposition coverage using SPICE.

### Aggregate Results

| Configuration | Avg Recall ↑ | Avg Precision ↑ | Avg F1 ↑ |
|--------------|--------------|-----------------|----------|
"""
    
    for config_name, results in all_results.items():
        agg = results.get("aggregate", {}).get("spice_objects", {})
        if agg:
            md_content += f"| {config_name} | {agg.get('recall', 0):.4f} | {agg.get('precision', 0):.4f} | {agg.get('f1', 0):.4f} |\n"
    
    # === Triplet (Graph) Evaluation ===
    md_content += """
---

## 4. Scene Graph Triplet Evaluation (CIDEr)

Evaluates [Subject] caption | relation | [Object] caption triplets.

### Aggregate Results

| Configuration | Avg Recall ↑ | Avg Precision ↑ | Avg F1 ↑ |
|--------------|--------------|-----------------|----------|
"""
    
    for config_name, results in all_results.items():
        agg = results.get("aggregate", {}).get("cider_triplets", {})
        if agg:
            md_content += f"| {config_name} | {agg.get('recall', 0):.4f} | {agg.get('precision', 0):.4f} | {agg.get('f1', 0):.4f} |\n"
    
    # === Summary ===
    md_content += """
---

## Summary

### Key Observations

- **CLIP Recall**: How much of Oracle's semantic content is captured
- **CLIP Precision**: How clean/non-hallucinated the generation is
- **CIDEr/SPICE**: Text-based caption and graph quality

### Interpretation Guide

| Metric | High Value Means |
|--------|------------------|
| Recall | Config covers most Oracle information |
| Precision | Config doesn't hallucinate extra objects |
| F1 | Balanced performance |

---

*Generated by run_ablation_eval.py*
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md_content)
    
    print(f"\nMarkdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation evaluation across configurations")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing oracle/, qwen/, paligemma/ folders")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--scenes", type=str, nargs="+", default=SCENES,
                        help="Scenes to evaluate")
    parser.add_argument("--configs", type=str, nargs="+", default=["qwen", "paligemma"],
                        help="Configurations to compare against Oracle")
    parser.add_argument("--skip_spice", action="store_true",
                        help="Skip SPICE metric (faster)")
    
    args = parser.parse_args()
    
    base_path = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ABLATION STUDY: Scene Graph Evaluation")
    print("=" * 60)
    print(f"Data Root: {base_path}")
    print(f"Scenes: {args.scenes}")
    print(f"Configs: {args.configs}")
    print("=" * 60)
    
    all_results = {}
    
    for config in args.configs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {config.upper()} vs ORACLE")
        print(f"{'='*60}")
        
        results = evaluate_config_vs_oracle(
            base_path=base_path,
            config_name=config,
            scenes=args.scenes,
            skip_spice=args.skip_spice
        )
        all_results[config] = results
    
    # Save raw JSON results
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nJSON results saved to: {json_path}")
    
    # Generate Markdown tables
    md_path = output_dir / "ABLATION_RESULTS.md"
    generate_markdown_tables(all_results, md_path)


if __name__ == "__main__":
    main()
