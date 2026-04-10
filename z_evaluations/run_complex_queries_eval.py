#!/usr/bin/env python3
"""
Complex Queries Evaluation Script
==================================

Evaluates VLM, CLIP, and YOLO+SentenceTransformer models on the custom
affordance/negation query dataset.

Reports:
- Overall Top-1 and Top-5 accuracy
- Breakdown by query type (affordance vs negation)
- Per-scene results
- Nicely formatted Markdown tables

Usage:
    python run_complex_queries_eval.py \
        --scene_graphs_root /path/to/data \
        --queries_path /path/to/complex_queries.json \
        --output_dir /path/to/results
"""

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

import numpy as np

# =============================================================================
# GPU Memory Management
# =============================================================================

def clear_gpu_memory():
    """Aggressively clear GPU memory between model runs."""
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    
    # Clear HuggingFace cache directory (models stay cached, just free RAM)
    gc.collect()


def clear_hf_model_cache():
    """Clear HuggingFace model cache to free disk space if needed."""
    import shutil
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists():
        # Don't delete, just report size
        size_gb = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / (1024**3)
        print(f"[Cache] HuggingFace cache size: {size_gb:.2f} GB")


# =============================================================================
# Configuration
# =============================================================================

SCENES = ["room0", "room1", "office2", "office3"]

CONFIG_SUFFIXES = {
    "oracle": "batch_vlm_local",
    "qwen": "batch_qwen", 
    "paligemma": "batch_paligemma",
}

MODEL_INFO = {
    # VLM models
    "qwen": {"name": "Qwen3-VL-2B-Instruct", "type": "vlm"},
    "paligemma": {"name": "PaliGemma2-3b-mix-224", "type": "vlm"},
    "gpt4": {"name": "GPT-4o-mini", "type": "vlm"},
    # CLIP models
    "mobileclip": {"name": "MobileCLIP2-S3", "type": "clip"},
    "pecore": {"name": "PE-Core-T-16-384", "type": "clip"},
    "tinyclip": {"name": "TinyCLIP-ViT-8M", "type": "clip"},
    # YOLO baseline
    "yolo": {"name": "YOLO + all-MiniLM-L6-v2", "type": "yolo"},
}


# =============================================================================
# Answer Matching
# =============================================================================

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.lower().strip()
    answer = re.sub(r'[^\w\s]', ' ', answer)
    answer = ' '.join(answer.split())
    return answer


def check_answer_match(
    predicted: str,
    ground_truth: str,
    is_retrieval: bool = False
) -> Tuple[bool, float]:
    """
    Check if predicted answer matches ground truth.
    
    For retrieval, we check if GT appears in retrieved labels.
    For VLM, we check semantic overlap.
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    # Exact match
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True, 1.0
    
    # Word overlap
    gt_words = set(gt_norm.split())
    pred_words = set(pred_norm.split())
    
    # Check if main object word matches
    # Ground truth is usually 1-3 words like "coffee table", "nightstand"
    if gt_words & pred_words:
        overlap = len(gt_words & pred_words) / len(gt_words)
        if overlap >= 0.5:
            return True, overlap
    
    return False, 0.0


def check_top_k_match(
    retrieved_objects: List[Dict],
    ground_truth: str,
    k: int = 5
) -> Tuple[bool, int, float]:
    """
    Check if any of top-k retrieved objects match ground truth.
    
    Returns:
        (is_correct, rank, score) where rank is 1-indexed (0 if no match)
    """
    gt_norm = normalize_answer(ground_truth)
    gt_words = set(gt_norm.split())
    
    for i, obj in enumerate(retrieved_objects[:k]):
        # Get label from various possible keys
        label = obj.get("label") or obj.get("caption") or obj.get("class_name", "")
        label_norm = normalize_answer(label)
        label_words = set(label_norm.split())
        
        # Check for match
        if gt_norm in label_norm or label_norm in gt_norm:
            score = obj.get("score", obj.get("confidence", 1.0))
            return True, i + 1, score
        
        # Word overlap
        if gt_words & label_words:
            overlap = len(gt_words & label_words) / len(gt_words)
            if overlap >= 0.5:
                score = obj.get("score", obj.get("confidence", 1.0))
                return True, i + 1, score
    
    return False, 0, 0.0


# =============================================================================
# Query Loading
# =============================================================================

def load_complex_queries(queries_path: Path) -> Dict[str, List[Dict]]:
    """
    Load complex queries and organize by scene.
    
    Returns:
        Dict mapping scene -> list of query dicts
    """
    with open(queries_path, 'r') as f:
        data = json.load(f)
    
    queries = data.get("complex_queries", data)
    
    # Organize by scene
    by_scene = defaultdict(list)
    for q in queries:
        scene = q.get("scene", "unknown")
        by_scene[scene].append(q)
    
    return dict(by_scene)


# =============================================================================
# Scene Graph Loading
# =============================================================================

def find_scene_files(base_path: Path, config: str, scene: str) -> Dict[str, Optional[Path]]:
    """Find scene graph files for a config/scene."""
    files = {"obj_json": None, "edge_json": None, "pkl": None, "detections": None}
    
    scene_path = base_path / config / scene
    if not scene_path.exists():
        return files
    
    for f in scene_path.glob("*.json"):
        fname = f.name.lower()
        if "obj_json" in fname:
            files["obj_json"] = f
        elif "edge_json" in fname:
            files["edge_json"] = f
        elif "config_params_detections" in fname:
            files["detections_config"] = f
    
    for f in scene_path.glob("*.pkl.gz"):
        if "pcd_" in f.name.lower():
            files["pkl"] = f
    
    # Look for detections folder
    det_folder = scene_path / "detections"
    if det_folder.exists():
        files["detections"] = det_folder
    
    return files


def load_scene_context(obj_json_path: Path, edge_json_path: Optional[Path] = None) -> str:
    """Build text context from scene graph."""
    context_parts = []
    
    if obj_json_path and obj_json_path.exists():
        with open(obj_json_path, 'r') as f:
            objects = json.load(f)
        
        context_parts.append("=== OBJECTS IN SCENE ===")
        for obj_id, obj_data in list(objects.items())[:50]:
            class_name = obj_data.get("class_name", "unknown")
            caption = obj_data.get("object_caption") or obj_data.get("consolidated_caption", "")
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            
            if caption:
                context_parts.append(f"- [{class_name}]: {caption}")
            else:
                context_parts.append(f"- [{class_name}]")
    
    if edge_json_path and edge_json_path.exists():
        with open(edge_json_path, 'r') as f:
            edges = json.load(f)
        
        context_parts.append("\n=== RELATIONSHIPS ===")
        edge_items = list(edges.items()) if isinstance(edges, dict) else edges
        for edge_data in edge_items[:30]:
            if isinstance(edge_data, tuple):
                edge_data = edge_data[1]
            obj1 = edge_data.get("obj1_class", "object")
            obj2 = edge_data.get("obj2_class", "object")
            relation = edge_data.get("relation", "related_to")
            context_parts.append(f"- {obj1} {relation} {obj2}")
    
    return "\n".join(context_parts)


# =============================================================================
# Model Evaluators
# =============================================================================

class VLMEvaluator:
    """Evaluate VLM models on complex queries."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
    
    def load(self):
        """Load VLM model."""
        from vlm_load_and_query import get_model
        self.model = get_model(self.model_name, device=self.device)
        self.model.load()
    
    def evaluate_scene(
        self,
        queries: List[Dict],
        context: str
    ) -> List[Dict]:
        """Evaluate queries for a scene."""
        results = []
        
        for q in queries:
            question = q["query"]
            gt_answer = q["answer"]
            q_type = q.get("type", "unknown")
            
            try:
                predicted = self.model.query(context, question)
            except Exception as e:
                predicted = f"ERROR: {e}"
            
            is_correct, confidence = check_answer_match(predicted, gt_answer)
            
            results.append({
                "id": q["id"],
                "type": q_type,
                "query": question,
                "predicted": predicted,
                "ground_truth": gt_answer,
                "correct": is_correct,
                "confidence": confidence,
            })
        
        return results
    
    def cleanup(self):
        """Release model resources."""
        if self.model:
            self.model.cleanup()
            self.model = None
        clear_gpu_memory()


class CLIPEvaluator:
    """Evaluate CLIP retrieval on complex queries."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
    
    def load(self):
        """Load CLIP model."""
        from clip_load_and_query import get_clip_model
        self.model = get_clip_model(self.model_name, device=self.device)
        self.model.load()
    
    def evaluate_scene(
        self,
        queries: List[Dict],
        pkl_path: Optional[Path] = None,
        obj_json_path: Optional[Path] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """Evaluate queries for a scene using CLIP retrieval."""
        from clip_load_and_query import ObjectDatabase
        
        # Load object database
        db = ObjectDatabase()
        if pkl_path and pkl_path.exists():
            db.load_from_pkl(pkl_path)
        elif obj_json_path and obj_json_path.exists():
            db.load_from_json(obj_json_path)
            db.compute_embeddings(self.model)
        else:
            return []
        
        results = []
        
        for q in queries:
            question = q["query"]
            gt_answer = q["answer"]
            q_type = q.get("type", "unknown")
            
            # Encode and retrieve
            q_embedding = self.model.encode_text([question])
            retrieved = db.retrieve_top_k(q_embedding, k=top_k)
            
            # Convert to dict format
            retrieved_dicts = [
                {"label": r[2], "score": r[1]}
                for r in retrieved
            ]
            
            # Check accuracy
            top1_correct, rank1, score1 = check_top_k_match(retrieved_dicts, gt_answer, k=1)
            top5_correct, rank5, score5 = check_top_k_match(retrieved_dicts, gt_answer, k=5)
            
            results.append({
                "id": q["id"],
                "type": q_type,
                "query": question,
                "top_retrieved": retrieved_dicts[:5],
                "ground_truth": gt_answer,
                "top1_correct": top1_correct,
                "top5_correct": top5_correct,
                "match_rank": rank5,
            })
        
        return results
    
    def cleanup(self):
        """Release model resources."""
        if self.model:
            self.model.cleanup()
            self.model = None
        clear_gpu_memory()


class YOLOEvaluator:
    """Evaluate YOLO + SentenceTransformers baseline."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.retriever = None
    
    def load(self):
        """Load sentence-transformers model."""
        from yolo_load_and_query import SentenceTransformerRetriever
        self.retriever = SentenceTransformerRetriever(device=self.device)
        self.retriever.load_model()
    
    def evaluate_scene(
        self,
        queries: List[Dict],
        obj_json_path: Optional[Path] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """Evaluate using YOLO class names only."""
        if not obj_json_path or not obj_json_path.exists():
            return []
        
        # Load objects (class names only)
        self.retriever.load_from_obj_json(obj_json_path)
        self.retriever.compute_embeddings()
        
        results = []
        
        for q in queries:
            question = q["query"]
            gt_answer = q["answer"]
            q_type = q.get("type", "unknown")
            
            # Query
            retrieved = self.retriever.query(question, top_k=top_k)
            retrieved_dicts = [
                {"label": r[2], "score": r[1]}
                for r in retrieved
            ]
            
            # Check accuracy
            top1_correct, rank1, _ = check_top_k_match(retrieved_dicts, gt_answer, k=1)
            top5_correct, rank5, _ = check_top_k_match(retrieved_dicts, gt_answer, k=5)
            
            results.append({
                "id": q["id"],
                "type": q_type,
                "query": question,
                "top_retrieved": retrieved_dicts[:5],
                "ground_truth": gt_answer,
                "top1_correct": top1_correct,
                "top5_correct": top5_correct,
                "match_rank": rank5,
            })
        
        return results
    
    def cleanup(self):
        """Release resources."""
        if self.retriever:
            self.retriever.cleanup()
            self.retriever = None
        clear_gpu_memory()


# =============================================================================
# Main Evaluation Runner
# =============================================================================

class ComplexQueriesEvaluator:
    """Run all evaluations on complex queries dataset."""
    
    def __init__(
        self,
        scene_graphs_root: Path,
        queries_path: Path,
        output_dir: Path,
        device: str = "cuda"
    ):
        self.scene_graphs_root = scene_graphs_root
        self.queries_path = queries_path
        self.output_dir = output_dir
        self.device = device
        
        self.queries_by_scene = load_complex_queries(queries_path)
        self.results: Dict[str, Dict] = {}
    
    def evaluate_vlm(
        self,
        model_name: str,
        config: str,
        scenes: List[str]
    ) -> Dict:
        """Evaluate a VLM model."""
        print(f"\n{'='*60}")
        print(f"Evaluating VLM: {model_name.upper()} (config: {config})")
        print(f"{'='*60}")
        
        evaluator = VLMEvaluator(model_name, device=self.device)
        evaluator.load()
        
        results = {
            "model": model_name,
            "model_info": MODEL_INFO.get(model_name, {}),
            "config": config,
            "scenes": {},
            "by_type": {"affordance": [], "negation": []},
        }
        
        for scene in scenes:
            if scene not in self.queries_by_scene:
                continue
            
            queries = self.queries_by_scene[scene]
            files = find_scene_files(self.scene_graphs_root, config, scene)
            
            if not files["obj_json"]:
                print(f"  Skipping {scene}: No scene graph found")
                continue
            
            context = load_scene_context(files["obj_json"], files.get("edge_json"))
            print(f"  Evaluating {scene} ({len(queries)} queries)...")
            
            scene_results = evaluator.evaluate_scene(queries, context)
            results["scenes"][scene] = scene_results
            
            # Track by type
            for r in scene_results:
                results["by_type"][r["type"]].append(r)
        
        evaluator.cleanup()
        return results
    
    def evaluate_clip(
        self,
        model_name: str,
        config: str,
        scenes: List[str]
    ) -> Dict:
        """Evaluate a CLIP model."""
        print(f"\n{'='*60}")
        print(f"Evaluating CLIP: {model_name.upper()} (config: {config})")
        print(f"{'='*60}")
        
        evaluator = CLIPEvaluator(model_name, device=self.device)
        evaluator.load()
        
        results = {
            "model": model_name,
            "model_info": MODEL_INFO.get(model_name, {}),
            "config": config,
            "scenes": {},
            "by_type": {"affordance": [], "negation": []},
        }
        
        for scene in scenes:
            if scene not in self.queries_by_scene:
                continue
            
            queries = self.queries_by_scene[scene]
            files = find_scene_files(self.scene_graphs_root, config, scene)
            
            if not files["pkl"] and not files["obj_json"]:
                print(f"  Skipping {scene}: No data found")
                continue
            
            print(f"  Evaluating {scene} ({len(queries)} queries)...")
            
            scene_results = evaluator.evaluate_scene(
                queries,
                pkl_path=files.get("pkl"),
                obj_json_path=files.get("obj_json")
            )
            results["scenes"][scene] = scene_results
            
            for r in scene_results:
                results["by_type"][r["type"]].append(r)
        
        evaluator.cleanup()
        return results
    
    def evaluate_yolo(
        self,
        config: str,
        scenes: List[str]
    ) -> Dict:
        """Evaluate YOLO + SentenceTransformers baseline."""
        print(f"\n{'='*60}")
        print(f"Evaluating YOLO Baseline (config: {config})")
        print(f"{'='*60}")
        
        evaluator = YOLOEvaluator(device=self.device)
        evaluator.load()
        
        results = {
            "model": "yolo",
            "model_info": MODEL_INFO.get("yolo", {}),
            "config": config,
            "scenes": {},
            "by_type": {"affordance": [], "negation": []},
        }
        
        for scene in scenes:
            if scene not in self.queries_by_scene:
                continue
            
            queries = self.queries_by_scene[scene]
            files = find_scene_files(self.scene_graphs_root, config, scene)
            
            if not files["obj_json"]:
                print(f"  Skipping {scene}: No obj_json found")
                continue
            
            print(f"  Evaluating {scene} ({len(queries)} queries)...")
            
            scene_results = evaluator.evaluate_scene(
                queries,
                obj_json_path=files.get("obj_json")
            )
            results["scenes"][scene] = scene_results
            
            for r in scene_results:
                results["by_type"][r["type"]].append(r)
        
        evaluator.cleanup()
        return results
    
    def compute_aggregate_stats(self, results: Dict) -> Dict:
        """Compute aggregate statistics from results."""
        stats = {
            "overall": {},
            "by_type": {},
            "by_scene": {},
        }
        
        # Check if VLM (has "correct") or retrieval (has "top1_correct")
        is_retrieval = any(
            "top1_correct" in r
            for scene_results in results.get("scenes", {}).values()
            for r in scene_results
        )
        
        if is_retrieval:
            # Retrieval metrics
            all_results = []
            for scene_results in results.get("scenes", {}).values():
                all_results.extend(scene_results)
            
            if all_results:
                stats["overall"] = {
                    "top1_accuracy": np.mean([r["top1_correct"] for r in all_results]),
                    "top5_accuracy": np.mean([r["top5_correct"] for r in all_results]),
                    "total": len(all_results),
                }
            
            # By type
            for q_type in ["affordance", "negation"]:
                type_results = results.get("by_type", {}).get(q_type, [])
                if type_results:
                    stats["by_type"][q_type] = {
                        "top1_accuracy": np.mean([r["top1_correct"] for r in type_results]),
                        "top5_accuracy": np.mean([r["top5_correct"] for r in type_results]),
                        "total": len(type_results),
                    }
            
            # By scene
            for scene, scene_results in results.get("scenes", {}).items():
                if scene_results:
                    stats["by_scene"][scene] = {
                        "top1_accuracy": np.mean([r["top1_correct"] for r in scene_results]),
                        "top5_accuracy": np.mean([r["top5_correct"] for r in scene_results]),
                        "total": len(scene_results),
                    }
        else:
            # VLM metrics
            all_results = []
            for scene_results in results.get("scenes", {}).values():
                all_results.extend(scene_results)
            
            if all_results:
                stats["overall"] = {
                    "accuracy": np.mean([r["correct"] for r in all_results]),
                    "total": len(all_results),
                }
            
            # By type
            for q_type in ["affordance", "negation"]:
                type_results = results.get("by_type", {}).get(q_type, [])
                if type_results:
                    stats["by_type"][q_type] = {
                        "accuracy": np.mean([r["correct"] for r in type_results]),
                        "total": len(type_results),
                    }
            
            # By scene
            for scene, scene_results in results.get("scenes", {}).items():
                if scene_results:
                    stats["by_scene"][scene] = {
                        "accuracy": np.mean([r["correct"] for r in scene_results]),
                        "total": len(scene_results),
                    }
        
        return stats
    
    def run_all(
        self,
        vlm_models: List[str],
        clip_models: List[str],
        config: str,
        scenes: List[str],
        include_yolo: bool = True
    ):
        """Run all evaluations sequentially."""
        
        # VLM models (one at a time)
        for model in vlm_models:
            try:
                results = self.evaluate_vlm(model, config, scenes)
                results["stats"] = self.compute_aggregate_stats(results)
                self.results[f"vlm_{model}"] = results
                self._print_summary(f"VLM: {model}", results["stats"])
            except Exception as e:
                print(f"Error evaluating VLM {model}: {e}")
            finally:
                clear_gpu_memory()
        
        # CLIP models (one at a time)
        for model in clip_models:
            try:
                results = self.evaluate_clip(model, config, scenes)
                results["stats"] = self.compute_aggregate_stats(results)
                self.results[f"clip_{model}"] = results
                self._print_summary(f"CLIP: {model}", results["stats"])
            except Exception as e:
                print(f"Error evaluating CLIP {model}: {e}")
            finally:
                clear_gpu_memory()
        
        # YOLO baseline
        if include_yolo:
            try:
                results = self.evaluate_yolo(config, scenes)
                results["stats"] = self.compute_aggregate_stats(results)
                self.results["yolo"] = results
                self._print_summary("YOLO Baseline", results["stats"])
            except Exception as e:
                print(f"Error evaluating YOLO: {e}")
            finally:
                clear_gpu_memory()
    
    def _print_summary(self, name: str, stats: Dict):
        """Print quick summary."""
        print(f"\n  {name} Summary:")
        overall = stats.get("overall", {})
        if "accuracy" in overall:
            print(f"    Overall Accuracy: {overall['accuracy']:.2%}")
        if "top1_accuracy" in overall:
            print(f"    Top-1: {overall['top1_accuracy']:.2%}, Top-5: {overall['top5_accuracy']:.2%}")
        
        for q_type, type_stats in stats.get("by_type", {}).items():
            if "accuracy" in type_stats:
                print(f"    {q_type.capitalize()}: {type_stats['accuracy']:.2%}")
            elif "top1_accuracy" in type_stats:
                print(f"    {q_type.capitalize()}: Top-1={type_stats['top1_accuracy']:.2%}, Top-5={type_stats['top5_accuracy']:.2%}")
    
    def generate_report(self) -> str:
        """Generate Markdown report with tables."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = f"""# Complex Queries Evaluation Report

Generated: {timestamp}

## Overview

This report evaluates models on 40 complex spatial queries (10 per scene):
- **Affordance queries** (5 per scene): What can I use to do X?
- **Negation queries** (5 per scene): What is NOT X that I can use for Y?

---

## VLM Model Results

### Overall Accuracy

| Model | Overall ↑ | Affordance ↑ | Negation ↑ |
|-------|-----------|--------------|------------|
"""
        
        for key, results in self.results.items():
            if not key.startswith("vlm_"):
                continue
            
            model_name = results.get("model_info", {}).get("name", results.get("model", "Unknown"))
            stats = results.get("stats", {})
            
            overall = stats.get("overall", {}).get("accuracy", 0)
            aff = stats.get("by_type", {}).get("affordance", {}).get("accuracy", 0)
            neg = stats.get("by_type", {}).get("negation", {}).get("accuracy", 0)
            
            md += f"| {model_name} | **{overall:.2%}** | {aff:.2%} | {neg:.2%} |\n"
        
        md += """
### Per-Scene VLM Results

| Model | Scene | Accuracy |
|-------|-------|----------|
"""
        
        for key, results in self.results.items():
            if not key.startswith("vlm_"):
                continue
            
            model_name = results.get("model_info", {}).get("name", results.get("model", "Unknown"))
            for scene, scene_stats in results.get("stats", {}).get("by_scene", {}).items():
                acc = scene_stats.get("accuracy", 0)
                md += f"| {model_name} | {scene} | {acc:.2%} |\n"
        
        md += """
---

## CLIP Retrieval Results

### Overall Accuracy

| Model | Top-1 ↑ | Top-5 ↑ | Aff Top-1 | Aff Top-5 | Neg Top-1 | Neg Top-5 |
|-------|---------|---------|-----------|-----------|-----------|-----------|
"""
        
        for key, results in self.results.items():
            if not key.startswith("clip_"):
                continue
            
            model_name = results.get("model_info", {}).get("name", results.get("model", "Unknown"))
            stats = results.get("stats", {})
            
            overall = stats.get("overall", {})
            top1 = overall.get("top1_accuracy", 0)
            top5 = overall.get("top5_accuracy", 0)
            
            aff = stats.get("by_type", {}).get("affordance", {})
            aff_top1 = aff.get("top1_accuracy", 0)
            aff_top5 = aff.get("top5_accuracy", 0)
            
            neg = stats.get("by_type", {}).get("negation", {})
            neg_top1 = neg.get("top1_accuracy", 0)
            neg_top5 = neg.get("top5_accuracy", 0)
            
            md += f"| {model_name} | **{top1:.2%}** | **{top5:.2%}** | {aff_top1:.2%} | {aff_top5:.2%} | {neg_top1:.2%} | {neg_top5:.2%} |\n"
        
        md += """
---

## YOLO Baseline Results

### Overall Accuracy (Detection Classes Only + Sentence-Transformers)

| Model | Top-1 ↑ | Top-5 ↑ | Aff Top-1 | Aff Top-5 | Neg Top-1 | Neg Top-5 |
|-------|---------|---------|-----------|-----------|-----------|-----------|
"""
        
        if "yolo" in self.results:
            results = self.results["yolo"]
            model_name = results.get("model_info", {}).get("name", "YOLO + MiniLM")
            stats = results.get("stats", {})
            
            overall = stats.get("overall", {})
            top1 = overall.get("top1_accuracy", 0)
            top5 = overall.get("top5_accuracy", 0)
            
            aff = stats.get("by_type", {}).get("affordance", {})
            aff_top1 = aff.get("top1_accuracy", 0)
            aff_top5 = aff.get("top5_accuracy", 0)
            
            neg = stats.get("by_type", {}).get("negation", {})
            neg_top1 = neg.get("top1_accuracy", 0)
            neg_top5 = neg.get("top5_accuracy", 0)
            
            md += f"| {model_name} | **{top1:.2%}** | **{top5:.2%}** | {aff_top1:.2%} | {aff_top5:.2%} | {neg_top1:.2%} | {neg_top5:.2%} |\n"
        
        md += """
---

## Summary Comparison

### Best Models by Category

| Category | Best Model | Score |
|----------|------------|-------|
"""
        
        # Find best VLM
        best_vlm = None
        best_vlm_score = 0
        for key, results in self.results.items():
            if key.startswith("vlm_"):
                score = results.get("stats", {}).get("overall", {}).get("accuracy", 0)
                if score > best_vlm_score:
                    best_vlm_score = score
                    best_vlm = results.get("model_info", {}).get("name", "Unknown")
        
        if best_vlm:
            md += f"| Best VLM | {best_vlm} | {best_vlm_score:.2%} |\n"
        
        # Find best retrieval (Top-5)
        best_ret = None
        best_ret_score = 0
        for key, results in self.results.items():
            if key.startswith("clip_") or key == "yolo":
                score = results.get("stats", {}).get("overall", {}).get("top5_accuracy", 0)
                if score > best_ret_score:
                    best_ret_score = score
                    best_ret = results.get("model_info", {}).get("name", "Unknown")
        
        if best_ret:
            md += f"| Best Retrieval (Top-5) | {best_ret} | {best_ret_score:.2%} |\n"
        
        md += """
---

### Key Observations

- **Affordance queries** test functional understanding (e.g., "something to sit on")
- **Negation queries** test constraint handling (e.g., "not the sofa, not the floor")
- Higher negation scores indicate better logical reasoning

---

*Generated by run_complex_queries_eval.py*
"""
        
        return md
    
    def save_results(self):
        """Save all results."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = self.output_dir / "complex_queries_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=float)
        print(f"JSON saved to: {json_path}")
        
        # Save Markdown
        md_path = self.output_dir / "COMPLEX_QUERIES_RESULTS.md"
        report = self.generate_report()
        with open(md_path, 'w') as f:
            f.write(report)
        print(f"Markdown saved to: {md_path}")
        
        # Print report
        print("\n" + "="*60)
        print(report)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on complex queries")
    parser.add_argument(
        "--scene_graphs_root", type=str, required=True,
        help="Root directory with config folders (oracle/, qwen/, etc.)"
    )
    parser.add_argument(
        "--queries_path", type=str, required=True,
        help="Path to complex_queries.json"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./complex_queries_results",
        help="Output directory"
    )
    parser.add_argument(
        "--config", type=str, default="oracle",
        help="Config folder to use for scene graphs"
    )
    parser.add_argument(
        "--scenes", type=str, nargs="+", default=SCENES,
        help="Scenes to evaluate"
    )
    parser.add_argument(
        "--vlm_models", type=str, nargs="+", default=["qwen", "paligemma"],
        help="VLM models to evaluate"
    )
    parser.add_argument(
        "--clip_models", type=str, nargs="+", default=["mobileclip"],
        help="CLIP models to evaluate"
    )
    parser.add_argument(
        "--skip_yolo", action="store_true",
        help="Skip YOLO baseline"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    evaluator = ComplexQueriesEvaluator(
        scene_graphs_root=Path(args.scene_graphs_root),
        queries_path=Path(args.queries_path),
        output_dir=Path(args.output_dir),
        device=args.device,
    )
    
    evaluator.run_all(
        vlm_models=args.vlm_models,
        clip_models=args.clip_models,
        config=args.config,
        scenes=args.scenes,
        include_yolo=not args.skip_yolo,
    )
    
    evaluator.save_results()


if __name__ == "__main__":
    main()
