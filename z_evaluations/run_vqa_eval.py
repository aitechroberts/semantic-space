#!/usr/bin/env python3
"""
Space3D-Bench VQA Evaluation Script
====================================

Runs VLM and CLIP models through Space3D-Bench questions and reports:
- Top-1 Accuracy
- Top-5 Accuracy (for retrieval methods)
- Per-category breakdown
- Nicely formatted tables

Usage:
    python run_vqa_eval.py \
        --scene_graphs_root /path/to/scene_graphs \
        --space3d_root /path/to/Space3D-Bench/data \
        --output_dir /path/to/results \
        --models qwen paligemma gpt4
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

import numpy as np

# Import our query modules
from vlm_load_and_query import (
    get_model as get_vlm_model,
    load_scene_graph_context,
)
from clip_load_and_query import (
    get_clip_model,
    ObjectDatabase,
    generate_answer_from_retrieved,
)


# =============================================================================
# Configuration
# =============================================================================

SCENES = ["room0", "room1", "office2", "office3"]

# Mapping from your scene names to Space3D-Bench scene names
SCENE_MAP = {
    "room0": "room_0",
    "room1": "room_1",
    "office2": "office_2",
    "office3": "office_3",
}

# Config folder to file suffix mapping
CONFIG_SUFFIXES = {
    "oracle": "batch_vlm_local",
    "qwen": "batch_qwen",
    "paligemma": "batch_paligemma",
}

# Model metadata
MODEL_INFO = {
    "qwen": {"name": "Qwen3-VL-2B-Instruct", "type": "vlm"},
    "paligemma": {"name": "PaliGemma2-3b-mix-224", "type": "vlm"},
    "gpt4": {"name": "GPT-4o-mini", "type": "vlm"},
    "mobileclip": {"name": "MobileCLIP2-S3", "type": "clip"},
    "pecore": {"name": "PE-Core-T-16-384", "type": "clip"},
    "tinyclip": {"name": "TinyCLIP-ViT-8M", "type": "clip"},
}


# =============================================================================
# Answer Matching / Evaluation
# =============================================================================

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Lowercase
    answer = answer.lower()
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', ' ', answer)
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    return answer


def extract_number(text: str) -> Optional[int]:
    """Extract number from text."""
    # Try to find digits
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    
    # Word to number mapping
    word_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'no': 0, 'none': 0,
    }
    
    text_lower = text.lower()
    for word, num in word_map.items():
        if word in text_lower:
            return num
    
    return None


def check_answer_match(
    predicted: str,
    ground_truth: str,
    question: str,
    answer_type: str = "default"
) -> Tuple[bool, float]:
    """
    Check if predicted answer matches ground truth.
    
    Returns:
        (is_correct, confidence_score)
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    # Exact match
    if pred_norm == gt_norm:
        return True, 1.0
    
    # Check for counting questions
    if any(kw in question.lower() for kw in ['how many', 'count', 'number of']):
        pred_num = extract_number(predicted)
        gt_num = extract_number(ground_truth)
        if pred_num is not None and gt_num is not None:
            if pred_num == gt_num:
                return True, 1.0
            # Allow ±1 tolerance
            if abs(pred_num - gt_num) <= 1:
                return True, 0.5
    
    # Check for yes/no questions
    if question.lower().startswith(('is ', 'are ', 'does ', 'do ', 'can ', 'will ')):
        pred_yes = any(w in pred_norm for w in ['yes', 'correct', 'true', 'right'])
        pred_no = any(w in pred_norm for w in ['no', 'incorrect', 'false', 'wrong'])
        gt_yes = any(w in gt_norm for w in ['yes', 'correct', 'true', 'right'])
        gt_no = any(w in gt_norm for w in ['no', 'incorrect', 'false', 'wrong'])
        
        if (pred_yes and gt_yes) or (pred_no and gt_no):
            return True, 1.0
        if (pred_yes and gt_no) or (pred_no and gt_yes):
            return False, 0.0
    
    # Substring match (gt in pred or pred in gt)
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True, 0.7
    
    # Keyword overlap
    pred_words = set(pred_norm.split())
    gt_words = set(gt_norm.split())
    
    if len(gt_words) > 0:
        overlap = len(pred_words & gt_words) / len(gt_words)
        if overlap > 0.5:
            return True, overlap
    
    return False, 0.0


def check_top_k_match(
    retrieved_objects: List[Dict],
    ground_truth: str,
    question: str,
    k: int = 5
) -> Tuple[bool, int]:
    """
    Check if any of top-k retrieved objects match ground truth.
    
    Returns:
        (is_correct, rank) where rank is 1-indexed position of match (0 if no match)
    """
    gt_norm = normalize_answer(ground_truth)
    
    for i, obj in enumerate(retrieved_objects[:k]):
        caption = obj.get("caption", "")
        caption_norm = normalize_answer(caption)
        
        # Check for overlap
        if gt_norm in caption_norm or caption_norm in gt_norm:
            return True, i + 1
        
        # Keyword overlap
        gt_words = set(gt_norm.split())
        cap_words = set(caption_norm.split())
        if len(gt_words) > 0 and len(gt_words & cap_words) / len(gt_words) > 0.3:
            return True, i + 1
    
    return False, 0


# =============================================================================
# Evaluation Runner
# =============================================================================

class VQAEvaluator:
    """Runs VQA evaluation across models and scenes."""
    
    def __init__(
        self,
        scene_graphs_root: Path,
        space3d_root: Path,
        output_dir: Path,
        device: str = "cuda"
    ):
        self.scene_graphs_root = scene_graphs_root
        self.space3d_root = space3d_root
        self.output_dir = output_dir
        self.device = device
        
        self.results: Dict[str, Dict] = {}
    
    def find_scene_graph_files(
        self,
        config: str,
        scene: str
    ) -> Dict[str, Optional[Path]]:
        """Find scene graph files for a config/scene."""
        files = {"obj_json": None, "edge_json": None, "pkl": None}
        
        scene_path = self.scene_graphs_root / config / scene
        if not scene_path.exists():
            return files
        
        # Search for files
        for f in scene_path.glob("*.json"):
            if "obj_json" in f.name.lower():
                files["obj_json"] = f
            elif "edge_json" in f.name.lower():
                files["edge_json"] = f
        
        for f in scene_path.glob("*.pkl.gz"):
            if "pcd_" in f.name.lower():
                files["pkl"] = f
        
        return files
    
    def load_space3d_data(self, scene: str) -> Tuple[Dict, Dict]:
        """Load Space3D-Bench questions and answers for a scene."""
        space3d_scene = SCENE_MAP.get(scene, scene)
        scene_path = self.space3d_root / space3d_scene
        
        questions = {}
        answers = {}
        
        questions_path = scene_path / "questions.json"
        if questions_path.exists():
            with open(questions_path, "r") as f:
                questions = json.load(f)
        
        answers_path = scene_path / "answers.json"
        if answers_path.exists():
            with open(answers_path, "r") as f:
                answers = json.load(f)
        
        return questions, answers
    
    def evaluate_vlm(
        self,
        model_name: str,
        config: str,
        scenes: List[str]
    ) -> Dict:
        """Evaluate a VLM model across scenes."""
        print(f"\n{'='*60}")
        print(f"Evaluating VLM: {model_name.upper()}")
        print(f"{'='*60}")
        
        results = {
            "model": model_name,
            "model_full_name": MODEL_INFO.get(model_name, {}).get("name", model_name),
            "config": config,
            "scenes": {},
            "aggregate": {},
        }
        
        all_correct = []
        all_total = []
        
        # Load model once
        vlm = get_vlm_model(model_name, device=self.device)
        vlm.load()
        
        for scene in scenes:
            print(f"\n--- Scene: {scene} ---")
            
            # Get files
            files = self.find_scene_graph_files(config, scene)
            if not files["obj_json"]:
                print(f"  Skipping: No scene graph found for {config}/{scene}")
                continue
            
            # Load context
            context = load_scene_graph_context(
                files["obj_json"],
                files.get("edge_json")
            )
            
            # Load questions/answers
            questions, answers = self.load_space3d_data(scene)
            if not questions:
                print(f"  Skipping: No Space3D-Bench questions for {scene}")
                continue
            
            print(f"  Processing {len(questions)} questions...")
            
            scene_results = []
            correct = 0
            
            for q_id, question in questions.items():
                # Get ground truth
                gt_data = answers.get(q_id, {})
                if isinstance(gt_data, dict):
                    gt_answer = gt_data.get("answer", "")
                else:
                    gt_answer = str(gt_data)
                
                # Query model
                try:
                    predicted = vlm.query(context, question)
                except Exception as e:
                    predicted = f"ERROR: {e}"
                
                # Check match
                is_correct, confidence = check_answer_match(
                    predicted, gt_answer, question
                )
                
                if is_correct:
                    correct += 1
                
                scene_results.append({
                    "q_id": q_id,
                    "question": question,
                    "predicted": predicted,
                    "ground_truth": gt_answer,
                    "correct": is_correct,
                    "confidence": confidence,
                })
            
            accuracy = correct / len(questions) if questions else 0
            print(f"  Accuracy: {correct}/{len(questions)} = {accuracy:.2%}")
            
            results["scenes"][scene] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(questions),
                "details": scene_results,
            }
            
            all_correct.append(correct)
            all_total.append(len(questions))
        
        # Cleanup
        vlm.cleanup()
        
        # Aggregate
        total_correct = sum(all_correct)
        total_questions = sum(all_total)
        results["aggregate"] = {
            "accuracy": total_correct / total_questions if total_questions else 0,
            "correct": total_correct,
            "total": total_questions,
        }
        
        return results
    
    def evaluate_clip_retrieval(
        self,
        model_name: str,
        config: str,
        scenes: List[str],
        top_k: int = 5
    ) -> Dict:
        """Evaluate CLIP retrieval model across scenes."""
        print(f"\n{'='*60}")
        print(f"Evaluating CLIP: {model_name.upper()}")
        print(f"{'='*60}")
        
        results = {
            "model": model_name,
            "model_full_name": MODEL_INFO.get(model_name, {}).get("name", model_name),
            "config": config,
            "scenes": {},
            "aggregate": {},
        }
        
        all_top1_correct = []
        all_top5_correct = []
        all_total = []
        
        # Load CLIP model once
        clip_model = get_clip_model(model_name, device=self.device)
        clip_model.load()
        
        for scene in scenes:
            print(f"\n--- Scene: {scene} ---")
            
            # Get files
            files = self.find_scene_graph_files(config, scene)
            if not files["pkl"] and not files["obj_json"]:
                print(f"  Skipping: No data found for {config}/{scene}")
                continue
            
            # Load object database
            db = ObjectDatabase()
            if files["pkl"]:
                db.load_from_pkl(files["pkl"])
            else:
                db.load_from_json(files["obj_json"])
                db.compute_embeddings(clip_model)
            
            # Load questions/answers
            questions, answers = self.load_space3d_data(scene)
            if not questions:
                print(f"  Skipping: No Space3D-Bench questions for {scene}")
                continue
            
            print(f"  Processing {len(questions)} questions...")
            
            scene_results = []
            top1_correct = 0
            top5_correct = 0
            
            for q_id, question in questions.items():
                # Get ground truth
                gt_data = answers.get(q_id, {})
                if isinstance(gt_data, dict):
                    gt_answer = gt_data.get("answer", "")
                else:
                    gt_answer = str(gt_data)
                
                # Encode question and retrieve
                q_embedding = clip_model.encode_text([question])
                retrieved = db.retrieve_top_k(q_embedding, k=top_k)
                
                # Convert to dict format
                retrieved_dicts = [
                    {"idx": r[0], "score": r[1], "caption": r[2]}
                    for r in retrieved
                ]
                
                # Check Top-1
                is_top1, rank1 = check_top_k_match(retrieved_dicts, gt_answer, question, k=1)
                if is_top1:
                    top1_correct += 1
                
                # Check Top-5
                is_top5, rank5 = check_top_k_match(retrieved_dicts, gt_answer, question, k=5)
                if is_top5:
                    top5_correct += 1
                
                scene_results.append({
                    "q_id": q_id,
                    "question": question,
                    "top_retrieved": retrieved_dicts[:3],
                    "ground_truth": gt_answer,
                    "top1_correct": is_top1,
                    "top5_correct": is_top5,
                    "match_rank": rank5,
                })
            
            top1_acc = top1_correct / len(questions) if questions else 0
            top5_acc = top5_correct / len(questions) if questions else 0
            print(f"  Top-1: {top1_correct}/{len(questions)} = {top1_acc:.2%}")
            print(f"  Top-5: {top5_correct}/{len(questions)} = {top5_acc:.2%}")
            
            results["scenes"][scene] = {
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "top1_correct": top1_correct,
                "top5_correct": top5_correct,
                "total": len(questions),
                "details": scene_results,
            }
            
            all_top1_correct.append(top1_correct)
            all_top5_correct.append(top5_correct)
            all_total.append(len(questions))
        
        # Cleanup
        clip_model.cleanup()
        
        # Aggregate
        total_top1 = sum(all_top1_correct)
        total_top5 = sum(all_top5_correct)
        total_q = sum(all_total)
        
        results["aggregate"] = {
            "top1_accuracy": total_top1 / total_q if total_q else 0,
            "top5_accuracy": total_top5 / total_q if total_q else 0,
            "top1_correct": total_top1,
            "top5_correct": total_top5,
            "total": total_q,
        }
        
        return results
    
    def run_all(
        self,
        vlm_models: List[str],
        clip_models: List[str],
        config: str,
        scenes: List[str]
    ):
        """Run all evaluations."""
        # VLM evaluations
        for model in vlm_models:
            try:
                results = self.evaluate_vlm(model, config, scenes)
                self.results[f"vlm_{model}"] = results
            except Exception as e:
                print(f"Error evaluating {model}: {e}")
        
        # CLIP evaluations
        for model in clip_models:
            try:
                results = self.evaluate_clip_retrieval(model, config, scenes)
                self.results[f"clip_{model}"] = results
            except Exception as e:
                print(f"Error evaluating {model}: {e}")
    
    def generate_report(self) -> str:
        """Generate Markdown report with tables."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = f"""# Space3D-Bench VQA Evaluation Report

Generated: {timestamp}

## Overview

This report evaluates VLM and CLIP models on the Space3D-Bench spatial question answering benchmark.

---

## VLM Model Results

### Per-Scene Accuracy

| Model | Scene | Accuracy | Correct | Total |
|-------|-------|----------|---------|-------|
"""
        
        for key, results in self.results.items():
            if not key.startswith("vlm_"):
                continue
            model_name = results.get("model_full_name", results.get("model", "Unknown"))
            for scene, scene_data in results.get("scenes", {}).items():
                acc = scene_data.get("accuracy", 0)
                correct = scene_data.get("correct", 0)
                total = scene_data.get("total", 0)
                md += f"| {model_name} | {scene} | {acc:.2%} | {correct} | {total} |\n"
        
        md += """
### Aggregate VLM Results

| Model | Accuracy ↑ | Correct | Total |
|-------|------------|---------|-------|
"""
        
        for key, results in self.results.items():
            if not key.startswith("vlm_"):
                continue
            model_name = results.get("model_full_name", results.get("model", "Unknown"))
            agg = results.get("aggregate", {})
            acc = agg.get("accuracy", 0)
            correct = agg.get("correct", 0)
            total = agg.get("total", 0)
            md += f"| {model_name} | **{acc:.2%}** | {correct} | {total} |\n"
        
        md += """
---

## CLIP Retrieval Results

### Per-Scene Accuracy

| Model | Scene | Top-1 ↑ | Top-5 ↑ | Total |
|-------|-------|---------|---------|-------|
"""
        
        for key, results in self.results.items():
            if not key.startswith("clip_"):
                continue
            model_name = results.get("model_full_name", results.get("model", "Unknown"))
            for scene, scene_data in results.get("scenes", {}).items():
                top1 = scene_data.get("top1_accuracy", 0)
                top5 = scene_data.get("top5_accuracy", 0)
                total = scene_data.get("total", 0)
                md += f"| {model_name} | {scene} | {top1:.2%} | {top5:.2%} | {total} |\n"
        
        md += """
### Aggregate CLIP Results

| Model | Top-1 Accuracy ↑ | Top-5 Accuracy ↑ | Total |
|-------|------------------|------------------|-------|
"""
        
        for key, results in self.results.items():
            if not key.startswith("clip_"):
                continue
            model_name = results.get("model_full_name", results.get("model", "Unknown"))
            agg = results.get("aggregate", {})
            top1 = agg.get("top1_accuracy", 0)
            top5 = agg.get("top5_accuracy", 0)
            total = agg.get("total", 0)
            md += f"| {model_name} | **{top1:.2%}** | **{top5:.2%}** | {total} |\n"
        
        md += """
---

## Summary

### Best Performing Models

| Category | Model | Score |
|----------|-------|-------|
"""
        
        # Find best VLM
        best_vlm = None
        best_vlm_score = 0
        for key, results in self.results.items():
            if key.startswith("vlm_"):
                score = results.get("aggregate", {}).get("accuracy", 0)
                if score > best_vlm_score:
                    best_vlm_score = score
                    best_vlm = results.get("model_full_name", "Unknown")
        
        if best_vlm:
            md += f"| Best VLM | {best_vlm} | {best_vlm_score:.2%} |\n"
        
        # Find best CLIP
        best_clip = None
        best_clip_score = 0
        for key, results in self.results.items():
            if key.startswith("clip_"):
                score = results.get("aggregate", {}).get("top5_accuracy", 0)
                if score > best_clip_score:
                    best_clip_score = score
                    best_clip = results.get("model_full_name", "Unknown")
        
        if best_clip:
            md += f"| Best CLIP (Top-5) | {best_clip} | {best_clip_score:.2%} |\n"
        
        md += """
---

*Generated by run_vqa_eval.py*
"""
        
        return md
    
    def save_results(self):
        """Save all results to files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw JSON
        json_path = self.output_dir / "vqa_results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2, default=float)
        print(f"JSON results saved to: {json_path}")
        
        # Save markdown report
        md_path = self.output_dir / "VQA_RESULTS.md"
        report = self.generate_report()
        with open(md_path, "w") as f:
            f.write(report)
        print(f"Markdown report saved to: {md_path}")
        
        # Print report
        print("\n" + "="*60)
        print(report)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Space3D-Bench VQA evaluation")
    parser.add_argument(
        "--scene_graphs_root", type=str, required=True,
        help="Root directory containing config folders (oracle/, qwen/, paligemma/)"
    )
    parser.add_argument(
        "--space3d_root", type=str, required=True,
        help="Path to Space3D-Bench data/ directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./vqa_results",
        help="Output directory for results"
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
        "--device", type=str, default="cuda",
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    evaluator = VQAEvaluator(
        scene_graphs_root=Path(args.scene_graphs_root),
        space3d_root=Path(args.space3d_root),
        output_dir=Path(args.output_dir),
        device=args.device,
    )
    
    evaluator.run_all(
        vlm_models=args.vlm_models,
        clip_models=args.clip_models,
        config=args.config,
        scenes=args.scenes,
    )
    
    evaluator.save_results()


if __name__ == "__main__":
    main()
