#!/usr/bin/env python3
"""
YOLO Detection + Sentence-Transformers Query Script
====================================================

Baseline that uses ONLY raw YOLO detection class names (no VLM captions, no CLIP)
with sentence-transformers for semantic matching.

This provides a "detection-only" baseline to compare against full scene graph methods.

Uses: all-MiniLM-L6-v2 (lightweight, 22M params)

Usage:
    python yolo_load_and_query.py \
        --detections_path /path/to/detections_folder \
        --questions_path /path/to/questions.json \
        --output_path /path/to/answers.json
"""

import argparse
import gzip
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = "all-MiniLM-L6-v2"


# =============================================================================
# Sentence Transformer Wrapper
# =============================================================================

class SentenceTransformerRetriever:
    """
    Simple retrieval using sentence-transformers on YOLO class labels.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        
        # Object database
        self.objects: List[Dict] = []
        self.labels: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def load_model(self):
        """Load sentence-transformers model."""
        print(f"[SentenceTransformers] Loading {self.model_name}...")
        
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"[SentenceTransformers] Model loaded on {self.device}")
    
    def load_detections_from_folder(self, detections_path: Path) -> int:
        """
        Load YOLO detections from a folder of pkl.gz files.
        
        Expected structure:
            detections_path/
                frame_000000.pkl.gz
                frame_000010.pkl.gz
                ...
        
        Each pkl contains detection info including class labels.
        """
        detections_path = Path(detections_path)
        if not detections_path.exists():
            raise FileNotFoundError(f"Detections path not found: {detections_path}")
        
        print(f"[YOLO] Loading detections from {detections_path}...")
        
        all_classes = set()
        all_objects = []
        
        # Find all detection files
        pkl_files = sorted(detections_path.glob("*.pkl.gz"))
        if not pkl_files:
            pkl_files = sorted(detections_path.glob("*.pkl"))
        
        for pkl_file in pkl_files:
            try:
                if pkl_file.suffix == '.gz':
                    with gzip.open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                else:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                
                # Extract class IDs and labels
                if isinstance(data, dict):
                    class_ids = data.get('class_id', [])
                    classes_arr = data.get('classes', [])
                    
                    if hasattr(class_ids, '__len__') and hasattr(classes_arr, '__len__'):
                        for cid in class_ids:
                            if 0 <= cid < len(classes_arr):
                                class_name = classes_arr[cid]
                                all_classes.add(class_name)
                                
            except Exception as e:
                print(f"  Warning: Failed to load {pkl_file}: {e}")
                continue
        
        # Build unique object list from detected classes
        for class_name in sorted(all_classes):
            all_objects.append({
                "class_name": class_name,
                "label": class_name,
                "source": "yolo_detection"
            })
        
        self.objects = all_objects
        self.labels = [obj["label"] for obj in all_objects]
        
        print(f"[YOLO] Found {len(self.labels)} unique object classes")
        return len(self.labels)
    
    def load_detections_from_config_params(self, config_params_path: Path) -> int:
        """
        Alternative: Load from config_params_detections.json which lists detected classes.
        """
        if not config_params_path.exists():
            raise FileNotFoundError(f"Config params not found: {config_params_path}")
        
        print(f"[YOLO] Loading from config params: {config_params_path}")
        
        with open(config_params_path, 'r') as f:
            config = json.load(f)
        
        # Try to extract classes
        classes = config.get('classes', config.get('class_names', []))
        
        if not classes:
            # Fallback to common indoor object classes
            classes = [
                "chair", "table", "sofa", "bed", "desk", "cabinet", "shelf",
                "lamp", "plant", "tv", "monitor", "window", "door", "rug",
                "pillow", "blanket", "book", "cup", "bottle", "clock"
            ]
            print(f"  Warning: No classes in config, using defaults")
        
        self.objects = [{"class_name": c, "label": c} for c in classes]
        self.labels = classes
        
        print(f"[YOLO] Loaded {len(self.labels)} classes")
        return len(self.labels)
    
    def load_from_obj_json(self, obj_json_path: Path) -> int:
        """
        Fallback: Extract just class names from obj_json (ignoring captions).
        """
        if not obj_json_path.exists():
            raise FileNotFoundError(f"obj_json not found: {obj_json_path}")
        
        print(f"[YOLO-fallback] Loading class names from {obj_json_path}...")
        
        with open(obj_json_path, 'r') as f:
            data = json.load(f)
        
        classes = set()
        for obj_id, obj_data in data.items():
            class_name = obj_data.get('class_name', 'object')
            classes.add(class_name)
        
        self.objects = [{"class_name": c, "label": c} for c in sorted(classes)]
        self.labels = [obj["label"] for obj in self.objects]
        
        print(f"[YOLO-fallback] Found {len(self.labels)} unique classes")
        return len(self.labels)
    
    def compute_embeddings(self):
        """Compute embeddings for all object labels."""
        if not self.labels:
            print("[SentenceTransformers] Warning: No labels to embed")
            return
        
        print(f"[SentenceTransformers] Encoding {len(self.labels)} labels...")
        self.embeddings = self.model.encode(
            self.labels,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        print(f"[SentenceTransformers] Embeddings shape: {self.embeddings.shape}")
    
    def query(self, question: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Query the object database with a question.
        
        Returns:
            List of (index, similarity, label) tuples
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        # Encode question
        q_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Compute similarities
        similarities = np.dot(self.embeddings, q_embedding)
        
        # Get top-k
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        results = []
        for idx in top_indices[:top_k]:
            results.append((
                int(idx),
                float(similarities[idx]),
                self.labels[idx]
            ))
        
        return results
    
    def cleanup(self):
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self.embeddings = None
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear HuggingFace cache for this model
        import gc
        gc.collect()
        
        print("[SentenceTransformers] Cleanup complete")


# =============================================================================
# Query Processing
# =============================================================================

def process_questions(
    retriever: SentenceTransformerRetriever,
    questions: Dict[str, str],
    top_k: int = 5
) -> Dict[str, Any]:
    """Process all questions and return results."""
    results = {}
    
    for q_id, question in questions.items():
        retrieved = retriever.query(question, top_k=top_k)
        
        if retrieved:
            top_label = retrieved[0][2]
            top_score = retrieved[0][1]
            answer = f"Based on detected objects: {top_label}"
        else:
            top_label = ""
            top_score = 0.0
            answer = "No matching objects found."
        
        results[q_id] = {
            "answer": answer,
            "top_object": top_label,
            "confidence": top_score,
            "top_k_objects": [
                {"rank": i+1, "label": r[2], "score": r[1]}
                for i, r in enumerate(retrieved)
            ]
        }
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLO + Sentence-Transformers baseline for object retrieval"
    )
    parser.add_argument(
        "--detections", type=str, default=None,
        help="Path to detections folder containing pkl.gz files"
    )
    parser.add_argument(
        "--obj_json", type=str, default=None,
        help="Fallback: Path to obj_json to extract class names only"
    )
    parser.add_argument(
        "--questions", type=str, required=True,
        help="Path to questions JSON file"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save output answers"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Sentence-transformers model name"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of objects to retrieve"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.detections is None and args.obj_json is None:
        raise ValueError("Must provide either --detections or --obj_json")
    
    # Initialize retriever
    retriever = SentenceTransformerRetriever(
        model_name=args.model,
        device=args.device
    )
    
    # Load model
    retriever.load_model()
    
    # Load objects
    if args.detections:
        retriever.load_detections_from_folder(Path(args.detections))
    else:
        retriever.load_from_obj_json(Path(args.obj_json))
    
    # Compute embeddings
    retriever.compute_embeddings()
    
    # Load questions
    print(f"\n=== Loading Questions ===")
    with open(args.questions, 'r') as f:
        questions = json.load(f)
    
    # Handle different question formats
    if isinstance(questions, dict) and "complex_queries" in questions:
        # Complex queries format
        queries_list = questions["complex_queries"]
        questions = {q["id"]: q["query"] for q in queries_list}
    
    print(f"Loaded {len(questions)} questions")
    
    # Process questions
    print(f"\n=== Processing Questions ===")
    results = process_questions(retriever, questions, top_k=args.top_k)
    
    # Show sample results
    for q_id, result in list(results.items())[:3]:
        print(f"  Q: {questions[q_id][:50]}...")
        print(f"  A: {result['top_object']} (score: {result['confidence']:.3f})")
    
    # Save results
    print(f"\n=== Saving Results ===")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    # Cleanup
    retriever.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()
