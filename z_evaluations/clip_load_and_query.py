#!/usr/bin/env python3
"""
CLIP Load and Query Script for Space3D-Bench VQA Evaluation
============================================================

Uses CLIP text-to-embedding similarity for retrieval-based QA.
For questions like "What color is the sofa?", this script:
1. Encodes the question with CLIP text encoder
2. Retrieves top-k most similar objects by CLIP embedding
3. Returns object captions as potential answers

Supported CLIP Models:
- MobileCLIP2-S3 (dfndr2b) - lightweight, fast
- PE-Core-T-16-384 - efficient transformer
- TinyCLIP-ViT-8M-16-Text-3M-YFCC15M - tiny/fast

Usage:
    python clip_load_and_query.py \
        --model mobileclip \
        --pkl_path /path/to/pcd_*.pkl.gz \
        --questions_path /path/to/questions.json \
        --output_path /path/to/answers.json \
        --top_k 5
"""

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

import numpy as np

# =============================================================================
# CLIP Model Configurations
# =============================================================================

CLIP_CONFIGS = {
    "mobileclip": {
        "name": "MobileCLIP2-S3",
        "pretrained": "dfndr2b",
        "library": "open_clip",
    },
    "pecore": {
        "name": "hf-hub:timm/PE-Core-T-16-384",
        "pretrained": None,
        "library": "open_clip",
    },
    "tinyclip": {
        "name": "hf-hub:wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        "pretrained": None,
        "library": "open_clip",
    },
    "vith14": {
        "name": "ViT-H-14",
        "pretrained": "laion2b_s32b_b79k",
        "library": "open_clip",
    },
}


# =============================================================================
# Base CLIP Interface
# =============================================================================

class BaseCLIP(ABC):
    """Abstract base class for CLIP models."""
    
    @abstractmethod
    def load(self):
        """Load the model."""
        pass
    
    @abstractmethod
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text to embeddings."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Release model resources."""
        pass


# =============================================================================
# OpenCLIP Implementation
# =============================================================================

class OpenCLIPModel(BaseCLIP):
    """OpenCLIP model wrapper supporting multiple architectures."""
    
    def __init__(
        self,
        model_name: str,
        pretrained: Optional[str] = None,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Load OpenCLIP model."""
        import open_clip
        import torch
        
        print(f"[OpenCLIP] Loading {self.model_name}...")
        
        if self.pretrained:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
            )
        else:
            # For HuggingFace hub models
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get tokenizer
        if self.model_name.startswith("hf-hub:"):
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
        else:
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
        
        print(f"[OpenCLIP] Model loaded on {self.device}")
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text to embeddings."""
        import torch
        import torch.nn.functional as F
        
        tokens = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features.cpu().numpy()
    
    def cleanup(self):
        """Release OpenCLIP resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[OpenCLIP] Cleanup complete")


# =============================================================================
# Object Database from PKL
# =============================================================================

class ObjectDatabase:
    """
    Stores objects and their CLIP embeddings for retrieval.
    """
    
    def __init__(self):
        self.objects: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.captions: List[str] = []
    
    def load_from_pkl(self, pkl_path: Path) -> int:
        """
        Load objects from pkl.gz file.
        
        Returns:
            Number of objects loaded
        """
        import torch
        import torch.nn.functional as F
        
        if not pkl_path.exists():
            raise FileNotFoundError(f"PKL file not found: {pkl_path}")
        
        print(f"[ObjectDB] Loading from {pkl_path}...")
        
        with gzip.open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        # Extract objects
        if isinstance(data, dict):
            objects_data = data.get('objects', [])
        else:
            objects_data = data
        
        embeddings = []
        valid_objects = []
        captions = []
        
        for obj in objects_data:
            # Get embedding
            emb = obj.get('clip_ft')
            if emb is None:
                continue
            
            if isinstance(emb, list):
                emb = np.array(emb)
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            
            # Normalize
            emb = F.normalize(emb.float().view(1, -1), dim=1)
            embeddings.append(emb.numpy())
            
            # Get caption
            caption = obj.get('object_caption') or obj.get('consolidated_caption', '')
            if isinstance(caption, list):
                caption = caption[0] if caption else ''
            class_name = obj.get('class_name', 'object')
            
            full_caption = f"[{class_name}] {caption}" if caption else f"[{class_name}]"
            captions.append(full_caption)
            
            valid_objects.append(obj)
        
        if len(embeddings) == 0:
            print("[ObjectDB] Warning: No valid embeddings found")
            return 0
        
        self.objects = valid_objects
        self.embeddings = np.vstack(embeddings)
        self.captions = captions
        
        print(f"[ObjectDB] Loaded {len(self.objects)} objects")
        return len(self.objects)
    
    def load_from_json(self, obj_json_path: Path) -> int:
        """
        Load objects from obj_json file (without embeddings).
        For use when PKL is not available.
        """
        if not obj_json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {obj_json_path}")
        
        print(f"[ObjectDB] Loading from {obj_json_path}...")
        
        with open(obj_json_path, "r") as f:
            data = json.load(f)
        
        self.objects = []
        self.captions = []
        
        for obj_id, obj_data in data.items():
            caption = obj_data.get('object_caption') or obj_data.get('consolidated_caption', '')
            if isinstance(caption, list):
                caption = caption[0] if caption else ''
            class_name = obj_data.get('class_name', 'object')
            
            full_caption = f"[{class_name}] {caption}" if caption else f"[{class_name}]"
            self.captions.append(full_caption)
            
            obj_data['_id'] = obj_id
            self.objects.append(obj_data)
        
        print(f"[ObjectDB] Loaded {len(self.objects)} objects (no embeddings)")
        return len(self.objects)
    
    def compute_embeddings(self, clip_model: BaseCLIP):
        """Compute embeddings for captions using CLIP text encoder."""
        if len(self.captions) == 0:
            return
        
        print(f"[ObjectDB] Computing embeddings for {len(self.captions)} captions...")
        self.embeddings = clip_model.encode_text(self.captions)
        print(f"[ObjectDB] Embeddings shape: {self.embeddings.shape}")
    
    def retrieve_top_k(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, float, str]]:
        """
        Retrieve top-k objects by cosine similarity.
        
        Returns:
            List of (index, similarity, caption) tuples
        """
        if self.embeddings is None:
            return []
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding.T).squeeze()
        
        # Get top-k indices
        if k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(similarities[idx]),
                self.captions[idx]
            ))
        
        return results


# =============================================================================
# Answer Generator
# =============================================================================

def generate_answer_from_retrieved(
    question: str,
    retrieved: List[Tuple[int, float, str]],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Generate an answer based on retrieved objects.
    
    Returns dict with:
    - answer: Combined answer from top objects
    - top_objects: List of retrieved object info
    - confidence: Average similarity score
    """
    if not retrieved:
        return {
            "answer": "Unable to find relevant objects.",
            "top_objects": [],
            "confidence": 0.0,
        }
    
    top = retrieved[:top_k]
    
    # Extract relevant captions
    captions = [r[2] for r in top]
    scores = [r[1] for r in top]
    
    # Build answer
    if len(captions) == 1:
        answer = f"Based on the scene, the answer involves: {captions[0]}"
    else:
        answer = f"Based on the scene, the relevant objects are: {'; '.join(captions[:3])}"
    
    return {
        "answer": answer,
        "top_objects": [
            {"idx": r[0], "score": r[1], "caption": r[2]}
            for r in top
        ],
        "confidence": float(np.mean(scores)),
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def get_clip_model(model_name: str, device: str = "cuda") -> BaseCLIP:
    """Factory function to get CLIP model by name."""
    model_name_lower = model_name.lower()
    
    if model_name_lower not in CLIP_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(CLIP_CONFIGS.keys())}")
    
    config = CLIP_CONFIGS[model_name_lower]
    
    return OpenCLIPModel(
        model_name=config["name"],
        pretrained=config["pretrained"],
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-based retrieval for Space3D-Bench VQA"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(CLIP_CONFIGS.keys()),
        help="CLIP model to use"
    )
    parser.add_argument(
        "--pkl", type=str, default=None,
        help="Path to pkl.gz file with object embeddings"
    )
    parser.add_argument(
        "--obj_json", type=str, default=None,
        help="Path to obj_json file (if pkl not available)"
    )
    parser.add_argument(
        "--questions", type=str, required=True,
        help="Path to questions.json file"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save output answers JSON"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of objects to retrieve per question"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.pkl is None and args.obj_json is None:
        raise ValueError("Must provide either --pkl or --obj_json")
    
    # Load CLIP model
    print(f"\n=== Loading {args.model.upper()} CLIP ===")
    clip_model = get_clip_model(args.model, args.device)
    clip_model.load()
    
    # Load object database
    print("\n=== Loading Object Database ===")
    db = ObjectDatabase()
    
    if args.pkl:
        db.load_from_pkl(Path(args.pkl))
    else:
        db.load_from_json(Path(args.obj_json))
        db.compute_embeddings(clip_model)
    
    # Load questions
    print("\n=== Loading Questions ===")
    with open(args.questions, "r") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")
    
    if args.limit:
        questions = dict(list(questions.items())[:args.limit])
        print(f"Limited to {len(questions)} questions")
    
    # Process questions
    print("\n=== Running Retrieval-based QA ===")
    results = {}
    
    for q_id, question in questions.items():
        print(f"  Q{q_id}: {question[:60]}...")
        
        # Encode question
        q_embedding = clip_model.encode_text([question])
        
        # Retrieve top-k objects
        retrieved = db.retrieve_top_k(q_embedding, k=args.top_k)
        
        # Generate answer
        answer_data = generate_answer_from_retrieved(question, retrieved, args.top_k)
        results[q_id] = answer_data
        
        print(f"    Top-1: {retrieved[0][2][:50]}... (score: {retrieved[0][1]:.3f})" if retrieved else "    No results")
    
    # Save results
    print(f"\n=== Saving Results ===")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save simple answers for evaluation
    simple_answers = {q_id: data["answer"] for q_id, data in results.items()}
    simple_path = output_path.with_name(output_path.stem + "_simple.json")
    with open(simple_path, "w") as f:
        json.dump(simple_answers, f, indent=2)
    
    print(f"Full results saved to: {output_path}")
    print(f"Simple answers saved to: {simple_path}")
    
    # Cleanup
    clip_model.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()
