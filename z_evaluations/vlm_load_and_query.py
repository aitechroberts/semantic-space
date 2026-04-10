#!/usr/bin/env python3
"""
VLM Load and Query Script for Space3D-Bench VQA Evaluation
==========================================================

Loads scene graph context from obj_json/edge_json files and queries VLMs
to answer Space3D-Bench questions.

Supported Models:
- Qwen3-VL-2B-Instruct (HuggingFace)
- PaliGemma2-3b-mix-224 (HuggingFace)
- GPT-4-mini (OpenAI API)

Usage:
    python vlm_load_and_query.py \
        --model qwen \
        --scene_graph_path /path/to/scene/obj_json.json \
        --questions_path /path/to/questions.json \
        --output_path /path/to/answers.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

# =============================================================================
# Base VLM Interface
# =============================================================================

class BaseVLM(ABC):
    """Abstract base class for VLM models."""
    
    @abstractmethod
    def load(self):
        """Load the model."""
        pass
    
    @abstractmethod
    def query(self, context: str, question: str) -> str:
        """Query the model with context and question."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Release model resources."""
        pass


# =============================================================================
# Qwen3-VL Implementation
# =============================================================================

class Qwen3VL(BaseVLM):
    """Qwen3-VL-2B-Instruct model wrapper."""
    
    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
    
    def load(self):
        """Load Qwen3-VL model."""
        print(f"[Qwen3-VL] Loading {self.MODEL_ID}...")
        
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch
        
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        print(f"[Qwen3-VL] Model loaded on {self.device}")
    
    def query(self, context: str, question: str) -> str:
        """Query Qwen3-VL with scene context and question."""
        import torch
        
        # Build prompt for text-only QA (no image)
        system_prompt = """You are a helpful assistant answering questions about a 3D scene.
You are given a scene description containing objects and their relationships.
Answer the question based ONLY on the provided scene information.
Be concise and specific."""
        
        user_message = f"""Scene Description:
{context}

Question: {question}

Answer:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        # Format for Qwen
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        
        # Decode response
        response = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return response.strip()
    
    def cleanup(self):
        """Release Qwen3-VL resources."""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[Qwen3-VL] Cleanup complete")


# =============================================================================
# PaliGemma Implementation
# =============================================================================

class PaliGemma2(BaseVLM):
    """PaliGemma2-3b-mix-224 model wrapper."""
    
    MODEL_ID = "google/paligemma2-3b-mix-224"
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
    
    def load(self):
        """Load PaliGemma2 model."""
        print(f"[PaliGemma2] Loading {self.MODEL_ID}...")
        
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        import torch
        
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        
        print(f"[PaliGemma2] Model loaded on {self.device}")
    
    def query(self, context: str, question: str) -> str:
        """Query PaliGemma2 with scene context and question."""
        import torch
        from PIL import Image
        
        # PaliGemma requires an image, create a dummy white image for text-only QA
        dummy_image = Image.new('RGB', (224, 224), color='white')
        
        # Build prompt
        prompt = f"""Scene: {context[:1500]}
Question: {question}
Answer:"""
        
        inputs = self.processor(
            text=prompt,
            images=dummy_image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        
        response = self.processor.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def cleanup(self):
        """Release PaliGemma2 resources."""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[PaliGemma2] Cleanup complete")


# =============================================================================
# OpenAI GPT Implementation
# =============================================================================

class OpenAIGPT(BaseVLM):
    """OpenAI GPT-4-mini model wrapper."""
    
    MODEL_ID = "gpt-4o-mini"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
    
    def load(self):
        """Initialize OpenAI client."""
        print(f"[OpenAI] Initializing {self.MODEL_ID}...")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            print(f"[OpenAI] Client initialized")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def query(self, context: str, question: str) -> str:
        """Query GPT-4-mini with scene context and question."""
        
        system_prompt = """You are a helpful assistant answering questions about a 3D scene.
You are given a scene description containing objects and their relationships.
Answer the question based ONLY on the provided scene information.
Be concise and specific."""
        
        user_message = f"""Scene Description:
{context}

Question: {question}"""
        
        response = self.client.chat.completions.create(
            model=self.MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=256,
            temperature=0,
        )
        
        return response.choices[0].message.content.strip()
    
    def cleanup(self):
        """Nothing to cleanup for API client."""
        print("[OpenAI] Cleanup complete")


# =============================================================================
# Scene Graph Context Builder
# =============================================================================

def load_scene_graph_context(
    obj_json_path: Path,
    edge_json_path: Optional[Path] = None,
    max_objects: int = 50,
    max_edges: int = 100,
) -> str:
    """
    Build a textual context from scene graph JSON files.
    
    Args:
        obj_json_path: Path to obj_json file
        edge_json_path: Optional path to edge_json file
        max_objects: Maximum objects to include
        max_edges: Maximum edges to include
    
    Returns:
        Formatted scene description string
    """
    context_parts = []
    
    # Load objects
    if obj_json_path.exists():
        with open(obj_json_path, "r") as f:
            objects = json.load(f)
        
        context_parts.append("=== OBJECTS IN SCENE ===")
        
        obj_items = list(objects.items())[:max_objects]
        for obj_id, obj_data in obj_items:
            class_name = obj_data.get("class_name", "unknown")
            caption = obj_data.get("object_caption") or obj_data.get("consolidated_caption", "")
            
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            
            # Include position if available
            centroid = obj_data.get("centroid", obj_data.get("bbox_center", None))
            pos_str = ""
            if centroid:
                if isinstance(centroid, (list, tuple)) and len(centroid) >= 3:
                    pos_str = f" at position ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
            
            if caption:
                context_parts.append(f"- Object {obj_id} [{class_name}]{pos_str}: {caption}")
            else:
                context_parts.append(f"- Object {obj_id} [{class_name}]{pos_str}")
    
    # Load edges/relationships
    if edge_json_path and edge_json_path.exists():
        with open(edge_json_path, "r") as f:
            edges = json.load(f)
        
        context_parts.append("\n=== RELATIONSHIPS ===")
        
        if isinstance(edges, dict):
            edge_items = list(edges.items())[:max_edges]
            for edge_id, edge_data in edge_items:
                obj1 = edge_data.get("obj1_class", f"Object {edge_data.get('obj1_idx', '?')}")
                obj2 = edge_data.get("obj2_class", f"Object {edge_data.get('obj2_idx', '?')}")
                relation = edge_data.get("relation", "related_to")
                context_parts.append(f"- {obj1} {relation} {obj2}")
        elif isinstance(edges, list):
            for edge_data in edges[:max_edges]:
                obj1 = edge_data.get("obj1_class", f"Object {edge_data.get('obj1_idx', '?')}")
                obj2 = edge_data.get("obj2_class", f"Object {edge_data.get('obj2_idx', '?')}")
                relation = edge_data.get("relation", "related_to")
                context_parts.append(f"- {obj1} {relation} {obj2}")
    
    return "\n".join(context_parts)


def load_questions(questions_path: Path) -> Dict[str, str]:
    """Load questions from Space3D-Bench format."""
    with open(questions_path, "r") as f:
        return json.load(f)


def load_ground_truth(answers_path: Path) -> Dict[str, Any]:
    """Load ground truth answers from Space3D-Bench format."""
    if not answers_path.exists():
        return {}
    with open(answers_path, "r") as f:
        return json.load(f)


# =============================================================================
# Model Factory
# =============================================================================

def get_model(model_name: str, **kwargs) -> BaseVLM:
    """Factory function to get VLM model by name."""
    models = {
        "qwen": Qwen3VL,
        "qwen3": Qwen3VL,
        "paligemma": PaliGemma2,
        "paligemma2": PaliGemma2,
        "gpt4": OpenAIGPT,
        "gpt": OpenAIGPT,
        "openai": OpenAIGPT,
    }
    
    model_name_lower = model_name.lower()
    if model_name_lower not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name_lower](**kwargs)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Query VLMs with scene graph context for Space3D-Bench VQA"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["qwen", "paligemma", "gpt4"],
        help="VLM model to use"
    )
    parser.add_argument(
        "--obj_json", type=str, required=True,
        help="Path to obj_json file containing scene objects"
    )
    parser.add_argument(
        "--edge_json", type=str, default=None,
        help="Optional path to edge_json file containing relationships"
    )
    parser.add_argument(
        "--questions", type=str, required=True,
        help="Path to questions.json file (Space3D-Bench format)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save output answers JSON"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for local models (cuda/cpu)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions to process"
    )
    
    args = parser.parse_args()
    
    # Load scene context
    print("\n=== Loading Scene Graph ===")
    obj_json_path = Path(args.obj_json)
    edge_json_path = Path(args.edge_json) if args.edge_json else None
    
    context = load_scene_graph_context(obj_json_path, edge_json_path)
    print(f"Context length: {len(context)} characters")
    print(f"Preview:\n{context[:500]}...")
    
    # Load questions
    print("\n=== Loading Questions ===")
    questions = load_questions(Path(args.questions))
    print(f"Loaded {len(questions)} questions")
    
    if args.limit:
        questions = dict(list(questions.items())[:args.limit])
        print(f"Limited to {len(questions)} questions")
    
    # Initialize model
    print(f"\n=== Initializing {args.model.upper()} ===")
    model_kwargs = {"device": args.device} if args.model != "gpt4" else {}
    vlm = get_model(args.model, **model_kwargs)
    vlm.load()
    
    # Query model
    print("\n=== Running VQA ===")
    answers = {}
    
    for q_id, question in questions.items():
        print(f"  Q{q_id}: {question[:60]}...")
        
        try:
            answer = vlm.query(context, question)
            answers[q_id] = answer
            print(f"    A: {answer[:80]}...")
        except Exception as e:
            print(f"    Error: {e}")
            answers[q_id] = f"ERROR: {str(e)}"
    
    # Save outputs
    print(f"\n=== Saving Results ===")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(answers, f, indent=2)
    
    print(f"Saved {len(answers)} answers to: {output_path}")
    
    # Cleanup
    vlm.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()
