"""
Universal VLM API Client for vLLM-served models.

Talks to any VLM behind an OpenAI-compatible /v1/chat/completions endpoint
(vLLM, SGLang, or OpenAI itself). Replaces model-specific clients like
vlm_qwen.py, vlm_gemma.py, etc. with a single class.
"""

import json
import re
import ast
import base64
import time
import logging
from io import BytesIO
from typing import List, Dict, Optional, Tuple, Any

from PIL import Image
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


# =============================================================================
# Response Parsing Utilities (shared with vlm_qwen.py)
# =============================================================================

def _clean_markdown_json(text: str) -> str:
    """Extracts JSON content from Markdown code blocks if present."""
    text = text.strip()
    pattern = r"```(?:json)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text


def _safe_eval_llm_output(text: str) -> Any:
    """Robustly parses LLM output into Python objects (Dict/List)."""
    cleaned_text = _clean_markdown_json(text)

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(cleaned_text)
    except (ValueError, SyntaxError):
        pass

    try:
        start_idx = -1
        end_idx = -1
        if "[" in cleaned_text and "]" in cleaned_text:
            start_idx = cleaned_text.find("[")
            end_idx = cleaned_text.rfind("]") + 1
        elif "{" in cleaned_text and "}" in cleaned_text:
            start_idx = cleaned_text.find("{")
            end_idx = cleaned_text.rfind("}") + 1

        if start_idx != -1 and end_idx != -1:
            candidate = cleaned_text[start_idx:end_idx]
            try:
                return json.loads(candidate)
            except Exception:
                return ast.literal_eval(candidate)
    except Exception:
        pass
    return text


# =============================================================================
# Server Health Check
# =============================================================================

def wait_for_server(base_url: str, timeout: int = 120, poll_interval: float = 2.0):
    """
    Polls the vLLM server health endpoint until it responds 200 or timeout.
    base_url should be like 'http://localhost:8000/v1'.
    """
    import urllib.request
    import urllib.error

    health_url = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/health"
    logger.info(f"[VLM-API] Waiting for server at {health_url} (timeout={timeout}s)...")

    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("[VLM-API] Server is ready.")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(poll_interval)

    raise TimeoutError(
        f"[VLM-API] Server at {health_url} did not become ready within {timeout}s"
    )


# =============================================================================
# Universal VLM API Client
# =============================================================================

class VLMAPIClient:
    """
    Client for VLMs served behind an OpenAI-compatible API (vLLM, SGLang, etc.).
    Drop-in replacement for Qwen3VLClient -- same method signatures.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        prompts: Optional[Any] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        jpeg_quality: int = 85,
        timeout: float = 120.0,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout

        if prompts is None:
            self.prompts = {}
        else:
            try:
                self.prompts = OmegaConf.to_container(prompts, resolve=True)
            except Exception:
                self.prompts = dict(prompts)

        logger.info(
            f"[VLM-API] Client ready: model={model_name}, url={base_url}"
        )

    def _encode_image(self, image: Image.Image) -> str:
        """Encode a PIL Image to a base64 JPEG string."""
        buf = BytesIO()
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        rgb.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send an image + text prompt to the VLM and return the text response."""
        tokens = max_tokens or self.max_tokens
        b64 = self._encode_image(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=tokens,
                temperature=self.temperature,
                timeout=self.timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[VLM-API] Generation error: {e}")
            return ""

    def caption_objects_with_labels(
        self,
        image: Image.Image,
        labels: List[str],
        caption_system_prompt: str,
        captions_with_labels_template: str,
    ) -> List[Dict[str, str]]:
        """Generates captions for labeled objects in a single VLM call."""
        labels_str = "\n".join(labels)
        user_content = captions_with_labels_template.replace("{labels}", labels_str)
        full_prompt = f"{caption_system_prompt}\n\n{user_content}"

        response = self.generate(image, full_prompt)
        parsed = _safe_eval_llm_output(response)

        results = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    results.append({
                        "id": str(item.get("id", "")),
                        "name": str(item.get("name", "")),
                        "caption": str(
                            item.get("caption", item.get("description", ""))
                        ),
                    })

        if not results:
            for label in labels:
                parts = label.split(":", 1)
                obj_id = parts[0].strip() if len(parts) > 0 else "0"
                obj_name = parts[1].strip() if len(parts) > 1 else label
                results.append({
                    "id": obj_id,
                    "name": obj_name,
                    "caption": f"A {obj_name}.",
                })

        return results

    def infer_relations_with_labels(
        self,
        image: Image.Image,
        labels: List[str],
        relation_system_prompt: str,
        relations_with_labels_template: str,
    ) -> List[Tuple[str, str, str]]:
        """Generates spatial relationships between labeled objects."""
        labels_str = ", ".join(labels)
        user_content = relations_with_labels_template.replace("{labels}", labels_str)
        full_prompt = f"{relation_system_prompt}\n\n{user_content}"

        response = self.generate(image, full_prompt)
        parsed = _safe_eval_llm_output(response)

        edges = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    edges.append((str(item[0]), str(item[1]), str(item[2])))
                elif isinstance(item, dict):
                    subj = item.get("subject", item.get("object1", ""))
                    rel = item.get("relation", item.get("predicate", ""))
                    obj = item.get("object", item.get("object2", ""))
                    if subj and rel and obj:
                        edges.append((str(subj), str(rel), str(obj)))
        return edges

    def cleanup(self):
        """No-op -- the vLLM container manages model lifecycle."""
        logger.info("[VLM-API] Client cleanup (no-op, container manages model).")


# =============================================================================
# Caption Consolidation (module-level, same interface as vlm_qwen.py)
# =============================================================================

def consolidate_captions(client: VLMAPIClient, captions: List[Dict]) -> str:
    """Consolidate multiple captions for the same object into one sentence."""
    valid_caps = [c["caption"] for c in captions if c.get("caption")]
    if not valid_caps:
        return "Unknown object."

    captions_block = "\n".join([f"- {c}" for c in valid_caps[:10]])
    template = client.prompts.get(
        "consolidate_prompt", "Summarize these: {captions}"
    )
    prompt = template.replace("{captions}", captions_block)

    dummy_image = Image.new("RGB", (28, 28), color=(0, 0, 0))
    response = client.generate(dummy_image, prompt, max_tokens=128)

    if "{" in response and "}" in response:
        parsed = _safe_eval_llm_output(response)
        if isinstance(parsed, dict) and "consolidated_caption" in parsed:
            return parsed["consolidated_caption"]

    return _clean_markdown_json(response)
