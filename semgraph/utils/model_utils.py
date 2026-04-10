"""
Model utility functions for the detection pipeline.

Contains:
  - ``compute_tinyclip_features_batched`` — CLIP feature extraction via
    HuggingFace ``transformers.CLIPModel`` (moved from monolith)
  - ``compute_ft_vector_closeness_statistics`` — debugging helper for
    comparing feature vectors
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.distance import cosine


def compute_tinyclip_features_batched(
    image_rgb_bgr_fixed,
    detections,
    clip_model,
    clip_processor,
    classes,
    device: str,
    padding: int = 20,
):
    """Compute TinyCLIP features for each detection.

    Parameters
    ----------
    image_rgb_bgr_fixed : np.ndarray
        (H, W, 3) uint8 RGB image.
    detections : sv.Detections
        Supervision detections with ``xyxy`` and ``class_id``.
    clip_model : transformers.CLIPModel
        Loaded TinyCLIP model.
    clip_processor : transformers.CLIPProcessor
        Corresponding processor.
    classes : list[str]
        Vocabulary for text features.
    device : str
        Torch device string.
    padding : int
        Pixel padding around each crop.

    Returns
    -------
    image_crops : list[PIL.Image]
    image_feats : np.ndarray, shape (N, D)
    text_feats : np.ndarray, shape (N, D)
    """
    image = Image.fromarray(image_rgb_bgr_fixed)

    image_crops = []
    texts = []

    if detections.xyxy.shape[0] == 0:
        dim = clip_model.config.projection_dim
        return [], np.zeros((0, dim), dtype=np.float32), np.zeros((0, dim), dtype=np.float32)

    img_w, img_h = image.size

    for idx in range(len(detections.xyxy)):
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

        left_padding   = min(padding, x_min)
        top_padding    = min(padding, y_min)
        right_padding  = min(padding, img_w - x_max)
        bottom_padding = min(padding, img_h - y_max)

        x_min = x_min - left_padding
        y_min = y_min - top_padding
        x_max = x_max + right_padding
        y_max = y_max + bottom_padding

        crop = image.crop((x_min, y_min, x_max, y_max))
        image_crops.append(crop)

        class_id = int(detections.class_id[idx])
        texts.append(classes[class_id])

    image_inputs = clip_processor(
        images=image_crops,
        return_tensors="pt",
    ).to(device)

    text_inputs = clip_processor(
        text=texts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
        text_features = clip_model.get_text_features(**text_inputs)

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    image_feats = image_features.cpu().numpy()
    text_feats = text_features.cpu().numpy()

    return image_crops, image_feats, text_feats


def compute_ft_vector_closeness_statistics(unbatched, batched):
    """Print statistics comparing two sets of feature vectors (debugging helper)."""
    mad = []
    max_diff = []
    mrd = []
    cosine_sim = []

    for i in range(len(unbatched)):
        diff = np.abs(unbatched[i] - batched[i])
        mad.append(np.mean(diff))
        max_diff.append(np.max(diff))
        mrd.append(np.mean(diff / (np.abs(batched[i]) + 1e-8)))
        cosine_sim.append(1 - cosine(unbatched[i].flatten(), batched[i].flatten()))

    mad = np.array(mad)
    max_diff = np.array(max_diff)
    mrd = np.array(mrd)
    cosine_sim = np.array(cosine_sim)

    print(f"Mean Absolute Difference: {np.mean(mad)}")
    print(f"Maximum Absolute Difference: {np.max(max_diff)}")
    print(f"Mean Relative Difference: {np.mean(mrd)}")
    print(f"Mean Cosine Similarity: {np.mean(cosine_sim)}")
    print(f"Min Cosine Similarity: {np.min(cosine_sim)}")
