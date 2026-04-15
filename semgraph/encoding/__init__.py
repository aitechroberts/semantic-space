"""
semgraph.encoding — pluggable embedding encoder backends.

Factory
-------
``get_encoder(encoder_type, encoder_name, ...)`` — returns an
:class:`EmbeddingEncoder` instance.

Usage::

    from semgraph.encoding import get_encoder

    encoder = get_encoder("hf_clip", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    feats = encoder.encode_images(crops)   # (N, D) float32
    text  = encoder.encode_texts(labels)   # (N, D) float32 or None
    dim   = encoder.feat_dim               # int
"""

from semgraph.encoding.base import EmbeddingEncoder


def get_encoder(
    encoder_type: str,
    encoder_name: str,
    device: str = "cuda",
    **kwargs,
) -> EmbeddingEncoder:
    """Factory that returns the appropriate :class:`EmbeddingEncoder`.

    Uses if-chain with lazy imports to avoid pulling in heavy
    dependencies (``transformers``, ``open_clip``) until they are needed.

    Parameters
    ----------
    encoder_type : str
        One of ``"hf_clip"``, ``"hf_siglip"``, ``"open_clip"``,
        ``"vlm_vision"``.
    encoder_name : str
        Model identifier.  Format depends on *encoder_type*:

        - ``hf_clip`` / ``hf_siglip``: HuggingFace model ID
        - ``open_clip``: ``"arch:pretrained"`` or ``"hf-hub:org/model"``
        - ``vlm_vision``: HuggingFace model ID
    device : str
        Target device (default ``"cuda"``).
    **kwargs
        Forwarded to the encoder constructor (e.g. ``dtype``,
        ``batch_size``, ``use_proj``).
    """
    if encoder_type == "hf_clip":
        from semgraph.encoding.hf_clip import HFCLIPEncoder

        return HFCLIPEncoder(encoder_name, device=device, **kwargs)

    if encoder_type == "hf_siglip":
        from semgraph.encoding.hf_siglip import HFSiglipEncoder

        return HFSiglipEncoder(encoder_name, device=device, **kwargs)

    if encoder_type == "open_clip":
        from semgraph.encoding.open_clip_enc import OpenCLIPEncoder

        return OpenCLIPEncoder(encoder_name, device=device, **kwargs)

    if encoder_type == "vlm_vision":
        from semgraph.encoding.vlm_vision import VLMVisionEncoder

        use_proj = kwargs.pop("use_proj", False)
        return VLMVisionEncoder(
            encoder_name, device=device, use_proj=use_proj, **kwargs,
        )

    raise ValueError(
        f"Unknown encoder_type '{encoder_type}'. "
        f"Valid: hf_clip, hf_siglip, open_clip, vlm_vision"
    )


__all__ = [
    "EmbeddingEncoder",
    "get_encoder",
]
