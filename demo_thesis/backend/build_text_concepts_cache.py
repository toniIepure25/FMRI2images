#!/usr/bin/env python3
"""
Build text concept embeddings cache for semantic probe.

Generates CLIP ViT-L/14 text embeddings for 22 concepts
and saves them to local_backend_data/clip/text_concepts.npz.

Run once:
  python3 backend/build_text_concepts_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure venv or root site-packages available for open_clip
_venv = Path(__file__).resolve().parents[2] / ".venv" / "lib" / "python3.10" / "site-packages"
if _venv.exists() and str(_venv) not in sys.path:
    sys.path.insert(0, str(_venv))

import numpy as np

CONCEPTS = [
    "person", "animal", "vehicle", "indoor room", "outdoor scene",
    "forest", "road", "sky", "water", "food",
    "building", "face", "dog", "cat", "beach",
    "mountain", "city street", "natural scene", "object",
    "snow", "grass", "tree",
]

OUTPUT = Path("local_backend_data/clip/text_concepts.npz")


def main():
    try:
        import open_clip
        import torch
    except ImportError:
        print("ERROR: open_clip not available. Install: pip install open_clip_torch")
        sys.exit(1)

    print(f"Building CLIP ViT-L/14 text embeddings for {len(CONCEPTS)} concepts...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    texts = [f"a photo of a {c}" for c in CONCEPTS]
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        text_embeds = model.encode_text(tokens).cpu().numpy()

    norms = np.linalg.norm(text_embeds, axis=1, keepdims=True)
    text_embeds = text_embeds / (norms + 1e-8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(OUTPUT), embeddings=text_embeds.astype(np.float32), concepts=CONCEPTS)
    print(f"Done: {OUTPUT} ({text_embeds.shape[0]} × {text_embeds.shape[1]}D)")


if __name__ == "__main__":
    main()
