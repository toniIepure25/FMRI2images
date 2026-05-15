#!/usr/bin/env python3
"""Build a CLIP text embedding cache for semantic probe analysis.

The output cache is a real CLIP text-encoder artifact. If open_clip or model
weights are unavailable, this script fails clearly rather than fabricating
embeddings.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_CONCEPTS = [
    "person",
    "human face",
    "animal",
    "dog",
    "cat",
    "vehicle",
    "car",
    "indoor room",
    "outdoor landscape",
    "natural scene",
    "city street",
    "forest",
    "mountain",
    "beach",
    "water",
    "sky",
    "building",
    "kitchen",
    "living room",
    "food",
    "road",
    "text sign",
    "object",
    "tool",
    "furniture",
    "sports",
    "human body",
    "grass",
    "urban scene",
]

DEFAULT_TEMPLATES = [
    "a photo of a {concept}",
    "an image of a {concept}",
    "a natural image containing {concept}",
]


def _load_lines(path: str | None, defaults: Sequence[str]) -> list[str]:
    if path is None:
        return list(defaults)
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _format_prompt(template: str, concept: str) -> str:
    if "{concept}" in template:
        return template.format(concept=concept)
    return template.format(concept)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(norms > eps, x / norms, 0.0)


def build_concept_embeddings(
    prompt_embeddings: np.ndarray,
    prompt_concepts: Sequence[str],
    concepts: Sequence[str],
) -> np.ndarray:
    """Average prompt embeddings per concept, then L2-normalize concepts."""
    concept_embeddings = []
    prompt_concepts_arr = np.asarray(prompt_concepts, dtype=object)
    for concept in concepts:
        idx = np.where(prompt_concepts_arr == concept)[0]
        if len(idx) == 0:
            raise ValueError(f"No prompts found for concept: {concept}")
        avg = prompt_embeddings[idx].mean(axis=0, keepdims=True)
        concept_embeddings.append(_l2_normalize(avg)[0])
    return np.stack(concept_embeddings).astype(np.float32)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="cache/clip_embeddings/text_probe_concepts_vitl14.npz",
        help="Output .npz path.",
    )
    parser.add_argument("--model-name", "--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default=None)
    parser.add_argument("--concepts-file", "--concepts", default=None)
    parser.add_argument("--templates-file", "--templates", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        import open_clip
        import torch
    except ImportError as exc:
        print(
            "ERROR: open_clip_torch is not installed, so text embeddings cannot be "
            "generated here.\n"
            "Install it in an environment with the right CLIP weights available, "
            "then run:\n"
            "  python scripts/build/build_text_probe_embeddings.py "
            "--output cache/clip_embeddings/text_probe_concepts_vitl14.npz "
            "--model-name ViT-L-14 --pretrained openai\n"
            "Copy the resulting .npz back to this repository/cluster and set "
            "artifacts.text_embeddings_path to that file.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    concepts = _load_lines(args.concepts_file, DEFAULT_CONCEPTS)
    templates = _load_lines(args.templates_file, DEFAULT_TEMPLATES)
    prompts: list[str] = []
    prompt_concepts: list[str] = []
    for concept in concepts:
        for template in templates:
            prompts.append(_format_prompt(template, concept))
            prompt_concepts.append(concept)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, _, _ = open_clip.create_model_and_transforms(
            args.model_name,
            pretrained=args.pretrained,
        )
        tokenizer = open_clip.get_tokenizer(args.model_name)
    except Exception as exc:
        print(
            "ERROR: could not create/load the CLIP text model. This often means "
            "the weights are unavailable in the current environment or model "
            "download is blocked.\n"
            f"Requested model={args.model_name!r}, pretrained={args.pretrained!r}.\n"
            "Generate the cache on a machine with open_clip_torch and the weights "
            "available, then copy the .npz to the cluster.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    model = model.to(device)
    model.eval()

    try:
        with torch.no_grad():
            tokens = tokenizer(prompts).to(device)
            prompt_features = model.encode_text(tokens)
            prompt_features = prompt_features / prompt_features.norm(
                dim=-1,
                keepdim=True,
            ).clamp(min=1e-8)
        prompt_embeddings = prompt_features.cpu().numpy().astype(np.float32)
    except Exception as exc:
        print(
            "ERROR: CLIP text encoding failed. No cache was written. Check GPU/CPU "
            "memory, model availability, and open_clip installation.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    prompt_embeddings = _l2_normalize(prompt_embeddings).astype(np.float32)
    concept_embeddings_arr = build_concept_embeddings(
        prompt_embeddings,
        prompt_concepts,
        concepts,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        concepts=np.array(concepts, dtype=object),
        templates=np.array(templates, dtype=object),
        prompts=np.array(prompts, dtype=object),
        prompt_concepts=np.array(prompt_concepts, dtype=object),
        prompt_embeddings=prompt_embeddings,
        text_embeddings=concept_embeddings_arr,
        concept_embeddings=concept_embeddings_arr,
        model_name=np.array(args.model_name),
        pretrained_name=np.array(args.pretrained),
        embedding_dim=np.array(concept_embeddings_arr.shape[1], dtype=np.int64),
        normalized=np.array(True),
        created_at=np.array(datetime.now(timezone.utc).isoformat()),
        notes=np.array(
            "Concept embeddings are template-averaged CLIP text embeddings, "
            "L2-normalized after averaging. Prompt embeddings are also normalized."
        ),
    )
    print(
        f"Wrote {output} with {len(concepts)} concepts, {len(prompts)} prompts, "
        f"dim={concept_embeddings_arr.shape[1]}, model={args.model_name}/{args.pretrained}."
    )


if __name__ == "__main__":
    main()
