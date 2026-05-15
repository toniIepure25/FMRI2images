"""Configuration schema and YAML loader for manifold analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Any] = {
    "artifacts": {
        "z_pred_path": None,
        "z_target_path": None,
        "gallery_embeddings_path": None,
        "target_ids_path": None,
        "metadata_path": None,
        "kappa_path": None,
        "roi_masks_path": None,
        "per_trial_pred_path": None,
        "per_trial_nsd_ids_path": None,
        "text_embeddings_path": "cache/clip_embeddings/text_probe_concepts_vitl14.npz",
        "semantic_axes_path": None,
        "gallery_token_h5_path": None,
        "gallery_id_column": "nsdId",
        "gallery_token_mmap": False,
        "experiment_dir": None,
        "subject": None,
        "experiment_id": None,
    },
    "output": {
        "output_dir": None,
        "save_formats": ["png"],
        "dpi": 300,
    },
    "analysis": {
        "seed": 42,
        "top_k": [1, 5, 10, 20, 50, 100],
        "csls_k": 10,
        "rsa_max_samples": 1500,
        "neighborhood_ks": [5, 10, 20, 50, 100],
        "normalize_embeddings": True,
        # full: use the full gallery and map target_ids into it.
        # target_ids: filter/reorder the gallery to target_ids.
        # target_embeddings: use z_target itself as the candidate gallery.
        "gallery_mode": "full",
        "allow_diagonal_fallback": False,
        "manifold_density_k": 20,
        "calibration_n_bins": 10,
        "coverage_levels": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        "bootstrap_n": 500,
    },
    "modules": {
        "retrieval": True,
        "hubness": True,
        "rsa": True,
        "neighborhood": True,
        "repeat_stability": True,
        "uncertainty": True,
        "manifold_density": True,
        "reliability": True,
        "semantic_probes": True,
        "semantic_axes": True,
        "vector_field": True,
        "counterfactuals": True,
        "interpolation": True,
        "roi_axes": False,
        "nmas": True,
    },
    "text_probes": {
        "concepts": [
            "person", "human face", "animal", "dog", "cat",
            "vehicle", "car", "indoor room", "outdoor landscape",
            "natural scene", "city street", "forest", "mountain",
            "beach", "water", "sky", "building", "kitchen",
            "living room", "food", "road", "text sign", "object",
            "tool", "furniture", "sports", "human body", "grass",
            "urban scene",
        ],
        "templates": [
            "a photo of a {concept}",
            "an image of a {concept}",
            "a natural image containing {concept}",
        ],
    },
    "semantic_probes": {
        "temperature": 0.07,
        "top_k": [1, 3, 5, 10],
        "max_examples": 12,
    },
    "semantic_axes": {
        "axes": [
            {"name": "animal_vs_inanimate", "positive": "animal", "negative": "object"},
            {"name": "indoor_vs_outdoor", "positive": "indoor room", "negative": "outdoor landscape"},
            {"name": "face_vs_landscape", "positive": "human face", "negative": "natural scene"},
            {"name": "natural_vs_urban", "positive": "natural scene", "negative": "urban scene"},
            {"name": "vehicle_vs_animal", "positive": "vehicle", "negative": "animal"},
            {"name": "food_vs_object", "positive": "food", "negative": "object"},
            {"name": "water_vs_land", "positive": "water", "negative": "road"},
            {"name": "person_vs_empty_scene", "positive": "person", "negative": "outdoor landscape"},
            {"name": "road_vs_nature", "positive": "road", "negative": "forest"},
            {"name": "building_vs_natural_landscape", "positive": "building", "negative": "natural scene"},
        ],
        "max_examples": 12,
    },
    "reliability_weights": {
        "kappa": 1.0,
        "csls_margin": 1.0,
        "manifold_density": 1.0,
        "semantic_probe_agreement": 0.5,
        "topk_entropy": 1.0,
        "hubness_score": 0.5,
    },
    "counterfactual": {
        "alphas": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        "max_cases": 24,
        "axes": "auto",
    },
    "interpolation": {
        "steps": 21,
        "max_pairs": 20,
        "pair_strategy": "mixed",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


@dataclass
class ManifoldAnalysisConfig:
    """Flat-access wrapper around the merged YAML config dict."""

    raw: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def artifacts(self) -> Dict[str, Any]:
        return self.raw.get("artifacts", {})

    @property
    def output(self) -> Dict[str, Any]:
        return self.raw.get("output", {})

    @property
    def analysis(self) -> Dict[str, Any]:
        return self.raw.get("analysis", {})

    @property
    def modules(self) -> Dict[str, bool]:
        return self.raw.get("modules", {})

    @property
    def text_probes(self) -> Dict[str, Any]:
        return self.raw.get("text_probes", {})

    @property
    def semantic_axes(self) -> Dict[str, Any]:
        return self.raw.get("semantic_axes", {})

    @property
    def semantic_probes(self) -> Dict[str, Any]:
        return self.raw.get("semantic_probes", {})

    @property
    def reliability_weights(self) -> Dict[str, float]:
        return self.raw.get("reliability_weights", {})

    @property
    def counterfactual(self) -> Dict[str, Any]:
        return self.raw.get("counterfactual", {})

    @property
    def interpolation(self) -> Dict[str, Any]:
        return self.raw.get("interpolation", {})

    # ------------------------------------------------------------------
    def module_enabled(self, name: str) -> bool:
        return self.modules.get(name, False)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ManifoldAnalysisConfig:
    """Load YAML config, merge with defaults, apply CLI overrides."""
    raw: Dict[str, Any] = dict(_DEFAULTS)
    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                user = yaml.safe_load(f) or {}
            raw = _deep_merge(raw, user)
            logger.info("Loaded config from %s", path)
        else:
            logger.warning("Config file %s not found, using defaults", path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    return ManifoldAnalysisConfig(raw=raw)
