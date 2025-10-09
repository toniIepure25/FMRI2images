from __future__ import annotations
import io, json
from typing import Iterator, Dict, Any
import fsspec
from PIL import Image
import nibabel as nib
import numpy as np

def open_anon(url: str):
    # fsspec handles s3:// URL directly (anon read)
    return fsspec.open(url, mode="rb", anon=True)

def load_image_s3(url: str) -> Image.Image:
    with open_anon(url) as f:
        with f as fh:
            data = fh.read()
    return Image.open(io.BytesIO(data)).convert("RGB")

def load_nifti_s3(url: str, cache_dir: str = ".cache/nsd", cache_mode="simplecache") -> np.ndarray:
    # NIfTI files are typically .nii.gz (compressed). To read with nibabel,
    # use fsspec's caching wrapper so nib gets a real local path.
    wrapped = f"{cache_mode}::{url}"
    with fsspec.open(wrapped, mode="rb", anon=True, target_protocol="s3", cache_storage=cache_dir) as f:
        # materialize to a temporary local file path via .open()'s .fs?
        # Easier: write to a NamedTemporaryFile; here we use .open().name if local
        tmp_path = f.name  # points to cached local copy
    img = nib.load(tmp_path)  # nibabel needs a file-like path; cached file works
    return np.asanyarray(img.get_fdata())

def iter_manifest(jsonl_path: str, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            yield json.loads(line)

def stream_samples(jsonl_path: str, cache_dir=".cache/nsd", limit=None):
    for rec in iter_manifest(jsonl_path, limit):
        x_img = load_image_s3(rec["stim"])
        x_bold = load_nifti_s3(rec["bold"], cache_dir=cache_dir)
        yield {"trial_id": rec["trial_id"], "subject": rec["subject"], "image": x_img, "bold": x_bold}
