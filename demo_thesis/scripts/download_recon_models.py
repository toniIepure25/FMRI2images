#!/usr/bin/env python3
"""Download SDXL 1.0 + IP-Adapter weights for reconstruction."""
import os
os.environ["HF_HOME"] = "/home/jovyan/work/.cache/hf"

from huggingface_hub import snapshot_download

print("Starting SDXL 1.0 download...")
snapshot_download(
    "stabilityai/stable-diffusion-xl-base-1.0",
    cache_dir="/home/jovyan/work/.cache/hf",
)
print("SDXL 1.0 downloaded!")

print("Starting IP-Adapter download...")
snapshot_download(
    "h94/IP-Adapter",
    cache_dir="/home/jovyan/work/.cache/hf",
    allow_patterns=["sdxl_models/*", "models/image_encoder/*"],
)
print("IP-Adapter downloaded!")

print("Starting open_clip ViT-L/14 check...")
try:
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    print("open_clip ViT-L/14 OK")
except Exception as e:
    print(f"open_clip check failed (non-critical): {e}")

print("ALL DONE")
