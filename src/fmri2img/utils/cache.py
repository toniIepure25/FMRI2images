from __future__ import annotations
import fsspec
from typing import Optional

def make_fs(anon: bool = True) -> fsspec.AbstractFileSystem:
    return fsspec.filesystem("s3", anon=anon)

def cached_url(s3_url: str, cache_dir: str, mode: str = "simplecache") -> str:
    """
    Wrap an S3 URL so reads stream and cache locally on first access.
    - simplecache::s3://bucket/key -> saves full file once fetched
    - filecache::s3://bucket/key   -> chunked, similar behavior
    """
    prefix = f"{mode}::{s3_url}"
    # For simplecache, you can set target cache dir via fsspec.open kwarg
    return prefix
