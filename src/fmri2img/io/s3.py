from __future__ import annotations
import fsspec
from typing import Iterable, List

def s3_ls(url: str, anon: bool = True) -> List[str]:
    fs = fsspec.filesystem("s3", anon=anon)
    return [f"s3://{p}" for p in fs.glob(url)]
