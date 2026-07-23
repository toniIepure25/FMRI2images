"""Atomic, resumable, integrity-checked downloader for public NSD S3 objects.

Anonymous HTTPS only (public bucket; no credentials). A file is marked
``verified`` only after **every** check passes: expected byte count, streamed
SHA-256, and -- for HDF5 -- signature plus a read-only ``h5py`` open.

ETag is recorded as metadata, never treated as a cryptographic checksum: S3
multipart ETags are not MD5 of the content, so they are informational only.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

BUCKET_HOST = "natural-scenes-dataset.s3.amazonaws.com"
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


@dataclass
class DownloadResult:
    key: str
    url: str
    dest: str
    bytes: int
    sha256: str
    etag: Optional[str]
    started_at: float
    ended_at: float
    resumed: bool
    retries: int
    hdf5_signature_ok: Optional[bool]
    h5py_open_ok: Optional[bool]
    status: str
    failure_reason: Optional[str] = None
    history: list = field(default_factory=list)


def _head(url: str, timeout: float) -> tuple[Optional[int], Optional[str], bool]:
    """Return (content_length, etag, accepts_ranges) via a HEAD request."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        cl = r.headers.get("Content-Length")
        etag = r.headers.get("ETag")
        ar = (r.headers.get("Accept-Ranges", "") or "").lower() == "bytes"
    return (int(cl) if cl is not None else None), etag, ar


def download_s3_object(
    key: str,
    dest: str,
    expected_bytes: int,
    expected_sha256: Optional[str] = None,
    is_hdf5: bool = True,
    timeout: float = 60.0,
    max_retries: int = 5,
    chunk: int = 8 << 20,
) -> DownloadResult:
    """Download an S3 key to ``dest`` atomically with resume + integrity checks.

    Writes to ``dest + ".part"``; renames to ``dest`` only after all validations
    pass. A resumable failure leaves the ``.part`` in place; a completed file that
    fails a *content* check (bad size/hash/HDF5) is quarantined to ``.corrupt``.

    Args:
        key: S3 object key. dest: final local path. expected_bytes: size from the
        tracked listing. expected_sha256: optional pin (unknown for NSD -- we
        record the computed hash instead). is_hdf5: run HDF5 signature + h5py open.
        timeout, max_retries, chunk: transfer controls.

    Returns:
        A :class:`DownloadResult`; ``status`` is ``"verified"`` only on full success.
    """
    url = f"https://{BUCKET_HOST}/{urllib.parse.quote(key)}"
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    started = time.time()
    history: list = []

    content_length, etag, accepts_ranges = _head(url, timeout)
    if content_length is not None and content_length != expected_bytes:
        return DownloadResult(key, url, dest, 0, "", etag, started, time.time(),
                              False, 0, None, None, "failed",
                              f"HEAD Content-Length {content_length} != expected {expected_bytes}",
                              history)

    resumed = False
    retries = 0
    while retries <= max_retries:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > expected_bytes:  # corrupt partial; restart clean
            os.remove(part); have = 0
        headers = {}
        if have and accepts_ranges:
            headers["Range"] = f"bytes={have}-"; resumed = True
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                server_ignored_range = have and r.status == 200  # full body despite Range
                mode = "wb" if (server_ignored_range or not have) else "ab"
                if mode == "wb":
                    have = 0; resumed = False
                with open(part, mode) as f:
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
            break
        except Exception as e:  # noqa: BLE001 - network transient
            retries += 1
            history.append(f"retry {retries}: {type(e).__name__}: {str(e)[:120]}")
            if retries > max_retries:
                return DownloadResult(key, url, dest, os.path.getsize(part) if os.path.exists(part) else 0,
                                      "", etag, started, time.time(), resumed, retries, None, None,
                                      "failed_resumable", f"exhausted retries: {e}", history)
            time.sleep(min(2 ** retries, 30))

    # --- content validations (all must pass before verified) ----------------
    size = os.path.getsize(part)
    if size != expected_bytes:
        os.replace(part, dest + ".corrupt")
        return DownloadResult(key, url, dest, size, "", etag, started, time.time(),
                              resumed, retries, None, None, "corrupt",
                              f"final size {size} != expected {expected_bytes}", history)

    h = hashlib.sha256()
    with open(part, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    sha = h.hexdigest()
    if expected_sha256 and sha != expected_sha256:
        os.replace(part, dest + ".corrupt")
        return DownloadResult(key, url, dest, size, sha, etag, started, time.time(),
                              resumed, retries, None, None, "corrupt",
                              "sha256 mismatch", history)

    sig_ok = h5open_ok = None
    if is_hdf5:
        with open(part, "rb") as f:
            sig_ok = f.read(8) == _HDF5_SIGNATURE
        if not sig_ok:
            os.replace(part, dest + ".corrupt")
            return DownloadResult(key, url, dest, size, sha, etag, started, time.time(),
                                  resumed, retries, sig_ok, None, "corrupt",
                                  "bad HDF5 signature", history)
        try:
            import h5py
            with h5py.File(part, "r"):
                h5open_ok = True
        except Exception as e:  # noqa: BLE001
            os.replace(part, dest + ".corrupt")
            return DownloadResult(key, url, dest, size, sha, etag, started, time.time(),
                                  resumed, retries, sig_ok, False, "corrupt",
                                  f"h5py open failed: {e}", history)

    os.replace(part, dest)  # atomic
    return DownloadResult(key, url, dest, size, sha, etag, started, time.time(),
                          resumed, retries, sig_ok, h5open_ok, "verified", None, history)
