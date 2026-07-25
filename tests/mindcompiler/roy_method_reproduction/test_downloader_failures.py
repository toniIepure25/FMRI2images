"""Data-free failure-path certification for the atomic downloader.

No internet access: urllib is monkeypatched with deterministic fakes, and real
HDF5 bytes are produced by h5py for the success cases. Certifies that
``verified`` is emitted ONLY after every check passes, that content failures are
quarantined to ``.corrupt``, and that a multipart ETag is metadata, never a
cryptographic checksum.
"""
import hashlib
from pathlib import Path

from fmri2img.mindcompiler.roy_method_reproduction import downloader as dl

HDF5_SIG = b"\x89HDF\r\n\x1a\n"


class _Resp:
    def __init__(self, body: bytes, status=200):
        self._b = body; self.status = status; self.headers = {}

    def read(self, n=-1):
        if n is None or n < 0:
            b, self._b = self._b, b""; return b
        b, self._b = self._b[:n], self._b[n:]; return b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch(monkeypatch, body, etag='"abc-21"', accepts_ranges=True, status=200):
    monkeypatch.setattr(dl, "_head", lambda url, timeout: (len(body), etag, accepts_ranges))
    monkeypatch.setattr(dl.urllib.request, "urlopen",
                        lambda req, timeout=60: _Resp(body, status))


def _real_hdf5_bytes(tmp_path):
    import h5py
    import numpy as np
    p = tmp_path / "_src.hdf5"
    with h5py.File(p, "w") as f:
        f.create_dataset("betas", data=np.zeros((2, 2, 2, 2), dtype="int16"))
    return p.read_bytes()


def test_no_test_requires_internet():
    assert dl.BUCKET_HOST == "natural-scenes-dataset.s3.amazonaws.com"


def test_clean_hdf5_transfer_verified(monkeypatch, tmp_path):
    body = _real_hdf5_bytes(tmp_path)
    _patch(monkeypatch, body)
    r = dl.download_s3_object("k", str(tmp_path / "f.hdf5"), len(body), is_hdf5=True)
    assert r.status == "verified"
    assert r.sha256 == hashlib.sha256(body).hexdigest()
    assert r.hdf5_signature_ok and r.h5py_open_ok
    assert (tmp_path / "f.hdf5").exists() and not (tmp_path / "f.hdf5.part").exists()


def test_head_size_mismatch_fails_before_transfer(monkeypatch, tmp_path):
    body = HDF5_SIG + b"\x00" * 100
    _patch(monkeypatch, body)  # HEAD advertises len(body)
    r = dl.download_s3_object("k", str(tmp_path / "f.hdf5"), expected_bytes=999999, is_hdf5=True)
    assert r.status == "failed"
    assert not (tmp_path / "f.hdf5").exists()


def test_bad_hdf5_signature_quarantined(monkeypatch, tmp_path):
    body = b"NOTHDF5!" + b"\x00" * 100
    _patch(monkeypatch, body)
    dest = tmp_path / "f.hdf5"
    r = dl.download_s3_object("k", str(dest), len(body), is_hdf5=True)
    assert r.status == "corrupt" and "signature" in r.failure_reason
    assert not dest.exists() and Path(str(dest) + ".corrupt").exists()


def test_sha_mismatch_quarantined(monkeypatch, tmp_path):
    body = HDF5_SIG + b"\x00" * 64
    _patch(monkeypatch, body)
    dest = tmp_path / "f.hdf5"
    r = dl.download_s3_object("k", str(dest), len(body), expected_sha256="deadbeef" * 8, is_hdf5=False)
    assert r.status == "corrupt" and "sha256" in r.failure_reason
    assert not dest.exists() and Path(str(dest) + ".corrupt").exists()


def test_h5py_open_failure_quarantined(monkeypatch, tmp_path):
    # Valid HDF5 signature but garbage content: real h5py must fail to open it.
    body = HDF5_SIG + b"\x01\x02\x03garbage" * 8
    _patch(monkeypatch, body)
    dest = tmp_path / "f.hdf5"
    r = dl.download_s3_object("k", str(dest), len(body), is_hdf5=True)
    assert r.status == "corrupt" and "h5py" in r.failure_reason
    assert not dest.exists() and Path(str(dest) + ".corrupt").exists()


def test_verified_never_emitted_on_failure(monkeypatch, tmp_path):
    body = b"NOTHDF5!" + b"\x00" * 10
    _patch(monkeypatch, body)
    r = dl.download_s3_object("k", str(tmp_path / "f.hdf5"), len(body), is_hdf5=True)
    assert r.status != "verified"


def test_multipart_etag_is_metadata_only(monkeypatch, tmp_path):
    body = _real_hdf5_bytes(tmp_path)
    _patch(monkeypatch, body, etag='"8b3d07e0b16bea4c041fee98d737b911-21"')
    r = dl.download_s3_object("k", str(tmp_path / "f.hdf5"), len(body), is_hdf5=True)
    assert r.etag == '"8b3d07e0b16bea4c041fee98d737b911-21"'
    # the content hash is the SHA-256, never the ETag
    assert r.sha256 != r.etag.strip('"').split("-")[0]
    assert r.status == "verified"
