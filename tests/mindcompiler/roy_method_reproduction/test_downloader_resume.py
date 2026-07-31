"""Resume / interruption certification against a REAL local HTTP server.

Unlike test_downloader_failures.py (which stubs urllib), this stands up a threaded
HTTP server that speaks Range/206 and can inject faults, then drives the *actual*
streaming, append, and retry code in download_s3_object. Only urlopen is rewritten
to point ``https://<bucket>`` at the local ``http://127.0.0.1:<port>`` -- the read
loop, Range header, 206 handling, retry/backoff, and integrity checks all run for
real. No network and no NSD data.
"""
from __future__ import annotations

import hashlib
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fmri2img.mindcompiler.roy_method_reproduction import downloader as dl

HDF5_SIG = b"\x89HDF\r\n\x1a\n"
BODY = HDF5_SIG + bytes((i * 7 + 3) % 256 for i in range(4096))  # deterministic, not HDF5-openable
ETAG = '"deadbeef-3"'


class _State:
    fail_first_get = False   # first GET -> 500, then succeed (retry path)
    ignore_range = False     # always return 200 full body (server ignores Range)
    _gets = 0


def _handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _common(self, write_body):
            rng = self.headers.get("Range")
            if state.fail_first_get and state._gets == 0 and self.command == "GET":
                state._gets += 1
                self.send_response(500); self.end_headers()
                return
            if rng and not state.ignore_range:
                start = int(rng.split("=")[1].split("-")[0])
                chunk = BODY[start:]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(BODY)-1}/{len(BODY)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", ETAG)
                self.end_headers()
                if write_body:
                    self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(BODY)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", ETAG)
                self.end_headers()
                if write_body:
                    self.wfile.write(BODY)

        def do_HEAD(self):
            self._common(write_body=False)

        def do_GET(self):
            self._common(write_body=True)

    return H


@pytest.fixture
def server(monkeypatch):
    state = _State()
    state.fail_first_get = False; state.ignore_range = False; state._gets = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    base = f"http://127.0.0.1:{port}"
    _orig = urllib.request.urlopen

    def _local_urlopen(req, timeout=60):
        newurl = req.full_url.replace(f"https://{dl.BUCKET_HOST}", base)
        nr = urllib.request.Request(newurl, headers=dict(req.header_items()),
                                    method=req.get_method())
        return _orig(nr, timeout=timeout)

    monkeypatch.setattr(dl.urllib.request, "urlopen", _local_urlopen)
    monkeypatch.setattr(dl.time, "sleep", lambda *_: None)  # no backoff wait in tests
    try:
        yield state, base
    finally:
        httpd.shutdown()


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def test_resume_from_existing_part_uses_range(server, tmp_path):
    state, _ = server
    dest = tmp_path / "f.hdf5"
    part = str(dest) + ".part"
    with open(part, "wb") as f:  # simulate a prior interrupted transfer
        f.write(BODY[:1500])
    r = dl.download_s3_object("k", str(dest), len(BODY), expected_sha256=_sha(BODY),
                              is_hdf5=False)
    assert r.status == "verified"
    assert r.resumed is True                 # Range path was taken
    assert r.sha256 == _sha(BODY)
    assert dest.read_bytes() == BODY
    assert not (tmp_path / "f.hdf5.part").exists()


def test_server_ignoring_range_triggers_clean_restart(server, tmp_path):
    state, _ = server
    state.ignore_range = True
    dest = tmp_path / "f.hdf5"
    with open(str(dest) + ".part", "wb") as f:
        f.write(BODY[:1500])  # stale partial; server will 200 the full body
    r = dl.download_s3_object("k", str(dest), len(BODY), expected_sha256=_sha(BODY),
                              is_hdf5=False)
    assert r.status == "verified"
    assert r.resumed is False                # restart reset the resume flag
    assert dest.read_bytes() == BODY         # no doubled/appended bytes


def test_retry_after_transient_500_then_success(server, tmp_path):
    state, _ = server
    state.fail_first_get = True
    dest = tmp_path / "f.hdf5"
    r = dl.download_s3_object("k", str(dest), len(BODY), expected_sha256=_sha(BODY),
                              is_hdf5=False, max_retries=3)
    assert r.status == "verified"
    assert r.retries >= 1                     # first GET 500 forced a retry
    assert dest.read_bytes() == BODY


def test_full_clean_download_verifies(server, tmp_path):
    dest = tmp_path / "f.hdf5"
    r = dl.download_s3_object("k", str(dest), len(BODY), expected_sha256=_sha(BODY),
                              is_hdf5=False)
    assert r.status == "verified" and r.resumed is False and r.retries == 0
    assert r.etag == ETAG
