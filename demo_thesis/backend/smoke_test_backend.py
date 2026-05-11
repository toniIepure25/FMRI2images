#!/usr/bin/env python3
"""Smoke-test a running Cortex2Canvas backend."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

def _green(s: str) -> str: return f"\033[92m{s}\033[0m"
def _red(s: str) -> str: return f"\033[91m{s}\033[0m"
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m"

def fetch_json(url: str, timeout: float = 10.0):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"raw": body[:500]}
    except Exception as e:
        return None, {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Smoke-test Cortex2Canvas backend")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print("=" * 60)
    print(f"Cortex2Canvas Backend Smoke Test — {base}")
    print("=" * 60)
    print()

    # 1. Health
    print("1. /api/health")
    code, data = fetch_json(f"{base}/api/health")
    if code is None:
        print(f"   {_red('UNREACHABLE')}: {data.get('error')}")
        sys.exit(1)
    status = data.get("status", "?")
    missing = data.get("missing", [])
    print(f"   Status:   {_green(status) if status in ('inference_ready', 'data_ready') else _yellow(status)}")
    print(f"   Torch:    {'yes' if data.get('torch_available') else 'no'}  Device: {data.get('device')}")
    print(f"   Features: {data.get('features_loaded')}  shape={data.get('features_shape')}")
    print(f"   Gallery:  {data.get('gallery_loaded')}  size={data.get('gallery_size')}  dim={data.get('gallery_dim')}")
    print(f"   Index:    {data.get('trial_index_loaded')}  count={data.get('trial_count')}")
    print(f"   Model:    {data.get('model_loaded')}")
    print(f"   Inference:{data.get('inference_available')}  LiveRetrieval:{data.get('live_retrieval_available')}")
    if missing:
        print(f"   Missing:  {', '.join(missing)}")
    if data.get("errors"):
        for k, v in data["errors"].items():
            print(f"   Error[{k}]: {v}")
    print()

    # 2. Trials
    print("2. /api/trials")
    code, data = fetch_json(f"{base}/api/trials?limit=5")
    if code == 200 and isinstance(data, list):
        print(f"   {_green('OK')} — {len(data)} rows returned (limit=5)")
        if data:
            print(f"   First: {data[0]}")
    elif code == 503:
        print(f"   {_yellow('503')} — {data.get('reason', '?')}: {data.get('message', '')}")
    else:
        print(f"   {_red(f'HTTP {code}')}: {data}")
    print()

    # 3. Infer
    print("3. /api/infer/0")
    t0 = time.time()
    code, data = fetch_json(f"{base}/api/infer/0", timeout=30.0)
    elapsed = time.time() - t0
    if code == 200 and data.get("ok"):
        top1 = data.get("top_k", [{}])[0] if data.get("top_k") else {}
        print(f"   {_green('OK')} — latency={elapsed:.2f}s")
        print(f"   kappa={data.get('kappa')}  delta={data.get('delta')}")
        print(f"   Gallery size: {data.get('gallery_size')}")
        print(f"   Top-1: nsdId={top1.get('nsd_id')} cos={top1.get('cosine', 0):.4f} csls={top1.get('csls', 0):.4f}")
    elif code == 503:
        reason = data.get("reason", "?")
        print(f"   {_yellow('Skipped')} — {reason}: {data.get('message', '')}")
    else:
        print(f"   {_red(f'HTTP {code}')}: {json.dumps(data)[:200]}")
    print()

    print("=" * 60)
    print("Done.")

if __name__ == "__main__":
    main()
