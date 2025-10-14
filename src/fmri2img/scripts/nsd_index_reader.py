#!/usr/bin/env python3
import argparse, logging, pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nsd_index_reader")

def main():
    ap = argparse.ArgumentParser("NSD Index Reader")
    ap.add_argument("--index", required=True, help="Path or S3 URL to Parquet (partition or single file)")
    ap.add_argument("--subject", default=None, help="Subject ID like 'subj01'")
    ap.add_argument("--session", type=int, default=None, help="Session number, optional")
    ap.add_argument("--n", type=int, default=10, help="Rows to show")
    args = ap.parse_args()

    # Read Parquet (fsspec-aware)
    df = pd.read_parquet(args.index)

    if args.subject:
        df = df[df["subject"] == args.subject]
    if args.session is not None and "session" in df.columns:
        df = df[df["session"] == args.session]

    cols = [
        "subject","session","trial_in_session","global_trial_index",
        "nsdId","beta_path","beta_index"
    ]
    cols = [c for c in cols if c in df.columns]
    log.info("Showing %d rows", min(args.n, len(df)))
    print(df[cols].head(args.n).to_string(index=False))

if __name__ == "__main__":
    main()