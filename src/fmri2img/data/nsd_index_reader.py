from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
from typing import Optional


logger = logging.getLogger(__name__)

def read_subject_index(
    index_root_or_file: str,
    subject: str,
    allow_fallback_index: bool = False,
) -> pd.DataFrame:
    """
    Read only one subject partition when available.
    If `index_root_or_file` points to a partitioned directory, append
    `subject=subjXX/index.parquet`. Otherwise read the file directly.
    """
    root_path = Path(index_root_or_file)

    if index_root_or_file.endswith(".parquet") and "subject=" in index_root_or_file:
        df = pd.read_parquet(index_root_or_file)  # already a partition
    elif index_root_or_file.endswith(".parquet"):
        df = pd.read_parquet(index_root_or_file)
        if "subject" in df.columns:
            df = df[df["subject"] == subject]
    elif index_root_or_file.endswith(".csv"):
        df = pd.read_csv(index_root_or_file)
        # Convert subject number to string format if needed
        if "subject" in df.columns:
            # Handle both numeric (1) and string ("subj01") subject formats
            subject_num = int(subject.replace("subj", "")) if subject.startswith("subj") else int(subject)
            df = df[df["subject"] == subject_num]
            # Rename columns to match expected format
            if "beta_file" in df.columns and "beta_path" not in df.columns:
                df = df.copy()
                df["beta_path"] = df["beta_file"]
            if "volume_index" in df.columns and "beta_index" not in df.columns:
                df = df.copy()
                df["beta_index"] = df["volume_index"]
            if "nsd_id" in df.columns and "nsdId" not in df.columns:
                df = df.copy()
                df["nsdId"] = df["nsd_id"]
    else:
        # assume root directory layout
        df = pd.read_parquet(f"{index_root_or_file.rstrip('/')}/subject={subject}/index.parquet")

    # Fail-fast on collapsed beta_path unless explicitly allowed.
    # If session info exists, only fail when multiple sessions collapse to one
    # beta_path. If session info is missing, be conservative and require an
    # explicit allow_fallback_index=True to proceed.
    if "beta_path" in df.columns and df["beta_path"].nunique() == 1 and not allow_fallback_index:
        if "session" not in df.columns or df["session"].nunique() > 1:
            raise ValueError(
                "Index appears collapsed to a single beta_path; rebuild canonical index or set allow_fallback_index=True explicitly."
            )

    return df

def sample_trials(df: pd.DataFrame, n: int = 8, session: Optional[int] = None) -> pd.DataFrame:
    if session is not None and "session" in df.columns:
        df = df[df["session"] == session]
    return df.sample(min(n, len(df)), random_state=0).reset_index(drop=True)