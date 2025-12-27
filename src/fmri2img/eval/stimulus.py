from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import logging
import re
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StimulusKey:
    """Unified stimulus identifier across legacy and canonical schemas.

    Attributes:
        nsd_id: Local zero-based NSD stimulus index (matches `nsd_id` column in caches).
        nsdId:  Global NSD stimulus id (matches `nsdId` column and original filenames).
    """

    nsd_id: Optional[int]
    nsdId: Optional[int]

    def best_id(self) -> Tuple[str, int]:
        """Return the preferred (field_name, value) for cache lookups."""
        if self.nsd_id is not None:
            return "nsd_id", int(self.nsd_id)
        if self.nsdId is not None:
            return "nsdId", int(self.nsdId)
        raise ValueError("StimulusKey has no usable ids")


_FILENAME_INT_RE = re.compile(r"(\d+)")


def parse_stimulus_id_from_filename(path: Path) -> Optional[int]:
    """Extract the first integer token from a filename stem.

    Examples
    --------
    - ``67574.png`` -> 67574
    - ``nsd67574_generated`` -> 67574
    - ``img_00123`` -> 123
    """
    match = _FILENAME_INT_RE.search(path.stem)
    return int(match.group(1)) if match else None


def resolve_from_index(parsed_id: int, index_df: pd.DataFrame) -> Optional[StimulusKey]:
    """Resolve a parsed id to StimulusKey using an index dataframe.

    Resolution order:
    1) match `nsdId` == parsed_id
    2) match `nsd_id` == parsed_id
    """
    nsd_id_col = "nsd_id" if "nsd_id" in index_df.columns else None
    nsdId_col = "nsdId" if "nsdId" in index_df.columns else None

    if nsdId_col and (index_df[nsdId_col] == parsed_id).any():
        row = index_df[index_df[nsdId_col] == parsed_id].iloc[0]
        nsd_id_val = int(row[nsd_id_col]) if nsd_id_col and not pd.isna(row[nsd_id_col]) else None
        return StimulusKey(nsd_id=nsd_id_val, nsdId=int(parsed_id))

    if nsd_id_col and (index_df[nsd_id_col] == parsed_id).any():
        row = index_df[index_df[nsd_id_col] == parsed_id].iloc[0]
        nsdId_val = int(row[nsdId_col]) if nsdId_col and not pd.isna(row[nsdId_col]) else None
        return StimulusKey(nsd_id=int(parsed_id), nsdId=nsdId_val)

    return None


def resolve_from_cache(parsed_id: int, cache_df: pd.DataFrame) -> Optional[Tuple[StimulusKey, str]]:
    """Try to resolve an id directly against cache columns.

    Returns tuple of (StimulusKey, matched_field).
    """
    if "nsdId" in cache_df.columns and (cache_df["nsdId"] == parsed_id).any():
        return StimulusKey(nsd_id=int(cache_df[cache_df["nsdId"] == parsed_id]["nsd_id"].iloc[0]) if "nsd_id" in cache_df.columns else None,
                           nsdId=int(parsed_id)), "nsdId"
    if "nsd_id" in cache_df.columns and (cache_df["nsd_id"] == parsed_id).any():
        return StimulusKey(nsd_id=int(parsed_id),
                           nsdId=int(cache_df[cache_df["nsd_id"] == parsed_id]["nsdId"].iloc[0]) if "nsdId" in cache_df.columns else None), "nsd_id"
    return None
