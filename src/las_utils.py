from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, List, Optional

import lasio
import numpy as np
import pandas as pd


@dataclass
class LasCurveInfo:
    name: str
    descr: str
    unit: str | None


def _decode_las_bytes(file_bytes: bytes) -> str:
    """Best-effort decode of LAS file bytes to text.

    LAS files are plain text (historically ASCII). We try UTF-8 first, then fall back
    to common single-byte encodings. As a last resort we replace undecodable characters.
    """
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    # Fallback with replacement to ensure we always return something
    return file_bytes.decode("latin-1", errors="replace")


def read_las_file(file_bytes: bytes, filename: str) -> dict:
    """Read a LAS file from raw bytes and return structured info.

    Returns a dict with keys: filename, df (DataFrame), curves (list[LasCurveInfo]), depth_col

    Note: We must pass a *text* file-like object to lasio. Passing a BytesIO causes lasio
    to iterate bytes and later call bytes.strip("\n"), which triggers the TypeError
    (bytes.strip expects a bytes argument). Hence we decode first then wrap with StringIO.
    """
    text = _decode_las_bytes(file_bytes)
    with io.StringIO(text) as sio:
        # Allow lasio to try its own encoding heuristics on already-decoded text.
        las = lasio.read(sio)

    # Build dataframe
    df = las.df()

    # Ensure we have a DEPTH column by materializing the index
    # las.df() typically sets the depth as index (often named DEPT/DEPTH)
    idx_col_name = str(df.index.name) if df.index.name is not None else "index"
    df = df.reset_index().rename(columns={idx_col_name: "DEPTH"})
    depth_col = "DEPTH"

    # If multiple DEPTH columns exist (can happen after reset), keep the first and drop the rest
    if list(df.columns).count("DEPTH") > 1:
        # Identify all DEPTH columns and drop duplicates keeping first occurrence
        cols = list(df.columns)
        first_idx = cols.index("DEPTH")
        to_drop = [i for i, c in enumerate(cols) if c == "DEPTH" and i != first_idx]
        if to_drop:
            df = df.drop(columns=[cols[i] for i in to_drop])

    # Replace null values defined in LAS header (robust across lasio versions)
    null_value = None
    try:
        # Some versions expose LASFile.null
        null_value = getattr(las, "null", None)
    except Exception:
        pass
    if null_value is None:
        try:
            if "NULL" in las.well:
                # Access via attribute to avoid static typing complaints
                null_item = getattr(las.well, "NULL", None)
                null_value = getattr(null_item, "value", None)
        except Exception:
            null_value = None
    if null_value is not None:
        df = df.replace(null_value, np.nan)

    # Clean curve names (strip spaces)
    df = df.rename(columns={c: c.strip() if isinstance(c, str) else c for c in df.columns})
    # Ensure DEPTH is numeric
    df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce")

    curves_info: List[LasCurveInfo] = []
    for c in las.curves:
        # Defensive: sometimes fields can be None
        name = getattr(c, "mnemonic", str(c)).strip()
        descr = (getattr(c, "descr", "") or "").strip()
        unit = getattr(c, "unit", None)
        curves_info.append(LasCurveInfo(name=name, descr=descr, unit=(unit or None)))

    # Extract well name if available
    well_name = None
    try:
        well_obj = getattr(las.well, "WELL", None)
        if well_obj is not None:
            wv = getattr(well_obj, "value", None)
            if wv is not None and str(wv).strip():
                well_name = str(wv).strip()
    except Exception:
        well_name = None

    display_name = well_name or filename

    return {
        "filename": filename,
        "well_name": well_name,
        "display_name": display_name,
        "df": df,
        "curves": curves_info,
        "depth_col": depth_col,
    }


def list_numeric_logs(df: pd.DataFrame, exclude: Optional[Iterable[str]] = None) -> List[str]:
    exclude_set = set(exclude or [])
    cols: List[str] = []
    for c in df.columns:
        if c in exclude_set:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def subset_depth(df: pd.DataFrame, depth_min: Optional[float], depth_max: Optional[float]) -> pd.DataFrame:
    if depth_min is not None:
        df = df[df["DEPTH"] >= depth_min]
    if depth_max is not None:
        df = df[df["DEPTH"] <= depth_max]
    return df


# Note: smoothing helpers et al. ont été retirés pour simplifier le module.
