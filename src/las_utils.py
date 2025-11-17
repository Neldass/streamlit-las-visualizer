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
    unit: Optional[str]


def _decode_las_bytes(file_bytes: bytes) -> str:
    """Decode LAS bytes to text with sensible fallbacks."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("latin-1", errors="replace")


def read_las_file(file_bytes: bytes, filename: str) -> dict:
    """Parse LAS bytes and return dataframe + metadata."""
    text = _decode_las_bytes(file_bytes)
    with io.StringIO(text) as sio:
        las = lasio.read(sio)

    df = las.df()
    idx_col_name = str(df.index.name) if df.index.name is not None else "index"
    df = df.reset_index().rename(columns={idx_col_name: "DEPTH"})
    depth_col = "DEPTH"
    if list(df.columns).count("DEPTH") > 1:
        cols = list(df.columns)
        first_idx = cols.index("DEPTH")
        to_drop = [i for i, c in enumerate(cols) if c == "DEPTH" and i != first_idx]
        if to_drop:
            df = df.drop(columns=[cols[i] for i in to_drop])

    null_value = None
    try:
        null_value = getattr(las, "null", None)
    except Exception:
        pass
    if null_value is None:
        try:
            if "NULL" in las.well:
                null_item = getattr(las.well, "NULL", None)
                null_value = getattr(null_item, "value", None)
        except Exception:
            null_value = None
    if null_value is not None:
        df = df.replace(null_value, np.nan)

    df = df.rename(columns={c: c.strip() if isinstance(c, str) else c for c in df.columns})
    df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce")

    # Preprocessing: coerce numeric logs and remove rows with only no-values
    # (common case: all curves become no-value below a certain depth).
    numeric_cols = [c for c in df.columns if c != "DEPTH"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if numeric_cols:
        df = df.dropna(subset=numeric_cols, how="all").reset_index(drop=True)

    curves_info: List[LasCurveInfo] = []
    for c in las.curves:
        name = getattr(c, "mnemonic", str(c)).strip()
        descr = (getattr(c, "descr", "") or "").strip()
        unit = getattr(c, "unit", None)
        curves_info.append(LasCurveInfo(name=name, descr=descr, unit=(unit or None)))

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
