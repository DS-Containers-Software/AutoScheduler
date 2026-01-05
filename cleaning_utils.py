from __future__ import annotations

from typing import Optional
import re

import pandas as pd
from FindColorsModule import ColorScanConfig, compute_primary_color_for_part

# NOTE: This module is intended to be a pure "input normalization / cleaning" layer.
# Keep scheduling logic out of here to make it easier to test and reuse.

COLOR_RANK = {
    "WHITE": 0, "IVORY": 1, "CREAM": 1, "YELLOW": 2, "GOLD": 3, "TAN": 4, "BEIGE": 4,
    "PINK": 5, "ORANGE": 6, "RED": 7, "MAROON": 8, "BURGUNDY": 8, "PURPLE": 9, "VIOLET": 9,
    "LAVENDER": 9, "BROWN": 10, "GREEN": 11, "TEAL": 12, "TURQUOISE": 12, "CYAN": 13,
    "AQUA": 13, "SKY": 14, "BLUE": 15, "NAVY": 16, "SILVER": 17, "GREY": 18, "GRAY": 18,
    "CHARCOAL": 19, "BLACK": 20
}

def normalize_decorator(v) -> Optional[str]:
    """Maps input decorator codes to 'A' or 'B'.

    Accepted:
      - 'A'/'B' pass through
      - 'N'/'W' -> 'A'
      - 'S'/'E' -> 'B'
      - blanks/NaN -> None
    """
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    if not s:
        return None

    mapping = {
        "A": "A", "B": "B",
        "N": "A", "W": "A",
        "S": "B", "E": "B",
    }
    return mapping.get(s)

def normalize_color(v) -> str:
    if pd.isna(v):
        return "UNKNOWN"
    s = str(v).strip().upper()
    s = s.replace("BUE", "BLUE")
    s = re.sub(r"\s+", " ", s)
    return s or "UNKNOWN"

def color_rank(v) -> int:
    return COLOR_RANK.get(normalize_color(v), 999)

def normalize_family(v) -> str:
    if pd.isna(v):
        return "UNKNOWN"
    s = str(v).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s or "UNKNOWN"

def is_white_override(v) -> bool:
    if pd.isna(v):
        return False
    return str(v).strip().upper() == "W"

def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find a column name in df.columns given candidate names.

    Tries exact (case-insensitive) match first, then substring match.
    """
    cand_lower = [c.lower() for c in candidates]
    for c in df.columns:
        if c.lower() in cand_lower:
            return c
    for c in df.columns:
        cl = c.lower()
        for cand in cand_lower:
            if cand in cl:
                return c
    raise KeyError(f"Could not find any of: {candidates}")

def find_col_optional(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    try:
        return find_col(df, candidates)
    except Exception:
        return None

def _first_non_empty(series: pd.Series, default: str = "") -> str:
    for x in series.tolist():
        if str(x).strip():
            return str(x).strip()
    return default

def _first_non_null(series: pd.Series) -> Optional[str]:
    for x in series.tolist():
        if pd.notna(x) and str(x).strip():
            return str(x).strip()
    return None

def _first_positive_int(series: pd.Series) -> int:
    vals = [int(x) for x in series.tolist() if pd.notna(x) and int(x) > 0]
    return min(vals) if vals else 0

def load_and_clean(
    xlsx_path,
    *,
    pdf_base_dir: str = r"\\dsfile4\Litho\Approved PDF",
    color_cfg: ColorScanConfig = ColorScanConfig(),
) -> pd.DataFrame:
    """Load input Excel and normalize into the aggregated per-WO table used by the scheduler.

    Intentionally preserves behavior and defaults from the original monolithic script.
    """
    raw = pd.read_excel(xlsx_path)
    raw.columns = [c.strip() for c in raw.columns]

    wo_col = find_col(raw, ["Order Number"])
    qty_col = find_col(raw, ["Quantity Ordered"])
    fam_col = find_col_optional(raw, ["Family"])
    desc_col = find_col_optional(raw, ["Description"])
    part_col = find_col_optional(raw, ["2nd Item Number"])
    deco_col = find_col_optional(raw, ["Decorator"])
    seq_col = find_col_optional(raw, ["Seq"])
    req_date = find_col_optional(raw, ["Request Date"])
    white = find_col_optional(raw, ["White/Litho"])

    colors = pd.Series(["UNKNOWN"] * len(raw), index=raw.index)
    
    # Always compute colors via PDF scan when a part column exists
    if part_col:
        parts = raw[part_col].fillna("").astype(str).str.strip()
        cache: dict[str, str] = {}

        def resolve_color(idx: int, part: str) -> str:
            if white and str(raw.at[idx, white]).strip().upper() == "W":
                return "WHITE"

            if not part:
                return "UNKNOWN"

            if part not in cache:
                color = compute_primary_color_for_part(
                    part, base_dir=pdf_base_dir, config=color_cfg
                )
                cache[part] = normalize_color(color) if color else "UNKNOWN"

            return cache[part]

        colors = pd.Series((resolve_color(i, part) for i, part in enumerate(parts)),index=parts.index,)

    data = pd.DataFrame({
        "WO": (raw[wo_col].where(pd.notna(raw[wo_col]), "").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)),
        "QTY": pd.to_numeric(raw[qty_col], errors="coerce").fillna(0).astype(int),
        "PRIMARY_COLOR": colors,
        "COLOR_RANK": colors.map(color_rank),
        "FAMILY": raw[fam_col].map(normalize_family) if fam_col else "UNKNOWN",
        "DESCRIPTION": raw[desc_col].fillna("").astype(str).str.strip() if desc_col else "",
        "ITEM_NUMBER": raw[part_col].fillna("").astype(str).str.strip() if part_col else "",
        "REQ_DECORATOR": raw[deco_col].map(normalize_decorator) if deco_col else None,
        "SEQ": pd.to_numeric(raw[seq_col], errors="coerce").fillna(0).astype(int) if seq_col else 0,
        "REQ_DATE": pd.to_datetime(raw[req_date], errors="coerce") if req_date else pd.NaT,
    })

    data = data[(data["WO"] != "") & (data["QTY"] > 0)].copy()

    wo_meta = data.groupby("WO", as_index=False).agg(
        DESCRIPTION=("DESCRIPTION", lambda s: _first_non_empty(s, "")),
        ITEM_NUMBER=("ITEM_NUMBER", lambda s: _first_non_empty(s, "")),
        FAMILY=("FAMILY", lambda s: _first_non_empty(s, "UNKNOWN")),
        PRIMARY_COLOR=("PRIMARY_COLOR", lambda s: _first_non_empty(s, "UNKNOWN")),
        COLOR_RANK=("COLOR_RANK", lambda s: int(next((x for x in s.tolist() if pd.notna(x)), 999))),
        REQ_DECORATOR=("REQ_DECORATOR", _first_non_null),
        SEQ=("SEQ", _first_positive_int),
        REQ_DATE=("REQ_DATE", "min"),
    )

    qty_agg = data.groupby("WO", as_index=False)["QTY"].sum()
    agg = qty_agg.merge(wo_meta, on="WO", how="left")

    agg["DESCRIPTION"] = agg["DESCRIPTION"].fillna("")
    agg["ITEM_NUMBER"] = agg["ITEM_NUMBER"].fillna("")
    agg["FAMILY"] = agg["FAMILY"].fillna("UNKNOWN")
    agg["PRIMARY_COLOR"] = agg["PRIMARY_COLOR"].fillna("UNKNOWN")
    agg["COLOR_RANK"] = pd.to_numeric(agg["COLOR_RANK"], errors="coerce").fillna(999).astype(int)
    agg["REQ_DECORATOR"] = agg["REQ_DECORATOR"].where(agg["REQ_DECORATOR"].isin(["A", "B"]), None)


    return agg
