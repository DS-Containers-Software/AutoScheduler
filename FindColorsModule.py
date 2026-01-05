from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import glob
import os

import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import pyodbc

def _get_sql_conn_str():
    cs = "DRIVER={ODBC Driver 17 for SQL Server}; SERVER=dscsqc; DATABASE=SchedulePlanning; Trusted_Connection=yes;"
    if not cs:
        raise RuntimeError("Missing SQL_CONN_STR environment variable.")
    return cs

def _fetch_cached_color(conn, part_number):
    pn = str(part_number).strip()
    sql = """
        SELECT TOP (1) [Color]
        FROM dbo.map_PartNumberToColor
        WHERE PartNumber = ?
        ORDER BY ID DESC
    """
    cur = conn.cursor()
    cur.execute(sql, pn)
    row = cur.fetchone()
    if not row:
        return None
    return str(row[0]).strip() if row[0] is not None else None

def nearest_palette_indices(lab_pixels: np.ndarray, palette_lab: np.ndarray, names: list[str]) -> np.ndarray:
    CHROMA_THRESH = 14.0
    PENALTY = 25.0
    GATED_LABELS = {"WHITE", "IVORY", "CREAM", "BEIGE", "TAN", "SILVER", "GREY", "CHARCOAL", "BLACK"}

    # (N,P) distances
    dists = np.linalg.norm(lab_pixels[:, None] - palette_lab[None], axis=2)

    a = lab_pixels[:, 1]
    b = lab_pixels[:, 2]
    chroma = np.sqrt(a * a + b * b)
    chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)

    rows = np.where(chroma > CHROMA_THRESH)[0]
    if rows.size:
        cols = [i for i, n in enumerate(names) if n in GATED_LABELS]
        if cols:
            dists[np.ix_(rows, cols)] += PENALTY  # in-place, no-copy

    return np.argmin(dists, axis=1)

def _upsert_cached_color(conn, part_number, color):
    pn = str(part_number).strip()
    c = str(color).strip()

    # pad to nchar(15) to match storage (optional, but keeps it consistent)
    c_padded = c.ljust(15)[:15]

    sql = """
    IF NOT EXISTS (
        SELECT 1
        FROM dbo.map_PartNumberToColor WITH (UPDLOCK, HOLDLOCK)
        WHERE PartNumber = ?
    )
    BEGIN
        INSERT INTO dbo.map_PartNumberToColor (PartNumber, Color)
        VALUES (?, ?)
    END
    """
    cur = conn.cursor()
    cur.execute(sql, pn, pn, c_padded)


# ----------------------------- Configuration -----------------------------

@dataclass(frozen=True)
class ColorScanConfig:
    dpi = 150
    sample_step = 1
    ignore_near_white = True
    white_thresh = 245
    suppress_edges = True         # turn on/off
    sobel_thresh = 60              # 0-255-ish; higher = keep more pixels
    edge_dilate = 1  


# ----------------------------- Color Constants -----------------------------

COLOR_RANK = {
    "WHITE": 0, "IVORY": 1, "CREAM": 1, "YELLOW": 2, "GOLD": 3, "TAN": 4, "BEIGE": 4,
    "PINK": 5, "ORANGE": 6, "RED": 7, "MAROON": 8, "BURGUNDY": 8, "PURPLE": 9, "VIOLET": 9,
    "LAVENDER": 9, "BROWN": 10, "GREEN": 11,"LIME":11, "TEAL": 12, "TURQUOISE": 12, "CYAN": 13,
    "SKY": 14, "BLUE": 15, "NAVY": 16, "SILVER": 17, "GREY": 18,
    "CHARCOAL": 19, "BLACK": 20,
}

DOWNWEIGHT = {
    "WHITE": 0.8,
    "IVORY": 0.6,
    "BLACK": 0.6,
    "CHARCOAL": 0.6,
}

BASE_RGB = {
    "WHITE": (255, 255, 255),
    "IVORY": (255, 255, 240),
    "CREAM": (255, 253, 208),
    "YELLOW": (255, 255, 0),
    "GOLD": (255, 215, 64),
    "TAN": (210, 180, 140),
    "BEIGE": (245, 245, 220),
    "PINK": (255, 192, 203),
    "ORANGE": (255, 165, 0),
    "RED": (255, 64, 64),
    "MAROON": (128, 32, 32),
    "BURGUNDY": (128, 32, 64),
    "PURPLE": (128, 32, 128),
    "VIOLET": (138, 43, 226),
    "LAVENDER": (230, 230, 250),
    "BROWN": (139, 69, 19),
    "GREEN": (32, 128, 32),
    "LIME": (64, 255, 64),
    "TEAL": (32, 128, 128),
    "TURQUOISE": (64, 224, 208),
    "CYAN": (0, 255, 255),
    "SKY": (135, 206, 235),
    "BLUE": (0, 64, 255),
    "NAVY": (0, 64, 128),
    "SILVER": (192, 192, 192),
    "GREY": (128, 128, 128),
    "CHARCOAL": (64, 64, 64),
    "BLACK": (0, 0, 0),
}


# ----------------------------- PDF Path Helpers -----------------------------

def build_approved_pdf_path(part_number, base_dir=r"\\dsfile4\Litho\Approved PDF"):
    prefix, suffix = part_number.split("-")
    suffix_floor = (int(suffix) // 100) * 100
    return os.path.join(base_dir, prefix, f"{suffix_floor:05d}")


def get_pdf_paths_for_parts(part_numbers, base_dir=r"\\dsfile4\Litho\Approved PDF"):
    results = {}

    for raw in part_numbers:
        if not raw:
            continue

        part = str(raw).strip()
        if "-" not in part:
            results[part] = []
            continue

        try:
            folder = build_approved_pdf_path(part, base_dir)
        except Exception:
            results[part] = []
            continue

        pattern = os.path.join(folder, f"*{part}*.pdf")
        results[part] = glob.glob(pattern)

    return results


def get_most_recent_pdf_paths(part_numbers, base_dir=r"\\dsfile4\Litho\Approved PDF"):
    all_matches = get_pdf_paths_for_parts(part_numbers, base_dir)
    most_recent = {}

    for part, paths in all_matches.items():
        most_recent[part] = max(paths, key=os.path.getctime) if paths else None

    return most_recent


# ----------------------------- Color Math -----------------------------

def _srgb_to_linear(srgb):
    srgb = srgb / 255.0
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )

def _linear_to_xyz(rgb_lin):
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    return rgb_lin @ M.T

def _xyz_to_lab(xyz):
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[..., 0] / Xn, xyz[..., 1] / Yn, xyz[..., 2] / Zn

    eps = 216 / 24389
    kappa = 24389 / 27

    def f(t):
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)

    fx, fy, fz = f(x), f(y), f(z)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)

def _srgb_to_lab(rgb):
    return _xyz_to_lab(_linear_to_xyz(_srgb_to_linear(rgb.astype(float))))

def _build_palette_lab():
    names = list(COLOR_RANK.keys())
    rgbs = np.array([BASE_RGB[n] for n in names], dtype=np.uint8)
    return names, _srgb_to_lab(rgbs)

def downweight_counts(counts):
    adjusted = Counter()
    for color, cnt in counts.items():
        adjusted[color] = int(round(cnt * DOWNWEIGHT.get(color, 1.0)))
    return adjusted

def _sobel_mag_u8(gray_u8: np.ndarray) -> np.ndarray:
    g = gray_u8.astype(np.int16)

    # pad edges
    p = np.pad(g, ((1, 1), (1, 1)), mode="edge")

    # Sobel kernels:
    # Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    # Gy = [[ 1,2,1],[ 0,0,0],[-1,-2,-1]]
    gx = (
        -p[:-2, :-2] + p[:-2, 2:]
        -2 * p[1:-1, :-2] + 2 * p[1:-1, 2:]
        -p[2:, :-2] + p[2:, 2:]
    )
    gy = (
         p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]
        -p[2:, :-2] - 2 * p[2:, 1:-1] - p[2:, 2:]
    )

    mag = np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)

    mmax = float(mag.max()) if mag.size else 0.0
    if mmax <= 1e-6:
        return np.zeros_like(gray_u8, dtype=np.uint8)

    mag_u8 = np.clip((mag / mmax) * 255.0, 0, 255).astype(np.uint8)
    return mag_u8


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask

    m = mask.astype(bool)
    p = np.pad(m, ((radius, radius), (radius, radius)), mode="constant", constant_values=False)

    out = np.zeros_like(m, dtype=bool)
    # square dilation via OR over shifted windows
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= p[radius + dy : radius + dy + m.shape[0], radius + dx : radius + dx + m.shape[1]]

    return out

# ----------------------------- Public Color API -----------------------------

def count_pdf_colors(pdf_path, config=ColorScanConfig()):
    names, palette_lab = _build_palette_lab()
    totals = Counter()

    doc = fitz.open(pdf_path)
    zoom = config.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    try:
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=True)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)

            rgb = arr[..., :3]
            alpha = arr[..., 3]

            if config.sample_step > 1:
                rgb = rgb[::config.sample_step, ::config.sample_step]
                alpha = alpha[::config.sample_step, ::config.sample_step]

            mask = alpha >= 200

            if config.ignore_near_white:
                mask &= ~(
                    (rgb[..., 0] >= config.white_thresh) &
                    (rgb[..., 1] >= config.white_thresh) &
                    (rgb[..., 2] >= config.white_thresh)
                )

            if getattr(config, "suppress_edges", False):
                gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
                mag = _sobel_mag_u8(gray)
                edges = mag >= int(getattr(config, "sobel_thresh", 60))
                edges = _dilate_mask(edges, int(getattr(config, "edge_dilate", 1)))
                mask &= ~edges

            pixels = rgb[mask]
            if pixels.size == 0:
                continue

            lab_pixels = _srgb_to_lab(pixels)
            nearest = nearest_palette_indices(lab_pixels, palette_lab, names)

            for idx, count in zip(*np.unique(nearest, return_counts=True)):
                totals[names[int(idx)]] += int(count)

    finally:
        doc.close()

    return totals

def primary_color(counts):
    if not counts:
        return None
    max_count = max(counts.values())
    tied = [c for c, n in counts.items() if n == max_count]
    tied.sort(key=lambda c: COLOR_RANK[c])
    return tied[0]

def classify_pdf_primary_color(pdf_path, config=ColorScanConfig(), apply_downweight=True):
    raw = count_pdf_colors(pdf_path, config)
    counts = downweight_counts(raw) if apply_downweight else raw
    return primary_color(counts), counts

def get_most_recent_pdf_path(part_number, base_dir=r"\\dsfile4\Litho\Approved PDF"):
    return get_most_recent_pdf_paths([part_number], base_dir).get(part_number)

def compute_primary_color_for_part(
    part_number,
    *,
    base_dir=r"\\dsfile4\Litho\Approved PDF",
    config=ColorScanConfig(),
    apply_downweight=True,
):
    conn_str = _get_sql_conn_str()
    with pyodbc.connect(conn_str) as conn:
        conn.autocommit = False

        cached = _fetch_cached_color(conn, part_number)
        if cached:
            return cached

        pdf_path = get_most_recent_pdf_path(part_number, base_dir=base_dir)
        if not pdf_path:
            return None

        primary, _counts = classify_pdf_primary_color(
            pdf_path,
            config=config,
            apply_downweight=apply_downweight,
        )
        if not primary:
            return None

        _upsert_cached_color(conn, part_number, primary)
        conn.commit()
        return primary

__all__ = [
    "ColorScanConfig",
    "get_most_recent_pdf_path",
    "compute_primary_color_for_part",
    "classify_pdf_primary_color",
    "count_pdf_colors",
    "nearest_palette_indices",
]


