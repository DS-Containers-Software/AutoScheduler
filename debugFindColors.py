from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF

import FindColorsModule as FC  # <-- your existing module


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_mask_png(path: str, mask_bool: np.ndarray, title: str):
    img = (mask_bool.astype(np.uint8) * 255)
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_overlay_png(path: str, rgb: np.ndarray, mask_bool: np.ndarray, title: str):
    overlay = rgb.copy()
    overlay[mask_bool] = np.array([255, 0, 0], dtype=np.uint8)  # highlight kept pixels
    plt.figure()
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_counts_bar(path: str, counts: Counter, title: str, top_n: int = 15):
    items = counts.most_common(top_n)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _primary_color_from_counts(counts: Counter) -> str | None:
    # Use your module’s tie-breaker / ranking
    return FC.primary_color(counts)


def debug_part(
    part_number: str,
    out_dir: str,
    *,
    base_dir: str,
    config: FC.ColorScanConfig,
    apply_downweight: bool,
    max_pages: int,
):
    _ensure_dir(out_dir)

    pdf_path = FC.get_most_recent_pdf_path(part_number, base_dir=base_dir)
    if not pdf_path:
        raise SystemExit(f"No PDF found for part: {part_number}")

    # Palette in Lab (your internal helper exists in the module)
    names, palette_lab = FC._build_palette_lab()

    raw_totals = Counter()

    doc = fitz.open(pdf_path)
    zoom = config.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    report_lines: list[str] = []
    report_lines.append(f"Part: {part_number}")
    report_lines.append(f"PDF:  {pdf_path}")
    report_lines.append(f"Config: {config}")
    report_lines.append("")

    try:
        for page_idx, page in enumerate(doc, start=1):
            if max_pages and page_idx > max_pages:
                break

            page_dir = _ensure_dir(os.path.join(out_dir, f"page_{page_idx:02d}"))

            pix = page.get_pixmap(matrix=mat, alpha=True)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)

            rgb = arr[:, :, :3]
            alpha = arr[:, :, 3]

            # Apply sampling exactly like your module
            if getattr(config, "sample_step", 1) and config.sample_step > 1:
                rgb_s = rgb[::config.sample_step, ::config.sample_step]
                alpha_s = alpha[::config.sample_step, ::config.sample_step]
            else:
                rgb_s = rgb
                alpha_s = alpha

            # Step 1: alpha mask (your module uses alpha >= 200) :contentReference[oaicite:3]{index=3}
            mask_alpha = alpha_s >= 200

            # Step 2: remove near-white if enabled :contentReference[oaicite:4]{index=4}
            mask_nowhite = mask_alpha.copy()
            if getattr(config, "ignore_near_white", False):
                wt = int(getattr(config, "white_thresh", 245))
                near_white = (
                    (rgb_s[:, :, 0] >= wt)
                    & (rgb_s[:, :, 1] >= wt)
                    & (rgb_s[:, :, 2] >= wt)
                )
                mask_nowhite &= ~near_white

            # Step 3: suppress edges if enabled (Sobel + dilate) :contentReference[oaicite:5]{index=5}
            mask_noedges = mask_nowhite.copy()
            edges = None
            if getattr(config, "suppress_edges", False):
                gray = (0.299 * rgb_s[:, :, 0] + 0.587 * rgb_s[:, :, 1] + 0.114 * rgb_s[:, :, 2]).astype(
                    np.uint8
                )
                mag = FC._sobel_mag_u8(gray)
                edges = mag >= int(getattr(config, "sobel_thresh", 60))
                edges = FC._dilate_mask(edges, int(getattr(config, "edge_dilate", 1)))
                mask_noedges &= ~edges

            mask_final = mask_noedges

            # Save visuals
            _save_mask_png(os.path.join(page_dir, "01_mask_alpha.png"), mask_alpha, "Mask: alpha >= 200")
            _save_mask_png(os.path.join(page_dir, "02_mask_nowhite.png"), mask_nowhite, "Mask: remove near-white")
            if edges is not None:
                _save_mask_png(os.path.join(page_dir, "03_mask_edges.png"), edges, "Edges (Sobel+dilate)")
            _save_mask_png(os.path.join(page_dir, "04_mask_final.png"), mask_final, "Final mask (pixels counted)")
            _save_overlay_png(
                os.path.join(page_dir, "05_overlay_final.png"),
                rgb_s,
                mask_final,
                "Overlay (red = pixels counted)",
            )

            # Run the exact same palette assignment as your module :contentReference[oaicite:6]{index=6}
            pixels = rgb_s[mask_final]
            if pixels.size == 0:
                report_lines.append(f"Page {page_idx}: No pixels after masking.")
                report_lines.append("")
                continue

            lab_pixels = FC._srgb_to_lab(pixels)
            nearest = FC.nearest_palette_indices(lab_pixels, palette_lab, names)
            # Optional: report confusion for CHARCOAL pixels
            report_confusion(
                pixels, nearest, names, palette_lab,
                out_txt=os.path.join(page_dir, "confusion.txt"),
            )

            page_counts = Counter()
            idxs, cnts = np.unique(nearest, return_counts=True)
            for idx, cnt in zip(idxs, cnts):
                color_name = names[int(idx)]
                page_counts[color_name] += int(cnt)
                raw_totals[color_name] += int(cnt)

            # Page summary
            top5 = page_counts.most_common(5)
            report_lines.append(f"Page {page_idx}: kept_pixels={int(mask_final.sum())}")
            report_lines.append("  Top 5 raw counts: " + ", ".join([f"{k}={v}" for k, v in top5]))
            report_lines.append("")

    finally:
        doc.close()

    # Totals + downweighting (your module’s helper) :contentReference[oaicite:7]{index=7}
    adjusted = FC.downweight_counts(raw_totals) if apply_downweight else raw_totals
    chosen = _primary_color_from_counts(adjusted)

    # Save charts
    _save_counts_bar(os.path.join(out_dir, "counts_raw.png"), raw_totals, "Raw palette counts (all pages)")
    if apply_downweight:
        _save_counts_bar(os.path.join(out_dir, "counts_downweighted.png"), adjusted, "Downweighted counts (used to decide)")

    # Final report
    report_lines.append("=== TOTALS ===")
    report_lines.append("Raw top 10: " + ", ".join([f"{k}={v}" for k, v in raw_totals.most_common(10)]))
    if apply_downweight:
        report_lines.append("Downweighted top 10: " + ", ".join([f"{k}={v}" for k, v in adjusted.most_common(10)]))
    report_lines.append(f"Chosen primary color: {chosen}")
    report_lines.append("")

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Wrote debug output to: {out_dir}")
    print(f"Chosen primary color: {chosen}")

def report_confusion(
    pixels_rgb: np.ndarray,
    nearest: np.ndarray,
    names: list[str],
    palette_lab: np.ndarray,
    out_txt: str,
    *,
    sample_n: int = 12,
):
    """
    Automatic confusion report:
      - Winner = highest-count label in `nearest`
      - Alternative = most common 2nd-closest label (per-pixel) among pixels assigned to winner
    """

    if pixels_rgb.size == 0 or nearest.size == 0:
        with open(out_txt, "a", encoding="utf-8") as f:
            f.write("\nNo pixels to analyze.\n")
        return

    # Winner by pixel count
    counts = np.bincount(nearest, minlength=len(names))
    winner_idx = int(np.argmax(counts))
    winner_name = names[winner_idx]
    winner_count = int(counts[winner_idx])

    # If only one label present, there is no "next best"
    unique = np.flatnonzero(counts)
    runnerup_idx = None
    if unique.size > 1:
        # runner-up by count (not by distance)
        sorted_by_count = np.argsort(counts)[::-1]
        runnerup_idx = int(sorted_by_count[1])
        runnerup_name = names[runnerup_idx]
        runnerup_count = int(counts[runnerup_idx])
    else:
        runnerup_name = None
        runnerup_count = 0

    mask_w = (nearest == winner_idx)
    w_pixels = pixels_rgb[mask_w]
    mean_rgb = w_pixels.mean(axis=0)

    # Compute Lab distances of WINNER pixels to all palette anchors
    lab = FC._srgb_to_lab(w_pixels)  # (M,3)
    d_all = np.linalg.norm(lab[:, None] - palette_lab[None], axis=2)  # (M,P)

    # For each pixel, get the best and 2nd-best palette indices by distance
    order = np.argsort(d_all, axis=1)  # (M,P)
    best = order[:, 0]
    second = order[:, 1]  # 2nd-closest by distance

    # Sanity: best should be the winner almost always (it’s how nearest was made)
    # But we won't assume; we’ll still compute "alt" from 2nd-best votes.
    alt_counts = np.bincount(second, minlength=len(names))
    alt_idx = int(np.argmax(alt_counts))
    alt_name = names[alt_idx]
    alt_votes = int(alt_counts[alt_idx])

    # Distances for reporting
    d_win = np.linalg.norm(lab - palette_lab[winner_idx], axis=1)
    d_alt = np.linalg.norm(lab - palette_lab[alt_idx], axis=1)

    with open(out_txt, "a", encoding="utf-8") as f:
        f.write("\n=== CONFUSION EVIDENCE (AUTO) ===\n")
        f.write(f"Winner by pixel count: {winner_name} ({winner_count})\n")
        if runnerup_name is not None:
            f.write(f"Runner-up by pixel count: {runnerup_name} ({runnerup_count})\n")

        f.write(f"\nFor pixels classified as WINNER ({winner_name}):\n")
        f.write(f"  Count: {w_pixels.shape[0]}\n")
        f.write(f"  Mean RGB: {mean_rgb.round(1).tolist()}\n")
        f.write(f"  Median d(Lab) to WINNER ({winner_name}): {float(np.median(d_win)):.2f}\n")

        f.write("\nMost common 'next best' (2nd-closest) label:\n")
        f.write(f"  Alternative: {alt_name} (votes among 2nd-best = {alt_votes})\n")
        f.write(f"  Median d(Lab) to ALT ({alt_name}): {float(np.median(d_alt)):.2f}\n")

        # Show top 5 alternatives (what it "could have been" besides winner)
        f.write("\nTop 5 'next best' labels (2nd-closest frequency):\n")
        top_alt = np.argsort(alt_counts)[::-1][:5]
        for idx in top_alt:
            if alt_counts[idx] == 0:
                continue
            f.write(f"  {names[int(idx)]}: {int(alt_counts[int(idx)])}\n")

        # Sample winner RGBs
        f.write("\nSample WINNER RGBs:\n")
        m = w_pixels.shape[0]
        take = min(sample_n, m)
        pick = np.random.choice(m, size=take, replace=False)
        for rgb in w_pixels[pick]:
            f.write(f"  {rgb.tolist()}\n")

def main():
    p = argparse.ArgumentParser(description="Debug visualizer for FindColorsModule classifier")
    p.add_argument("part", help="Part number like ABC-00123")
    p.add_argument("--out", default="debug_color_output", help="Output folder")
    p.add_argument("--base-dir", default=r"\\dsfile4\Litho\Approved PDF", help="Approved PDF base dir")
    p.add_argument("--no-downweight", action="store_true", help="Disable downweighting")
    p.add_argument("--max-pages", type=int, default=0, help="Limit pages (0 = all)")
    args = p.parse_args()

    # Use your existing defaults
    cfg = FC.ColorScanConfig()

    debug_part(
        args.part,
        args.out,
        base_dir=args.base_dir,
        config=cfg,
        apply_downweight=(not args.no_downweight),
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()
