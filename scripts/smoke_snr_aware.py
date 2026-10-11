"""
smoke_snr_aware.py — visual sanity check for the SNR-Aware enhancement.

Owner: Jason Johnson
Date: 2026-10-11

Purpose
-------
The SNR-Aware module shipped on 2026-10-03 (commit 6065f3c) with the
explicit caveat that "visual correctness of the enhancement on real
low-light images has not been verified end-to-end." This script closes
that follow-up: run a handful of low-light images through
SNRAwareEnhancement.enhance() and write per-image contact sheets plus
a combined grid for a bare eye test.

Run this from the REPO ROOT with the cs5243 conda env active:

    cd <repo-root>
    python smoke_snr_aware.py \
        --input-dir  data/lowlight \
        --output-dir experiments/smoke_snr_aware/run-01 \
        [--weights-path enhancements/snr_aware/weights/LOLv1.pth] \
        [--device cuda] \
        [--limit 10]

Outputs in --output-dir
-----------------------
    <stem>_snr.png       enhanced version of each input
    <stem>_compare.png   1x2 contact sheet: input | enhanced
    compare_all.png      2-row grid: row 1 = inputs, row 2 = enhanced

The script does NOT touch git. Move it into the repo at
experiments/smoke_snr_aware.py and commit only if you want to keep it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Repo-relative import. The script may live in scripts/ or anywhere else
# under the repo — anchor the import against the repo root, NOT the cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from enhancements.snr_aware import SNRAwareEnhancement  # noqa: E402


def discover_images(input_dir: Path, limit: int) -> list[Path]:
    """Find image files in input_dir; return up to `limit`, sorted by name."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    return paths[:limit]


def contact_sheet(in_bgr: np.ndarray, out_bgr: np.ndarray,
                  target_h: int = 400) -> np.ndarray:
    """1x2 BGR contact sheet. Each panel resized so its HEIGHT == target_h."""
    def resize(img):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (max(1, int(w * scale)), target_h),
                          interpolation=cv2.INTER_AREA)

    a = resize(in_bgr)
    b = resize(out_bgr)
    h = target_h
    w = max(a.shape[1], b.shape[1])
    gutter = 4

    canvas = np.zeros((h, w * 2 + gutter, 3), dtype=np.uint8)
    a_pad = np.zeros((h, w, 3), dtype=np.uint8)
    b_pad = np.zeros((h, w, 3), dtype=np.uint8)
    a_pad[:, (w - a.shape[1]) // 2:(w - a.shape[1]) // 2 + a.shape[1]] = a
    b_pad[:, (w - b.shape[1]) // 2:(w - b.shape[1]) // 2 + b.shape[1]] = b

    canvas[:, :w] = a_pad
    canvas[:, w + gutter:] = b_pad

    # White labels on top-left of each panel (no font file needed).
    cv2.putText(canvas, "INPUT", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "SNR-AWARE", (w + gutter + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def combined_grid(pairs: list[tuple[np.ndarray, np.ndarray]],
                  target_h: int = 300) -> np.ndarray:
    """2-row grid: row 1 = all inputs left-to-right, row 2 = all enhanced."""
    n = len(pairs)
    if n == 0:
        raise ValueError("No image pairs to grid-ify.")

    def resize(img):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (max(1, int(w * scale)), target_h),
                          interpolation=cv2.INTER_AREA)

    resized = [(resize(a), resize(b)) for a, b in pairs]
    col_width = max(max(a.shape[1], b.shape[1]) for a, b in resized)
    gutter = 4

    canvas_w = col_width * n + gutter * (n - 1)
    canvas_h = target_h * 2 + gutter
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, (a, b) in enumerate(resized):
        x = i * (col_width + gutter)
        ay = (col_width - a.shape[1]) // 2
        by = (col_width - b.shape[1]) // 2
        canvas[:target_h, x + ay:x + ay + a.shape[1]] = a
        canvas[target_h + gutter:, x + by:x + by + b.shape[1]] = b

    cv2.putText(canvas, "INPUTS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "SNR-AWARE", (10, target_h + gutter + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visual smoke test for SNR-Aware enhancement."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory of low-light input images (jpg/png/bmp/tif). "
             "Resolved relative to repo root if not absolute.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write enhanced PNGs + contact sheets + grid. "
             "Resolved relative to repo root if not absolute.",
    )
    parser.add_argument(
        "--weights-path", type=Path, default=None,
        help="Path to LOLv1.pth. If omitted, falls back to "
             "<repo-root>/enhancements/snr_aware/weights/LOLv1.pth.",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "cuda:0"],
        help="Inference device. Default: cpu (cuda is much faster).",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Max images to process. Default: 10.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve all paths relative to REPO_ROOT when not absolute, so the
    # script behaves the same regardless of which directory you invoke it
    # from.
    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else (REPO_ROOT / p)
    args.input_dir = _abs(args.input_dir)
    args.output_dir = _abs(args.output_dir)

    # Resolve weights path. Honour --weights-path if given, else fall back
    # to the module default under the repo root. FAIL LOUD if neither
    # exists on disk — vendoring recipe is in the README.
    if args.weights_path is not None:
        weights = _abs(args.weights_path)
    else:
        weights = REPO_ROOT / "enhancements" / "snr_aware" / "weights" / "LOLv1.pth"
    if not weights.is_file():
        print(f"ERROR: weights not found at {weights}", file=sys.stderr)
        print("  Vendoring recipe: enhancements/snr_aware/README.md", file=sys.stderr)
        return 1

    print(f"[smoke] device={args.device}  weights={weights}")
    enh = SNRAwareEnhancement(weights_path=str(weights), device=args.device)

    images = discover_images(args.input_dir, limit=args.limit)
    if not images:
        print(f"ERROR: no images found in {args.input_dir}", file=sys.stderr)
        return 1
    print(f"[smoke] {len(images)} image(s) in {args.input_dir}")
    print(f"[smoke] outputs will go to {args.output_dir}")

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    timings_ms: list[float] = []
    skipped = 0

    for img_path in images:
        print(f"[smoke] {img_path.name}", end="", flush=True)
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print("  WARN: cv2 failed to read, skipping.")
            skipped += 1
            continue

        # Warm-up not needed: the first call handles lazy model load.
        t0 = time.perf_counter()
        out = enh.enhance(bgr)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        timings_ms.append(ms)

        stem = img_path.stem
        cv2.imwrite(str(args.output_dir / f"{stem}_snr.png"), out)
        sheet = contact_sheet(bgr, out)
        cv2.imwrite(str(args.output_dir / f"{stem}_compare.png"), sheet)
        print(f"  {bgr.shape[1]}x{bgr.shape[0]} -> {out.shape[1]}x{out.shape[0]}"
              f"  ({ms:.1f} ms)")

        pairs.append((bgr, out))

    if not pairs:
        print("ERROR: no images processed.", file=sys.stderr)
        return 1

    grid = combined_grid(pairs)
    cv2.imwrite(str(args.output_dir / "compare_all.png"), grid)
    print(f"[smoke] wrote {args.output_dir / 'compare_all.png'}")

    n = len(timings_ms)
    sorted_ms = sorted(timings_ms)
    avg = sum(timings_ms) / n
    p50 = sorted_ms[n // 2]
    print(f"[smoke] latency  avg={avg:.1f} ms  median={p50:.1f} ms"
          f"  min={sorted_ms[0]:.1f} ms  max={sorted_ms[-1]:.1f} ms"
          f"  (n={n}, device={args.device})")
    if skipped:
        print(f"[smoke] skipped {skipped} unreadable image(s)")
    print(f"[smoke] outputs in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
