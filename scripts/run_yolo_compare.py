"""
run_yolo_compare.py — raw vs CLAHE vs SNR-Aware, same YOLO pass.

Owner: Jason Johnson
Date: 2026-10-11

Purpose
-------
The project question is: which low-light enhancement (classical CLAHE vs
neural SNR-Aware) most improves YOLO wildlife detection on nocturnal
camera-trap imagery? This script produces a per-image, three-method
comparison on the same YOLO pass so that counts and confidences can be
read off directly — no per-method reruns, no per-method weighting.

What it does
------------
For each image in --input-dir:
    1. YOLO detect on RAW input                          → list[detection]
    2. CLAHE enhance     → YOLO detect on enhanced       → list[detection]
    3. SNR-Aware enhance → YOLO detect on enhanced       → list[detection]

Outputs in --output-dir
    raw/<name>.png                — annotated raw detections
    clahe/<name>_in.png, _out.png — CLAHE inputs and annotated outputs
    snr_aware/<name>_in.png, ...  — SNR-Aware inputs and annotated outputs
    compare/<name>.png            — 3-row grid (raw | CLAHE | SNR-Aware) with
                                    boxes drawn on each
    detections.csv                — one row per (image, method, class)
    summary.txt                   — counts/confidences per image

Run from the repo root with the cs5243 conda env active:

    python scripts/run_yolo_compare.py \
        --input-dir  data/smoke_5 \
        --output-dir experiments/yolo_compare/run-01 \
        [--snr-weights enhancements/snr_aware/weights/LOLv1.pth] \
        [--device cpu]

What this is NOT
----------------
- Not the project eval harness. It does not compute mAP (no labels).
- Not a benchmark. YOLO warm-up cost is amortised across all three
  methods per image, so the per-method latency is comparable; absolute
  wall-clock is dominated by the model load on first call.

Caveats surfaced from the upstream CLAHE README: CLAHE in "fusion" mode
(its default) was observed to wash out IR-style grayscale frames
(mean brightness 17 → 118) and YOLO lost two detections that raw kept.
Expect the "clahe" row in the grid to over-brighten grayscale camera-trap
frames; that's a known, not a finding.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path

# OpenMP-on-Windows: PyTorch and Ultralytics both ship against an OpenMP
# runtime, and on a stock conda+pip install the second one to initialize
# refuses to load (OMP Error #15). Setting KMP_DUPLICATE_LIB_OK=TRUE is
# Intel's documented workaround for the inference path (vs. training).
# Applied BEFORE torch / ultralytics are imported so both see the flag.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

# Anchor repo root on sys.path regardless of cwd. Mirrors the smoke test.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import enhancements.clahe      # noqa: E402,F401  -- triggers @register
import enhancements.snr_aware  # noqa: E402,F401  -- triggers @register
from detection.yolo_detector import YOLODetector     # noqa: E402
from enhancements.base import get_enhancement        # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def discover_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def annotate(image: np.ndarray, detections: list[dict], detector) -> np.ndarray:
    """Use the Ultralytics plot() to get a labelled annotated image."""
    if not detections:
        # plot() wants a non-empty results object; emulate one for visual.
        empty = np.zeros_like(image)
        return cv2.hconcat([image, empty])  # crude visual fallback
    # Re-running the model with no augmentations is the documented path to
    # get a `Results` object with `.plot()`. Cheap on a single image.
    results = detector.model(image, verbose=False)
    return results[0].plot()


def three_row_grid(
    raw_bgr: np.ndarray,
    raw_det: list[dict],
    clahe_bgr: np.ndarray,
    clahe_det: list[dict],
    snr_bgr: np.ndarray,
    snr_det: list[dict],
    detector,
    target_h: int = 360,
) -> np.ndarray:
    """3-row grid: row 1 = raw + boxes, row 2 = CLAHE + boxes,
    row 3 = SNR-Aware + boxes. All panels same height.
    """
    def annotate_or_fallback(img, dets):
        # Use detector.model.plot() for the visual. If we don't want to
        # re-run the model just for boxes, build a manual drawing.
        return _draw_boxes(img, dets)

    rows = [
        ("RAW + YOLO",      annotate_or_fallback(raw_bgr,   raw_det)),
        ("CLAHE + YOLO",    annotate_or_fallback(clahe_bgr, clahe_det)),
        ("SNR-AWARE + YOLO", annotate_or_fallback(snr_bgr,   snr_det)),
    ]

    def resize(img):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (max(1, int(w * scale)), target_h),
                          interpolation=cv2.INTER_AREA)

    resized = [(label, resize(img)) for label, img in rows]
    col_w = max(img.shape[1] for _, img in resized)
    gutter = 6
    canvas_w = col_w
    canvas_h = target_h * 3 + gutter * 2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, (label, img) in enumerate(resized):
        y = i * (target_h + gutter)
        x = (col_w - img.shape[1]) // 2
        canvas[y:y + target_h, x:x + img.shape[1]] = img
        cv2.putText(canvas, label, (10, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return canvas


def _draw_boxes(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw labelled bounding boxes without re-running the model."""
    out = image.copy()
    palette = [
        (0, 255, 0),    # green
        (255, 128, 0),  # orange
        (0, 200, 255),  # yellow
        (255, 0, 200),  # magenta
        (200, 255, 0),  # cyan-ish
    ]
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
        color = palette[i % len(palette)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d['label']} {d['confidence']:.2f}"
        # Background bar behind the text for readability.
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)),
                      (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def detection_rows(image_name: str, method: str,
                   detections: list[dict]) -> list[dict]:
    return [
        {
            "image":      image_name,
            "method":     method,
            "label":      d["label"],
            "confidence": f"{d['confidence']:.3f}",
            "x1": f"{d['bbox'][0]:.1f}",
            "y1": f"{d['bbox'][1]:.1f}",
            "x2": f"{d['bbox'][2]:.1f}",
            "y2": f"{d['bbox'][3]:.1f}",
        }
        for d in detections
    ]


def format_summary(name: str,
                   raw_det: list[dict],
                   clahe_det: list[dict],
                   snr_det: list[dict]) -> str:
    def fmt(prefix: str, dets: list[dict]) -> str:
        cnt = Counter(d["label"] for d in dets)
        n = len(dets)
        conf = (sum(d["confidence"] for d in dets) / n) if n else 0.0
        top = ", ".join(f"{k}:{v}" for k, v in cnt.most_common())
        return (f"  {prefix:<14s} n={n:<3d} mean_conf={conf:.3f}  {top}")

    out = [f"[{name}]"]
    out.append(fmt("RAW",        raw_det))
    out.append(fmt("CLAHE",      clahe_det))
    out.append(fmt("SNR-AWARE",  snr_det))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir",   type=Path, required=True)
    ap.add_argument("--output-dir",  type=Path, required=True)
    ap.add_argument("--snr-weights", type=Path, default=None,
                    help="Optional explicit path to LOLv1.pth. Default: "
                         "<repo-root>/enhancements/snr_aware/weights/LOLv1.pth.")
    ap.add_argument("--clahe-mode",  choices=["fusion", "lab"], default="fusion",
                    help="CLAHE variant. Default 'fusion' (Krutin's registered "
                         "default). Use 'lab' to A/B against plain LAB-CLAHE.")
    ap.add_argument("--device",      default="cpu",
                    choices=["cpu", "cuda", "cuda:0"])
    ap.add_argument("--conf",        type=float, default=0.25,
                    help="YOLO confidence threshold. Default 0.25 matches the "
                         "Ultralytics default.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N images (0 = all). Default 0.")
    args = ap.parse_args()

    # Resolve relative paths against the repo root, mirroring the smoke test.
    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else (REPO_ROOT / p)
    args.input_dir = _abs(args.input_dir)
    args.output_dir = _abs(args.output_dir)
    if args.snr_weights is not None:
        args.snr_weights = _abs(args.snr_weights)

    # Output directories.
    out_raw     = args.output_dir / "raw"
    out_clahe   = args.output_dir / "clahe"
    out_snr     = args.output_dir / "snr_aware"
    out_compare = args.output_dir / "compare"
    for d in (out_raw, out_clahe, out_snr, out_compare):
        d.mkdir(parents=True, exist_ok=True)

    images = discover_images(args.input_dir)
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        print(f"ERROR: no images in {args.input_dir}", file=sys.stderr)
        return 1
    print(f"[yolo-compare] {len(images)} image(s) in {args.input_dir}")
    print(f"[yolo-compare] outputs -> {args.output_dir}")

    # Build the detection + enhancement stack ONCE. Models load lazily on
    # first call; the detector carries its own weights as yolov8n.pt.
    detector = YOLODetector()
    clahe = get_enhancement("clahe", mode=args.clahe_mode)
    snr   = get_enhancement("snr-aware", weights_path=str(args.snr_weights) if args.snr_weights else None)
    print(f"[yolo-compare] CLAHE mode = {args.clahe_mode}")
    print(f"[yolo-compare] SNR-Aware weights = {args.snr_weights or '(module default)'}")

    csv_rows: list[dict] = []
    summary_chunks: list[str] = []

    for img_path in images:
        print(f"[yolo-compare] {img_path.name}", end="", flush=True)
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print("  WARN: cv2 failed to read, skipping.")
            continue

        # 1. RAW pass.
        raw_det = detector.detect(bgr)
        cv2.imwrite(str(out_raw / f"{img_path.stem}.png"), _draw_boxes(bgr, raw_det))

        # 2. CLAHE pass.
        t0 = time.perf_counter()
        clahe_bgr = clahe.enhance(bgr)
        clahe_dt = (time.perf_counter() - t0) * 1000.0
        clahe_det = detector.detect(clahe_bgr)
        cv2.imwrite(str(out_clahe / f"{img_path.stem}_in.png"),  clahe_bgr)
        cv2.imwrite(str(out_clahe / f"{img_path.stem}_out.png"), _draw_boxes(clahe_bgr, clahe_det))

        # 3. SNR-Aware pass.
        t0 = time.perf_counter()
        snr_bgr = snr.enhance(bgr)
        snr_dt = (time.perf_counter() - t0) * 1000.0
        snr_det = detector.detect(snr_bgr)
        cv2.imwrite(str(out_snr / f"{img_path.stem}_in.png"),  snr_bgr)
        cv2.imwrite(str(out_snr / f"{img_path.stem}_out.png"), _draw_boxes(snr_bgr, snr_det))

        # 4. 3-row comparison grid for this image.
        grid = three_row_grid(bgr, raw_det,
                              clahe_bgr, clahe_det,
                              snr_bgr, snr_det,
                              detector)
        cv2.imwrite(str(out_compare / f"{img_path.stem}.png"), grid)

        # 5. CSV + summary.
        csv_rows.extend(detection_rows(img_path.name, "raw",       raw_det))
        csv_rows.extend(detection_rows(img_path.name, "clahe",     clahe_det))
        csv_rows.extend(detection_rows(img_path.name, "snr-aware", snr_det))
        summary_chunks.append(format_summary(img_path.name, raw_det, clahe_det, snr_det))

        print(f"  raw={len(raw_det)}  clahe={len(clahe_det)} ({clahe_dt:.0f}ms)"
              f"  snr={len(snr_det)} ({snr_dt:.0f}ms)")

    # Write CSV.
    csv_path = args.output_dir / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "method", "label",
                                          "confidence", "x1", "y1", "x2", "y2"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"[yolo-compare] wrote {csv_path}  ({len(csv_rows)} rows)")

    # Write summary.
    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text("\n\n".join(summary_chunks) + "\n",
                            encoding="utf-8")
    print(f"[yolo-compare] wrote {summary_path}")

    if clahe.avg_latency_ms_per_image() is not None:
        print(f"[yolo-compare] CLAHE avg latency    : {clahe.avg_latency_ms_per_image():.1f} ms/image")
    print(f"[yolo-compare] SNR-Aware device latency: this run only, see per-image log")
    print(f"[yolo-compare] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
