"""Low-light image -> CLAHE enhancement -> YOLO detection.

Usage (from repo root):
    python scripts/run_clahe_yolo.py path/to/lowlight.jpg
    python scripts/run_clahe_yolo.py path/to/lowlight.jpg --compare-raw

Writes the annotated result to results/figures/<name>_clahe_detected.png and
prints detections. With --compare-raw it also runs YOLO on the un-enhanced
image so you can see what CLAHE changed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Allow running from anywhere: put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import enhancements.clahe  # noqa: E402,F401  -- triggers @register
from detection.yolo_detector import YOLODetector  # noqa: E402
from enhancements.base import get_enhancement  # noqa: E402


def print_detections(title: str, detections: list[dict]) -> None:
    print(f"{title}: {len(detections)} detection(s)")
    for d in detections:
        print(f"  {d['label']:12s} {d['confidence']:.3f} {[round(v, 1) for v in d['bbox']]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="path to a low-light image")
    ap.add_argument("--compare-raw", action="store_true", help="also detect on the un-enhanced image")
    args = ap.parse_args()

    src = Path(args.image)
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read image: {src}")

    enhancer = get_enhancement("clahe")
    detector = YOLODetector()

    enhanced = enhancer.enhance(img)                 # low-light in -> enhanced out
    detections = detector.detect(enhanced)           # enhanced -> YOLO
    print_detections("CLAHE + YOLO", detections)
    if args.compare_raw:
        print_detections("raw + YOLO  ", detector.detect(img))

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}_clahe_detected.png"
    cv2.imwrite(str(out_path), detector.model(enhanced)[0].plot())
    print(f"Saved annotated result to {out_path}")
    print(f"CLAHE latency: {enhancer.avg_latency_ms_per_image():.1f} ms/image (CPU, this machine)")


if __name__ == "__main__":
    main()
