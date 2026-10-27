"""
Running enhancements through the detector and caching the detections.

Owner: Jason Johnson

This is the expensive part of the harness: for each method, each frame is
enhanced and then passed to YOLO. The results are written to the prediction
cache so that every statistic can be recomputed without repeating the work.

The detector is called directly via `model(...)` rather than
`YOLODetector.detect(...)`, because detect() drops everything below
Ultralytics' default threshold, which would make the PR curve incomplete.
This matches the upstream evaluation script.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# OpenMP-on-Windows: see the note in evaluation/__init__.py. This module
# triggers the torch import (via the enhancement modules) so the flag is set
# here too, in case predict.py is imported without the package __init__
# having run first.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from evaluation.cache import MethodPredictions, PredictionCache


def _load_model(yolo_weights: str, device: str):
    """Load the YOLO model, falling back to Ultralytics' own download."""
    from ultralytics import YOLO
    p = Path(yolo_weights)
    if p.is_file():
        model = YOLO(str(p))
    else:
        # Let Ultralytics fetch it (or use its cache) by name.
        model = YOLO(p.name)
    if device.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                model.to("cuda")
            else:
                print(f"[predict] CUDA requested but not available; using CPU.")
        except Exception as e:
            print(f"[predict] could not move model to CUDA ({e}); using CPU.")
    return model


def _get_enhancement(method: str, repo_root: Path):
    """Instantiate a registered enhancement by name."""
    # The enhancements package lives at the repo root. Add it to sys.path so
    # the harness works no matter which directory the script was launched
    # from (repo root, scripts/, or tests/).
    root = str(Path(repo_root))
    if root not in sys.path:
        sys.path.insert(0, root)

    from enhancements.base import get_enhancement
    # Import the known enhancement modules so they register themselves.
    # A module that cannot be imported (missing weights, missing
    # dependency) is reported and skipped: one broken module must not
    # prevent the other methods from running.
    for mod in ("raw", "clahe", "snr_aware", "retinexformer"):
        try:
            __import__(f"enhancements.{mod}")
        except Exception as e:
            print(f"[predict] (enhancements.{mod} not importable: {e})")
    try:
        return get_enhancement(method)
    except KeyError:
        raise SystemExit(
            f"ERROR: enhancement '{method}' is not registered. A module may "
            f"have failed to import (see any warning above)."
        )


def run_method(method: str, frames, model, imgsz: int, conf_floor: float,
               repo_root: Path, device: str = "cpu",
               yolo_weights: str = "detection/yolov8n.pt",
               progress_every: int = 25) -> MethodPredictions:
    """Enhance every frame, detect on the result, and return the detections.

    Latency figures skip the first frame, which absorbs model-load and
    first-call warm-up.
    """
    import cv2

    enh = _get_enhancement(method, repo_root)
    all_boxes, all_confs, all_classes, all_img = [], [], [], []
    enh_times, yolo_times = [], []

    for k, fr in enumerate(frames, 1):
        img = cv2.imread(fr.path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[predict] {method}: could not read {fr.path}, skipping")
            continue

        t0 = time.perf_counter()
        out = enh.enhance(img)
        enh_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        res = model(out, conf=conf_floor, imgsz=imgsz, verbose=False)[0]
        yolo_times.append((time.perf_counter() - t0) * 1000)

        boxes = res.boxes.xyxy.cpu().numpy().reshape(-1, 4)
        confs = res.boxes.conf.cpu().numpy().reshape(-1)
        classes = res.boxes.cls.cpu().numpy().reshape(-1).astype(np.int16)

        all_boxes.append(boxes)
        all_confs.append(confs)
        all_classes.append(classes)
        all_img.append(np.full(len(confs), k - 1, dtype=np.int32))

        if progress_every and k % progress_every == 0:
            print(f"[predict] {method}: {k}/{len(frames)}", flush=True)

    skip = 1 if len(enh_times) > 5 else 0
    enh_ms = float(np.mean(enh_times[skip:])) if enh_times else 0.0
    yolo_ms = float(np.mean(yolo_times[skip:])) if yolo_times else 0.0

    return MethodPredictions(
        method=method,
        boxes=np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4), np.float32),
        confs=np.concatenate(all_confs) if all_confs else np.zeros(0, np.float32),
        classes=np.concatenate(all_classes) if all_classes else np.zeros(0, np.int16),
        img_idx=np.concatenate(all_img) if all_img else np.zeros(0, np.int32),
        frame_ids=[f.path for f in frames],
        enh_ms=enh_ms,
        yolo_ms=yolo_ms,
        n_frames=len(frames),
    )


def predictions_to_per_image(preds: MethodPredictions) -> list:
    """Convert the flat detection arrays into a per-frame list of
    (boxes, confs) tuples aligned with the frame order."""
    out = []
    for i in range(preds.n_frames):
        sel = preds.img_idx == i
        out.append((preds.boxes[sel], preds.confs[sel]))
    return out
