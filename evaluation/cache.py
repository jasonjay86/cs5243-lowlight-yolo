"""
Prediction cache for the evaluation harness.

Owner: Jason Johnson

Why this exists
---------------
The upstream evaluation script re-runs YOLO on every frame for every method
every time it is invoked. With a neural enhancement at ~2500 ms/image on CPU
(or ~100 ms on GPU) and a few hundred frames, that turns "change one number
in the report" into a long re-run.

This module stores the raw detections per method, once. Every statistic is
then computed from the cache, so the expensive part happens exactly once per
(run, method) pair.

Format
------
One .npz per method under <run>/predictions/:

    boxes     (D, 4) float32   xyxy in the on-disk pixel space
    confs     (D,)   float32   detection confidence
    classes   (D,)   int16     COCO class id
    img_idx   (D,)   int32     index into the frame list
    frame_ids (N,)   unicode   the frame paths, so the cache is self-describing

Alongside it, <method>.json holds the fingerprint of the inputs that
produced it. If the fingerprint does not match the current run, the cache is
stale and must be recomputed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MethodPredictions:
    """Detections for one enhancement method across all frames."""
    method: str
    boxes: np.ndarray     # (D, 4) xyxy
    confs: np.ndarray     # (D,)
    classes: np.ndarray   # (D,)
    img_idx: np.ndarray   # (D,)
    frame_ids: list[str]
    enh_ms: float = 0.0
    yolo_ms: float = 0.0
    n_frames: int = 0

    @property
    def n_detections(self) -> int:
        return int(self.boxes.shape[0])

    def per_image(self) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Group detections by frame index."""
        out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for i in range(self.n_frames):
            sel = self.img_idx == i
            out[i] = (self.boxes[sel], self.confs[sel], self.classes[sel])
        return out


class PredictionCache:
    """Reads and writes the per-method detection cache for one run."""

    def __init__(self, run_dir: Path, fingerprint: str,
                 legacy_fingerprints: list[str] | None = None):
        self.dir = Path(run_dir) / "predictions"
        self.fingerprint = fingerprint
        # Fingerprints from earlier schemes whose caches are still valid.
        # A cache written under one of these is accepted as current.
        self.legacy_fingerprints = list(legacy_fingerprints or [])

    # ------------------------------------------------------------------
    def _paths(self, method: str) -> tuple[Path, Path]:
        safe = method.replace("/", "_")
        return self.dir / f"{safe}.npz", self.dir / f"{safe}.json"

    def has(self, method: str) -> bool:
        npz_path, meta_path = self._paths(method)
        if not (npz_path.is_file() and meta_path.is_file()):
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        stored = meta.get("fingerprint")
        if stored == self.fingerprint:
            return True
        return stored in self.legacy_fingerprints

    def why_stale(self, method: str) -> str:
        """Explain why a method's cache is not current, or ''. """
        npz_path, meta_path = self._paths(method)
        if not npz_path.is_file():
            return "no cached predictions"
        if not meta_path.is_file():
            return "cache metadata missing"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return "cache metadata unreadable"
        stored = meta.get("fingerprint")
        if stored == self.fingerprint:
            return ""
        if stored in self.legacy_fingerprints:
            return ""
        return (f"fingerprint changed (cache {stored}, "
                f"run {self.fingerprint})")

    def save(self, preds: MethodPredictions) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        npz_path, meta_path = self._paths(preds.method)
        np.savez_compressed(
            npz_path,
            boxes=preds.boxes.astype(np.float32),
            confs=preds.confs.astype(np.float32),
            classes=preds.classes.astype(np.int16),
            img_idx=preds.img_idx.astype(np.int32),
            frame_ids=np.array(preds.frame_ids, dtype=object).astype("U"),
        )
        meta = {
            "fingerprint": self.fingerprint,
            "method": preds.method,
            "n_detections": preds.n_detections,
            "n_frames": preds.n_frames,
            "enh_ms": preds.enh_ms,
            "yolo_ms": preds.yolo_ms,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, method: str) -> MethodPredictions | None:
        if not self.has(method):
            return None
        npz_path, meta_path = self._paths(method)
        try:
            data = np.load(npz_path, allow_pickle=False)
        except Exception:
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return MethodPredictions(
            method=method,
            boxes=data["boxes"],
            confs=data["confs"],
            classes=data["classes"],
            img_idx=data["img_idx"],
            frame_ids=[str(s) for s in data["frame_ids"].tolist()],
            enh_ms=float(meta.get("enh_ms", 0.0)),
            yolo_ms=float(meta.get("yolo_ms", 0.0)),
            n_frames=int(meta.get("n_frames", 0)),
        )

    # ------------------------------------------------------------------
    def available_methods(self) -> list[str]:
        """Methods whose cache is present AND current."""
        return sorted(m.stem for m in self.dir.glob("*.npz")) if self.dir.is_dir() else []

    def stale_methods(self) -> list[str]:
        """Methods with a cache file that does not match this fingerprint."""
        out = []
        if not self.dir.is_dir():
            return out
        for npz_path in self.dir.glob("*.npz"):
            method = npz_path.stem
            if not self.has(method):
                out.append(method)
        return sorted(out)

    def clear(self) -> None:
        if self.dir.is_dir():
            for p in self.dir.glob("*"):
                p.unlink()
