"""Smoke test for enhancements/clahe (no YOLO weights or dataset needed).

Usage (from repo root):
    python scripts/test_clahe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Allow `python scripts/test_clahe.py` from anywhere: put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import enhancements.clahe  # noqa: E402,F401  -- triggers @register
from enhancements.base import available_enhancements, get_enhancement  # noqa: E402
from enhancements.clahe import CLAHEEnhancement  # noqa: E402


def synthetic_dark_frame(h: int = 480, w: int = 640, gray: bool = False) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = np.full((h, w), 18.0, dtype=np.float32)
    cv2.ellipse(base, (w // 2, h // 2), (90, 55), 0, 0, 360, 32, -1)  # dim "animal"
    base += rng.normal(0, 6, (h, w)).astype(np.float32)
    g = np.clip(base, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if gray:
        return img
    img = img.astype(np.float32)
    img[:, :, 0] *= 1.3  # blue-ish tint so colour handling is exercised
    return np.clip(img, 0, 255).astype(np.uint8)


def check(cond: bool, msg: str) -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        sys.exit(1)


def main() -> None:
    check("clahe" in available_enhancements(), "'clahe' is in the registry")

    enh = get_enhancement("clahe")
    check(enh.name() == "clahe" and enh.mode == "fusion", "default is the image_enhance.py pipeline, name() == 'clahe'")
    check(enh.avg_latency_ms_per_image() is None, "latency is None before any call")

    img = synthetic_dark_frame()
    out = enh.enhance(img)
    check(out.shape == img.shape and out.dtype == np.uint8, "same shape/dtype as input")
    check(np.array_equal(out, enh.enhance(img)), "deterministic (same input -> same output)")
    check(out.mean() > img.mean(), f"brightness increased (mean {img.mean():.1f} -> {out.mean():.1f})")
    check(enh.avg_latency_ms_per_image() > 0, "latency is recorded after calls")

    ir = synthetic_dark_frame(gray=True)  # IR-style: B == G == R
    ir_out = enh.enhance(ir)
    drift = int(np.abs(ir_out[:, :, 0].astype(int) - ir_out[:, :, 2].astype(int)).max())
    check(drift <= 2, f"IR (B==G==R) frame stays grayscale (max channel drift {drift})")

    flat = enh.enhance(np.full((64, 64, 3), 40, np.uint8))
    check(flat.shape == (64, 64, 3), "flat image (degenerate covariance) doesn't crash")

    for bad, exc in [(img.astype(np.float32), TypeError), (img[:, :, 0], ValueError)]:
        try:
            enh.enhance(bad)
            check(False, f"rejects bad input ({exc.__name__})")
        except exc:
            check(True, f"rejects bad input ({exc.__name__})")

    lab = CLAHEEnhancement(mode="lab")
    lab_out = lab.enhance(img)
    check(lab.name() == "clahe_lab", "mode='lab' name() == 'clahe_lab'")
    check(lab_out.shape == img.shape and lab_out.mean() > img.mean(), "mode='lab' brightens, same shape")
    try:
        CLAHEEnhancement(mode="nope")
        check(False, "rejects unknown mode")
    except ValueError:
        check(True, "rejects unknown mode")

    print("All checks passed.")


if __name__ == "__main__":
    main()
