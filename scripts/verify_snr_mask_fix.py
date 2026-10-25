"""
verify_snr_mask_fix.py — confirm the SNR mask fix changed model behaviour.

Owner: Jason Johnson
Date: 2026-10-25

WHAT THIS VERIFIES
------------------
Before the fix, model.py defaulted a missing SNR mask to all-ones. That
routed every pixel down the "light" reconstruction branch and multiplied the
transformer output by zero, so the SNR-aware path never ran. Symptoms:
SNR-Aware outputs were very bright (mean ~160-200 vs ~36-70 raw) and
over-cooked.

This script checks three things:

  1. The mask is real. compute_snr_mask() on a synthetic image returns a
     float32 map in [0,1] where bright/clean regions score higher than
     dark/noisy ones.

  2. The network now REQUIRES a mask. Calling model(x) with no mask raises
     ValueError instead of silently defaulting to all-ones.

  3. The mask changes the output. Running the same image with (a) a
     computed SNR mask and (b) a forced all-ones mask produces DIFFERENT
     output images. If they are identical, the fix did not take effect.

WHAT SUCCESS LOOKS LIKE
-----------------------
  [1] PASS  mask shape/dtype/range ok
  [2] PASS  model rejects a missing mask
  [3] PASS  masked output differs from all-ones output (max abs delta > 0)
  [3] mean brightness with real mask vs all-ones: <numbers>

Then it re-runs the enhancement on the smoke-test frames and reports mean
brightness before/after so you can compare against the midpoint numbers
(raw ~36-70, SNR-Aware ~160-200).

Run from the repo root with the cs5243 conda env active:

    python scripts/verify_snr_mask_fix.py \
        --input-dir data/smoke_5 \
        [--weights-path enhancements/snr_aware/weights/LOLv1.pth]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from enhancements.snr_aware import SNRAwareEnhancement, compute_snr_mask  # noqa: E402
from enhancements.snr_aware.model import SNRAwareGenerator  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def mean_brightness(img: np.ndarray) -> float:
    """Mean of the grayscale luma, 0-255. Matches the reporting style used
    on the CLAHE slide ('Mean brightness 25.7 to 108.9')."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def check_mask_is_real() -> bool:
    print("\n[1] Checking compute_snr_mask() ...")
    rng = np.random.default_rng(0)
    img = np.full((120, 160, 3), 30, dtype=np.uint8)
    img[:, 100:140, :] = 220                     # clean bright patch
    img = np.clip(img.astype(np.int16) + rng.normal(0, 12, img.shape),
                  0, 255).astype(np.uint8)

    m = compute_snr_mask(img)
    bright = float(m[:, 100:140].mean())
    dark = float(m[:, 0:60].mean())

    ok = True
    if m.dtype != np.float32:
        print(f"    FAIL  dtype is {m.dtype}, expected float32")
        ok = False
    if m.shape != (120, 160):
        print(f"    FAIL  shape is {m.shape}, expected (120, 160)")
        ok = False
    if not (0.0 <= float(m.min()) and float(m.max()) <= 1.0):
        print(f"    FAIL  range [{m.min():.6f}, {m.max():.6f}] outside [0,1]")
        ok = False
    if bright <= dark:
        print(f"    FAIL  bright patch mask ({bright:.6f}) not > "
              f"dark region mask ({dark:.6f})")
        ok = False
    if ok:
        print(f"    PASS  shape={m.shape} dtype={m.dtype} "
              f"range=[{m.min():.6f}, {m.max():.6f}]")
        print(f"          bright-patch mean mask={bright:.6f}  "
              f"dark-region mean mask={dark:.6f}")
    return ok


def check_model_requires_mask(weights: Path) -> bool:
    print("\n[2] Checking model rejects a missing mask ...")
    model = SNRAwareGenerator.from_checkpoint(str(weights), device="cpu")
    x = torch.zeros(1, 3, 32, 32, dtype=torch.float32)
    try:
        model(x)
    except ValueError as e:
        print(f"    PASS  ValueError raised: {str(e)[:70]}...")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    FAIL  raised {type(e).__name__} instead of ValueError: {e}")
        return False
    print("    FAIL  no exception raised; the all-ones default is still there")
    return False


def check_mask_changes_output(weights: Path) -> bool:
    print("\n[3] Checking the mask actually changes the output ...")
    model = SNRAwareGenerator.from_checkpoint(str(weights), device="cpu")

    rng = np.random.default_rng(1)
    img = np.full((64, 64, 3), 28, dtype=np.uint8)
    img[:, 32:56, :] = 200
    img = np.clip(img.astype(np.int16) + rng.normal(0, 10, img.shape),
                  0, 255).astype(np.uint8)

    x = torch.from_numpy(
        (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    )

    m_real = torch.from_numpy(compute_snr_mask(img))[None, None]
    m_ones = torch.ones(1, 1, 64, 64, dtype=torch.float32)

    with torch.no_grad():
        y_real = model(x, m_real)
        y_ones = model(x, m_ones)

    delta = float((y_real - y_ones).abs().max())
    if delta <= 0:
        print(f"    FAIL  masked output identical to all-ones output "
              f"(max abs delta = {delta})")
        return False
    print(f"    PASS  masked output differs from all-ones output "
          f"(max abs delta = {delta:.6f})")
    print("          The SNR-aware path is now live.")
    return True


def brightness_report(input_dir: Path, weights: Path) -> None:
    print("\n[4] Mean brightness on the smoke-test frames "
          "(compare against midpoint numbers) ...")
    images = sorted(p for p in input_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"    (no images in {input_dir}, skipping)")
        return

    enh = SNRAwareEnhancement(weights_path=str(weights), device="cpu")
    print(f"    {'frame':<34s} {'raw':>8s} {'snr(mask)':>11s}")
    raws, masked = [], []
    for p in images:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        out = enh.enhance(bgr)
        rb, mb = mean_brightness(bgr), mean_brightness(out)
        raws.append(rb)
        masked.append(mb)
        print(f"    {p.name[:32]:<34s} {rb:8.1f} {mb:11.1f}")

    if raws:
        print(f"    {'MEAN':<34s} {np.mean(raws):8.1f} {np.mean(masked):11.1f}")
        print()
        print("    Midpoint numbers for reference:")
        print("      raw ~36-70, SNR-Aware (pre-fix, all-ones mask) ~160-200")
        print("    A masked mean in the 60-120 range suggests the transformer")
        print("    path is now contributing instead of being zeroed out.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=None,
                    help="Optional dir of frames for the brightness report.")
    ap.add_argument("--weights-path", type=Path, default=None,
                    help="Default: <repo-root>/enhancements/snr_aware/weights/LOLv1.pth")
    args = ap.parse_args()

    weights = args.weights_path or (REPO_ROOT / "enhancements" / "snr_aware"
                                    / "weights" / "LOLv1.pth")
    if not weights.is_file():
        print(f"ERROR: weights not found at {weights}", file=sys.stderr)
        print("  Vendoring recipe: enhancements/snr_aware/README.md", file=sys.stderr)
        return 1

    print(f"weights: {weights}")
    results = [
        check_mask_is_real(),
        check_model_requires_mask(weights),
        check_mask_changes_output(weights),
    ]

    if args.input_dir is not None:
        in_dir = args.input_dir if args.input_dir.is_absolute() else (REPO_ROOT / args.input_dir)
        brightness_report(in_dir, weights)

    print("\n" + "=" * 60)
    if all(results):
        print("ALL CHECKS PASSED — the SNR mask fix is in place.")
        return 0
    print("SOME CHECKS FAILED — see above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
