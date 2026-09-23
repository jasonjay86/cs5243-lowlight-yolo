# CLAHE (Classical)

**Owner:** (CLAHE lead — Krutin / Richard)

**Contrast Limited Adaptive Histogram Equalization** — the classical baseline
for low-light image enhancement. Fast, no learned parameters, deterministic.

## Files

- `__init__.py` — exports `CLAHEEnhancement` implementing `Enhancement`.
- `tune.py` *(optional)* — script to sweep clip-limit / tile-grid hyperparameters
  on a held-out subset.

## Implementation reference

`cv2.createCLAHE(clipLimit=…, tileGridSize=(…))` applied per-channel on the
**L channel of LAB** (not on RGB directly — that introduces color shifts).

## Notes for the eval harness

- This method is **deterministic** — same image always produces same output.
- Latency should be **<5 ms/image** at 1024×1024 on a modern CPU. Record the
  actual number on the reference GPU/CPU and store via
  `flops_per_megapixel()` / `avg_latency_ms_per_image()`.
