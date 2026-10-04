# CLAHE (Classical)

**Owner:** Krutin Patel

**Contrast Limited Adaptive Histogram Equalization** — the classical baseline
for low-light image enhancement. Fast, no learned parameters, deterministic.

## Files

- `__init__.py` — `CLAHEEnhancement`, implements `Enhancement`, registered as `"clahe"`.
- `image_enhance.py` — the algorithm (`low_light_enhance`), adapted from
  [ThomasWangWeiHong/Low-Light-Image-Enhancement-CLAHE-Based](https://github.com/ThomasWangWeiHong/Low-Light-Image-Enhancement-CLAHE-Based).
- `scripts/test_clahe.py` — smoke test (contract, determinism, IR frames, edge cases).
- `scripts/run_clahe_yolo.py` — low-light image → CLAHE → YOLO, saves the annotated result.

## Pipeline (`image_enhance.py`)

Works on the HSV **V** channel only (H and S untouched):

1. invert V
2. CLAHE on the inverted V (`clip_limit=2.0`, `tile_grid_size=(8, 8)` by default)
3. gamma (=5) boost → 7×7 elliptical morphological top-hat
4. PCA fusion of the CLAHE output and the top-hat output
5. invert back and write into V

Input/output follow `enhancements/base.py`: HxWx3 uint8 BGR in → same out.

## Modes

- `CLAHEEnhancement()` / `mode="fusion"` — the pipeline above. **This is the
  `clahe` row in result tables.**
- `CLAHEEnhancement(mode="lab")` — plain CLAHE on the L channel of LAB.
  Reports as `clahe_lab`; opt-in, not registered separately.

## Run it

```bash
python scripts/test_clahe.py
python scripts/run_clahe_yolo.py path/to/lowlight.jpg --compare-raw
```

## Known behaviour (read before trusting results)

Observed on **one** synthetic test image (a well-lit photo darkened by gamma +
noise — not real camera-trap data), so treat as a risk to check, not a finding:

- **Colour input:** `fusion` brightens V but keeps the original hue/saturation,
  which are noise in very dark pixels, so it can introduce blotchy colour
  artifacts. Plain `lab` mode did not.
- **IR-style grayscale input:** no colour artifacts, but `fusion` output looked
  washed-out (mean brightness 17 → 118) and YOLO lost two detections that raw
  and `lab` kept.

Camera-trap night frames are mostly IR grayscale, so the second point matters
most. Check on real CCT / Snapshot Serengeti frames and look at the
false-positive-on-empty-frames metric before settling on a default.

## Notes for the eval harness

- **Deterministic** — same image always produces the same output.
- `flops_per_megapixel()` returns `None` on purpose: this is histogram /
  morphology arithmetic, not a network, so a FLOPs figure isn't meaningful.
  Report latency.
- `avg_latency_ms_per_image()` is a running mean of CPU wall-clock time per
  `enhance()` call. In a single-core sandbox the fusion pipeline took roughly
  50 ms at ~1080×810 — re-measure on the reference machine.
