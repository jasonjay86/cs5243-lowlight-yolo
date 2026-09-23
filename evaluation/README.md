# evaluation

**Owner:** (unassigned — depends on detection/contract being stable; grab after raw + clahe are running)

Compares detection results across enhancement methods and reports:

- **Δ mAP** (enhanced − raw baseline) per method — the primary result.
- **False-positive rate** on "empty" frames per method (for hallucinated detections).
- **Latency / GFLOPs** for enhancement stage vs. YOLO stage, attributed separately.

## Inputs

- Ground-truth annotations (YOLO format, in `data/`).
- Per-method predictions produced by `detection/` runs (under `experiments/<run>/predictions/`).

## Outputs

- `results/method_comparison.csv` — one row per (method, metric).
- `results/figures/` — bar plots, Δ mAP chart.
