# results

Aggregated CSV summaries from the **sampled** evaluation runs, plus training
outputs. Small text files are committed; figures, predictions and logs are not.

## Read this first

The CSVs in this directory come from 100-frame stratified samples. They were
useful while the harness was being built, and they are still a reasonable sanity
check, but they are **not** the reported result.

The reported result is [`experiments/eval-4cond/grid.csv`](../experiments/eval-4cond/grid.csv):
every fine-tuned checkpoint scored on the full 6,761-frame held-out test split.

| | `results/` (sampled) | `experiments/eval-4cond/` (reported) |
|---|---|---|
| frames | 100 | 6,761 |
| split | stratified sample | frozen test split |
| models | COCO YOLOv8n, zero-shot | four fine-tuned checkpoints |
| baseline mAP@0.5 | 0.366 | 0.914 |

Do not compare a number from one column with a number from the other.

## Files

```
results/
├── method_comparison.csv    # sampled headline table: mAP, precision, recall, deltas, FP rate
├── false_positives.csv      # empty-frame FP rate per method
├── latency.csv              # per-method enhancement and detection timing
├── sample_manifest.csv      # composition of the 100-frame sample
├── snr_finetune/            # Ultralytics training output for the SNR-Aware run
└── yolo_finetune/           # Ultralytics training output for the raw run
```

## Caveats that apply to everything here

**Latency was measured on CPU.** `latency.csv` reports SNR-Aware at about 4 s
per image because it ran on CPU without a GPU. The figure used in the report is
about 122 ms per image on a V100. Treat the CSV as a relative comparison on one
machine, not as an absolute cost.

**The detection-time column is unreliable.** In the four-condition grid,
`yolo_ms` is empty and the GFLOPs column disagrees by roughly 2x for the same
detector across conditions. Attribution between enhancement and detection time
is not trustworthy yet, so no latency or FLOPs claim is made in the report.

**Day and night are not separated.** The box annotation file carries no
timestamps, so the harness treats every sampled frame as night and reports
`day_night_measured: false`. Per-stratum numbers in a `breakdown.json` inherit
that limitation unless a timestamped annotation file is supplied.
