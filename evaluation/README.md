# evaluation

**Owner:** Jason Johnson

Compares detection results across enhancement methods and reports:

- **Δ mAP** (enhanced − raw baseline) per method — the primary result.
- **False-positive rate** on "empty" frames per method (for hallucinated detections).
- **Latency** for enhancement stage vs. YOLO stage, attributed separately.

Plus per-stratum breakdowns (night vs day, empty vs animal, per species) and
confidence intervals on every headline number.

Replaces the placeholder in `evaluation/`. The metric implementation is
factored out of `scripts/evaluate_enhancements.py` (Krutin Patel, PR #8),
which still works unchanged.

## Running the pipeline

```bash
python scripts/run_evaluation.py \
    --images data/cct20/eccv_18_all_images_sm \
    --annotations data/cct20/caltech_bboxes_20200316.json \
    --image-labels data/cct20/caltech_camera_traps.json.zip \
    --methods raw clahe snr-aware retinexformer \
    --max-images 100 --device cuda
```

Run it from the repo root. The paths above are relative to it.

**`--methods`** takes any registered enhancement name. The **first one is the
baseline**, so put `raw` first. Available: `raw`, `clahe`, `snr-aware`,
`retinexformer`.

**`--image-labels`** is the LILA image-level annotation file. Pass it. It is
the only source of real empty frames, and without it the empty-frame
false-positive rate is `n/a`.

**`--device`** takes `cpu`, `cuda` or `cuda:0`. Use `cuda` if the local
torch has a CUDA build:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**`--max-images`** sets the sample size. 100 is a good first pass, 500 for
the final numbers. On CPU, budget roughly 4 s/image for `snr-aware` and
8 s/image for `retinexformer`.

## Re-scoring without re-running inference

Predictions are cached per method, so changing a threshold or adding a
metric is a seconds-long operation rather than a full re-run:

```bash
python scripts/run_evaluation.py --config experiments/<run>/config.yaml --rescore
```

The cache is invalidated automatically when anything that could change the
numbers changes: the frame list, the ground-truth boxes, the annotation
file, the thresholds, the detector weights, the library versions, or the
device.

### Other options

| Flag | Default | Purpose |
|---|---|---|
| `--imgsz` | 640 | detector input size |
| `--conf` | 0.25 | threshold for precision / recall / FP rate |
| `--conf-floor` | 0.001 | YOLO confidence floor for the PR curve |
| `--yolo-weights` | `detection/yolov8n.pt` | detector weights |
| `--bootstrap` | 1000 | bootstrap resamples for the confidence intervals |
| `--seed` | 0 | sample selection seed |
| `--empty-fraction` | 0.25 | share of the sample that is animal-free frames |
| `--day-fraction` | 0.25 | share of the animal frames that are daytime |
| `--species-balance` | `natural` | `natural` keeps the dataset distribution, `uniform` spreads across species |
| `--run-id` | auto | name the run directory |
| `--no-cache` | off | ignore any existing cache and recompute |
| `--no-write-results` | off | write the run artefacts but not the `results/` summaries |

### Smoke testing the pipeline

```bash
python tests/test_evaluation.py        # metric arithmetic, 29 tests
python tests/test_pipeline_smoke.py    # full pipeline with a fake detector
```

The smoke test runs the whole path — config, selection, prediction, caching,
scoring, reporting — against a synthetic dataset and a fake detector, so it
finishes in seconds and needs no weights or GPU.

## What you get

Per run, under `experiments/<run-id>/`:

| File | Contents |
|---|---|
| `config.yaml` | full provenance: paths, thresholds, seed, strata, git SHA, library versions, device |
| `frames.json` | the exact frame list, the sample composition, the fingerprint |
| `predictions/<method>.npz` | cached detections: boxes, confs, classes, frame indices |
| `predictions/<method>.json` | the fingerprint that produced the cache |
| `breakdown.json` | per-stratum numbers: night/day, empty/animal, per species |

Committed summaries, under `results/`:

| File | Contents |
|---|---|
| `method_comparison.csv` | the headline table: mAP, precision, recall, Δ mAP, FP rate, latency, each with a 95% CI |
| `false_positives.csv` | empty-frame FP rate per method |
| `latency.csv` | per-method, per-stage timing and the enhancement's share of total time |
| `sample_manifest.csv` | how the frame sample is composed |
| `figures/comparison.png` | results table, PR curves, Δ mAP bars with CIs |

## Reading the results

**The sample is stratified, not random.** Empty frames are included on
purpose, because without them the false-positive metric is undefined. Night
and day are separated so a daylight frame acts as a control on the
enhancement itself. Frames containing a car are skipped.

**YOLOv8n is a COCO model.** It has no bobcat, opossum, raccoon or skunk
class, so a per-class score would be meaningless. Every ground-truth box is
treated as one "animal" class, and a prediction counts as a true positive if
it overlaps the right place, whatever its COCO label says. Four CCT species
do have COCO counterparts (bird, dog, cat, cow); on every other frame the
label is ignored. Say this in any write-up that quotes the numbers, because
"mAP 0.366" otherwise reads as "36% of animals correctly classified" when
it means "36% correctly localised".

**A note on the CCT-20 coordinates.** The benchmark images are downsized to
a maximum of 1024 px per side while the bbox JSON carries original-resolution
coordinates. The harness rescales the ground truth to match. If you ever
see recall collapse to exactly zero while the detector still fires, that
rescale is the thing to check.

## Windows

PyTorch and Ultralytics both link against an OpenMP runtime, and on a stock
conda+pip install the second one to initialise refuses to load:

```
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
```

The harness sets `KMP_DUPLICATE_LIB_OK=TRUE` itself, before torch is
imported anywhere, so the pipeline runs without it. If you hit this in your
own script, set it before importing torch:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

## Layout

```
evaluation/
├── __init__.py     public API: RunConfig, StrataConfig, run_evaluation
├── config.py       RunConfig, YAML round-trip, run fingerprint, env capture
├── dataset.py      CCT loading, stratified frame selection, ground truth
├── cache.py        per-method prediction cache (.npz) with invalidation
├── predict.py      run enhancements + YOLO, write detections
├── metrics.py      AP, IoU matching, PR curve, bootstrap CI, breakdowns
├── report.py       CSV writers + the comparison figure
└── harness.py      orchestration: select -> predict -> cache -> score -> report
```

`scripts/run_evaluation.py` is the CLI. `scripts/evaluate_enhancements.py`
(Krutin's original, PR #8) still works unchanged; this package factors its
metric implementation into `evaluation/metrics.py` and adds the caching,
stratification and manifest layers around it.
