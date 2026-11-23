# cs5243-lowlight-yolo

CS5243 Computer Vision (Fall 2026), Group 4.

**Question:** does low-light image enhancement help YOLOv8n detect animals in
nocturnal camera-trap images, and does the answer change if the detector is
trained on the enhanced data rather than only tested on it?

| Name | Contribution |
|---|---|
| Jason Johnson | SNR-Aware enhancement, evaluation harness, reporting |
| Krutin Patel | CLAHE enhancement, project page |
| Richard Zanni | YOLO detection pipeline, training |

Project page: [`docs/index.html`](docs/index.html).

## Headline result

mAP@0.5 on the 6,761 held-out test frames. Rows are the data a detector was
trained on, columns are the images it was tested on. The top row is **Q1**
(test-time enhancement applied to a raw-trained detector); the diagonal is
**Q2** (each detector retrained on its own condition).

| trained on / tested on | raw | CLAHE | SNR-Aware | RetinexFormer |
|---|---|---|---|---|
| **raw** | **0.914** | 0.834 | 0.449 | 0.513 |
| **CLAHE** | | 0.901 | | |
| **SNR-Aware** | | | 0.743 | |
| **RetinexFormer** | | | | 0.777 |

No enhancement beat the raw baseline, and every paired 95% interval excludes
zero. Enhancing at test time hurts a raw-trained detector, mostly through lost
recall. Retraining on the enhanced domain recovers 84% (CLAHE), 66%
(RetinexFormer) and 63% (SNR-Aware) of that loss, and never all of it.

Full numbers, including precision, recall, empty-frame false positives and
per-stratum breakdowns: [`experiments/eval-4cond/grid.csv`](experiments/eval-4cond/grid.csv).

## Reproducing

### 1. Environment

```bash
conda env create -f environment.yml
conda activate cs5243-lowlight-yolo
```

On ARC V100 nodes use `bash scripts/setup_arc_env.sh` instead. The default
PyTorch wheel is a CUDA 13 build with no Volta kernels, so it fails at the first
kernel launch even though `torch.cuda.is_available()` returns True.

### 2. Data

Nothing under `data/` is tracked except the frozen split manifest. See
[`data/README.md`](data/README.md) for the expected layout and the download
sources.

### 3. The four-condition grid (main result)

The grid scores every trained checkpoint on the frozen `test` split, under
every test condition, on a GPU node:

```bash
sbatch --export=ALL,RUN_ID=eval-4cond,CELLS=minimal scripts/sbatch-eval-grid
```

`CELLS=minimal` scores 7 cells (baseline row plus diagonal), `CELLS=full`
scores all 16. The same run in the foreground:

```bash
python scripts/run_evaluation.py \
  --images data/cct20/eccv_18_all_images_sm \
  --annotations data/cct20/caltech_bboxes_20200316.json \
  --image-labels data/cct20/caltech_camera_traps.json.zip \
  --split-manifest data/cct20_splits/split_manifest.csv \
  --split test \
  --pixel-source precomputed \
  --weights raw=models/best_yolo-finetune-raw.pt \
  --weights clahe=models/best_yolo-finetune-clahe.pt \
  --weights snr-aware=models/best_yolo-finetune-snr.pt \
  --weights retinexformer=models/best_yolo-finetune-retinexformer.pt \
  --condition raw=data/cct20_splits/split_manifest.csv \
  --condition clahe=data/cct20_clahe/split_manifest.csv \
  --condition snr-aware=data/cct20_snr/split_manifest.csv \
  --condition retinexformer=data/cct20_retinex/split_manifest.csv \
  --device cuda
```

The first `--weights` and the first `--condition` must both be `raw`, because
every delta is measured against that cell. Each condition resolves its image
paths through its own manifest, so an enhanced set needs a `split_manifest.csv`
whose `image_path` column points at real files.

Checkpoints are not in git (`*.pt` is ignored). They are produced by the
fine-tuning jobs under `scripts/sbatch-yolo-finetuning*`.

### 4. Figures

Figures are drawn from the committed CSV, so no inference is needed to redraw
them and the picture cannot drift from the numbers:

```bash
python -m evaluation.grid_figure experiments/eval-4cond/grid.csv \
  --split test --figure both --slide
```

Generated PNGs are gitignored by design. Only `grid.csv`,
`grid_breakdown.json` and `config.yaml` are committed.

### 5. Single-method sampled runs

For a quick look at one or more enhancement methods on a stratified sample,
instead of the full grid:

```bash
python scripts/run_evaluation.py \
  --images data/cct20/eccv_18_all_images_sm \
  --annotations data/cct20/caltech_bboxes_20200316.json \
  --image-labels data/cct20/caltech_camera_traps.json.zip \
  --methods raw clahe snr-aware retinexformer \
  --max-images 100 --device cuda
```

Predictions are cached, so re-scoring with a different threshold takes seconds:

```bash
python scripts/run_evaluation.py --rescore --conf 0.5
```

These sampled runs write to `results/` and are exploratory. **The grid on the
frozen test split is the reported result.**

### 6. Tests

```bash
python tests/run_all_tests.py
```

Roughly three and a half minutes, no GPU, no dataset and no weights required.
It uses synthetic frames and a fake detector, so it runs anywhere.

## Repository layout

```
cs5243-lowlight-yolo/
├── data/                  # Datasets, not in git (except the split manifest)
├── enhancements/          # One module per method, all implementing base.py
│   ├── raw/               # No-op baseline
│   ├── clahe/             # Classical contrast enhancement
│   ├── snr_aware/         # CVPR 2022 transformer, pretrained weights
│   └── retinexformer/     # ICCV 2023 one-stage Retinex transformer
├── detection/             # Shared YOLO wrapper
├── evaluation/            # Harness: sampling, caching, metrics, figures
├── experiments/           # One folder per run: config plus numbers
├── results/               # Aggregated CSV summaries from sampled runs
├── scripts/               # CLIs, Slurm jobs, dataset preparation
├── tests/                 # Plain-script test suite
└── docs/                  # Project page (GitHub Pages)
```

## Dataset preparation

The YOLO dataset is generated from the frozen split, not committed:

```bash
python scripts/prepare_yolo_dataset.py \
  --manifest data/cct20_splits/split_manifest.csv \
  --annotations data/cct20/caltech_bboxes_20200316.json \
  --output data/cct20_yolo
```

One class, `0 = animal`. The split is location-disjoint, so test scores
reflect camera sites the detector never saw:

| split | images | boxes | empty | locations |
|---|---|---|---|---|
| train | 32,029 | 30,419 | 2,731 | 14 |
| val | 7,441 | 7,404 | 585 | 3 |
| test | 6,761 | 6,431 | 601 | 3 |

The generated `data/cct20_yolo` directory is not committed and can always be
reconstructed from the split manifest plus the source annotations.

## Fine-tuning

```bash
sbatch scripts/sbatch-yolo-finetuning          # raw
sbatch scripts/sbatch-yolo-finetuning-clahe    # CLAHE
sbatch scripts/sbatch-yolo-finetuning-snr      # SNR-Aware
```

Shared settings: image size 640, batch 64, AdamW, learning rate 1e-3, weight
decay 5e-4, cosine schedule, early-stopping patience 20, seed 42.

Validation metrics are **not** reported. Early stopping selects the checkpoint
on validation, so those numbers are optimistic by construction. Every reported
number comes from the held-out test split, and the test locations are never
used for model or threshold selection.

## Conventions

Branch, commit and review rules are in [`CONTRIBUTING.md`](CONTRIBUTING.md).
Commits are prefixed with a first name (`jason:`, `krutin:`, `rich:`) so
`git log --oneline | grep jason:` stays useful.
