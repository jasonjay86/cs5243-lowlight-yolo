# cs5243-lowlight-yolo

CS5243 Computer Vision (Fall 2026) — Group 4 project.

**Question:** Which low-light image enhancement method (classical CLAHE vs. neural SNR-Aware) most improves YOLO wildlife detection accuracy on nocturnal camera-trap images?

**Team:**
- **Jason Johnson** — SNR-Aware enhancement, integration
- **Krutin Patel** — CLAHE (classical) enhancement
- **Rich Zanni** — YOLO detection pipeline, scripts

## Layout & ownership

```
cs5243-lowlight-yolo/
├── data/                  # Datasets — NOT in git (see data/README.md)
├── enhancements/
│   ├── base.py            # READ FIRST — the Enhancement contract every module implements
│   ├── raw/               # No-op baseline (unassigned — do this first to validate the contract)
│   ├── clahe/             # → Krutin Patel (classical contrast enhancement)
│   │   └── weights/       # (n/a — CLAHE has no learned weights)
│   └── snr_aware/         # → Jason Johnson (neural transformer)
│       └── weights/       # Pretrained weights — gitignored
├── detection/             # → Rich Zanni (shared YOLO wrapper)
├── evaluation/            # unassigned — depends on detection contract being stable
├── experiments/           # shared (one folder per run: config + outputs)
├── results/               # shared (aggregated tables and figures)
└── scripts/               # → Rich Zanni (convenience runners)
```

## Order of operations

1. **Anyone:** implement `enhancements/raw/` to nail down the `Enhancement` contract end-to-end.
   ~30 min. Until this is done, Jason, Krutin, and Rich can't validate their modules.
2. **Jason:** branch `feature/snr-aware`, implement `enhancements/snr_aware/`.
3. **Krutin:** branch `feature/clahe`, implement `enhancements/clahe/`.
4. **Rich:** branch `feature/yolo-detector`, implement `detection/`.
5. **Whoever steps up:** branch `feature/eval-harness`, implement `evaluation/`.
6. **All three:** schedule a 30-min sync to merge, smoke-test the full pipeline, agree on a reference GPU for latency numbers.

## Quick start

```bash
conda env create -f environment.yml
conda activate cs5243-lowlight-yolo
```

See `CONTRIBUTING.md` for branches, commit-message convention (`<firstname>: ...`), and PR workflow.

## Data

Datasets are not in the repo. See [`data/README.md`](data/README.md) for download links and expected directory layout.


## CCT-20 YOLO Fine-Tuning

The project includes a location-disjoint CCT-20 split for fine-tuning
YOLO as a single-class animal detector.

### Prepare the dataset

```bash
python scripts/prepare_yolo_dataset.py \
  --manifest data/cct20_splits/split_manifest.csv \
  --annotations data/cct20/caltech_bboxes_20200316.json \
  --output data/cct20_yolo
```

The generated YOLO dataset uses one class:

```text
0 = animal
```

The preparation step produced:

- 32,029 training images / 30,419 boxes
- 7,441 validation images / 7,404 boxes
- 6,761 test images / 6,431 boxes
- 0 box-count mismatches

The generated `data/cct20_yolo` directory is not committed and can be
reconstructed from the split manifest and source annotations.

### Fine-tune YOLOv8n

Submit:

'scripts/sbatch-yolo-finetuning'

The current YOLOv8n experiment uses:

- image size: 640
- batch size: 64
- optimizer: AdamW
- initial learning rate: 0.001
- weight decay: 0.0005
- cosine learning-rate schedule
- early-stopping patience: 20
- seed: 42

Training stopped after 61 epochs. The best validation result occurred at
epoch 41.

Validation of the selected `best.pt` checkpoint:

- Precision: 0.922
- Recall: 0.786
- mAP50: 0.858
- mAP50-95: 0.557

The held-out test locations must not be used for model or threshold
selection.
