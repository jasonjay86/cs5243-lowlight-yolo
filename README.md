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
   ~30 min. Until this is done, Krutin and Rich can't validate their modules.
2. **Krutin:** branch `feature/clahe`, implement `enhancements/clahe/`.
3. **Rich:** branch `feature/yolo-detector`, implement `detection/`.
4. **Whoever steps up:** branch `feature/eval-harness`, implement `evaluation/`.
5. **All three:** schedule a 30-min sync to merge, smoke-test the full pipeline, agree on a reference GPU for latency numbers.

## Quick start

```bash
conda env create -f environment.yml
conda activate cs5243-lowlight-yolo
```

See `CONTRIBUTING.md` for branches, commit-message convention (`<firstname>: ...`), and PR workflow.

## Data

Datasets are not in the repo. See [`data/README.md`](data/README.md) for download links and expected directory layout.
