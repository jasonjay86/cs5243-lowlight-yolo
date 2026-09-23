# cs5243-lowlight-yolo

CS5243 Computer Vision (Fall 2026) — Group 4 project.

**Question:** Which low-light image enhancement method (classical CLAHE vs. neural SNR-Aware) most improves YOLO wildlife detection accuracy on nocturnal camera-trap images?

**Team:**
- **Jason Johnson** — SNR-Aware enhancement, integration
- **Krutin Patel** — CLAHE (classical) enhancement
- **Rich Zanni** — YOLO detection pipeline, scripts

## Module ownership

```
cs5243-lowlight-yolo/
├── enhancements/
│   ├── base.py            # Enhancement interface — shared contract
│   ├── raw/               # No-op baseline (unassigned — see README)
│   ├── clahe/             # → Krutin Patel
│   └── snr_aware/         # → Jason Johnson
├── detection/             # → Rich Zanni (shared YOLO wrapper)
├── evaluation/            # unassigned — depends on detection contract
├── experiments/           # shared (one folder per run)
├── results/               # shared output
└── scripts/               # → Rich Zanni (convenience runners)
```

## Layout

```
cs5243-lowlight-yolo/
├── data/                  # Datasets — NOT in git (see data/README.md)
├── enhancements/
│   ├── base.py            # Enhancement interface — implement this FIRST
│   ├── raw/               # No-op baseline
│   ├── clahe/             # Classical contrast enhancement
│   └── snr_aware/         # Neural transformer enhancement
│       └── weights/       # Pretrained weights — gitignored
├── detection/             # Shared YOLO wrapper
├── evaluation/            # mAP + ΔmAP comparison harness
├── experiments/           # One folder per run (config + outputs)
├── results/               # Aggregated tables and figures
└── scripts/               # Convenience shell scripts
```

## Quick start

```bash
conda env create -f environment.yml
conda activate cs5243-lowlight-yolo
```

See `CONTRIBUTING.md` for the workflow (branches, PRs, what each person owns).

## Data

Datasets are not in the repo. See [`data/README.md`](data/README.md) for download links and expected directory layout.
