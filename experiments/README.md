# experiments

One folder per experimental run. Convention:

```
experiments/
├── 2026-09-25_clahe_default/      # one run = one (method, hyperparams) combo
│   ├── config.yaml                  # what was run
│   ├── results.json                 # final metrics (small, in git)
│   ├── predictions/                 # per-image detections (gitignored)
│   └── logs/                        # training/eval logs (gitignored)
└── 2026-09-25_snr_aware_default/
    └── ...
```

This makes the final Δ mAP table reproducible: each row references a config
that fully specifies what was run.

Use `scripts/new_experiment.sh <name>` to scaffold a new run folder with the
required subdirectories.
