# experiments

One directory per run. The rule is simple: **the numbers that matter live here,
the artefacts that regenerate do not.**

```
experiments/
├── eval-4cond/                  # the reported grid result
│   ├── config.yaml              # provenance: paths, weights, hashes, git SHA, versions
│   ├── grid.csv                 # 7 cells, every metric, with intervals
│   └── grid_breakdown.json      # night/day, animal/empty, per species
├── grid-raw-snr/                # earlier 3-cell grid, kept for history
│   ├── config.yaml
│   └── grid.csv
└── run-100f-s0-20261026-*/      # sampled 100-frame runs
    ├── config.yaml
    ├── breakdown.json
    └── frames.json              # the exact frame list and its fingerprint
```

Committed: `config.yaml`, `grid.csv`, and the small JSON breakdowns. They are a
few kilobytes each and they are the evidence.

Not committed, by `.gitignore`: `predictions/` (cached `.npz` detections),
`logs/`, `checkpoints/`, and every generated `*.png`. Figures are redrawn from
the CSV, so committing them would only create two things that can disagree.

## `eval-4cond` is the reported result

Everything else here is exploratory. `eval-4cond/config.yaml` records the git
revision, the checkpoints and their hashes, the manifest hashes, the resolved
configuration and the library versions, which is what makes the table in the
README reproducible rather than merely plausible.

## Naming

`run-<frames>f-s<seed>-<timestamp>` for sampled runs, `<purpose>` for anything
else (`eval-4cond`, `grid-raw-snr`). A directory name should tell you what
question it answers without opening it.
