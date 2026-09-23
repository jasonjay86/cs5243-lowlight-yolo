# scripts

Convenience shell scripts. Examples:

- `new_experiment.sh <name>` — scaffold a new run folder under `experiments/`.
- `run_all_methods.sh` — run YOLO detection on every registered enhancement.
- `aggregate_results.py` — collect per-run `results.json` into `results/method_comparison.csv`.

Each script should be **idempotent and self-documenting** — `bash scripts/foo.sh -h`
should print usage.
