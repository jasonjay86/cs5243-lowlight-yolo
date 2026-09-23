# Contributing

This is a 3-person semester project. The workflow is light but consistent.

## Branching

- `main` is **protected** — no direct commits, no force-pushes.
- Each enhancement module gets its own feature branch:
  - `feature/raw-baseline`
  - `feature/clahe`
  - `feature/snr-aware`  ← Jayson
- `feature/eval-harness` for shared evaluation/detection code.

## Workflow

1. Branch from `main`.
2. Commit small, focused changes with messages like `claude: implement LAB-CLAHE passthrough`.
3. Open a PR into `main` when your module is ready.
4. Jayson (repo owner) reviews + merges.

## Code style

- Python 3.10+ (conda env from `environment.yml`).
- Type hints on all new code.
- One Enhancement class per module under `enhancements/<method>/`.
- Every Enhancement must be `@register`-decorated so the eval harness can find it.

## The one rule that matters

**Implement `enhancements/base.py` and the first registered `Enhancement` BEFORE anyone
writes enhancement code.** Otherwise we'll spend the first week re-aligning interfaces
instead of doing science. The raw baseline (`enhancements/raw/`) is the cheapest way
to nail down the contract — write that first.

## Datasets and weights

- Datasets → under `data/` (gitignored). Don't commit.
- Pretrained weights → under `enhancements/<method>/weights/` (gitignored).
  Use Git LFS or external storage if a weight is small enough to share.
