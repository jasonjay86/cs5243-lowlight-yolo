# scripts

Command-line entry points, dataset preparation and Slurm jobs. Run everything
from the repository root.

## Evaluation (the main result)

| Script | Purpose |
|---|---|
| `run_evaluation.py` | the harness CLI: sampled runs, grid mode, `--rescore`, `--redraw` |
| `sbatch-eval-grid` | Slurm job that scores every checkpoint on every test condition |
| `evaluate_enhancements.py` | earlier standalone evaluator, kept as a cross-check |
| `plot_finetune_eval.py` | independent re-plot of the fine-tuned results, second opinion |
| `plot_clahe_finetune.py` | training curves for the CLAHE run |

The grid:

```bash
sbatch --export=ALL,RUN_ID=eval-4cond,CELLS=minimal scripts/sbatch-eval-grid
```

`CELLS=full` scores all 16 cells instead of the 7 in the minimal set.

## Data preparation

| Script | Purpose |
|---|---|
| `create_cct_split.py` | builds the frozen location-disjoint split (already committed, do not re-run) |
| `prepare_yolo_dataset.py` | manifest plus annotations to a single-class YOLO dataset |
| `enhance_dataset.py` | applies one enhancement to a condition set and writes its manifest |

## Training

| Script | Purpose |
|---|---|
| `sbatch-yolo-finetuning` | YOLOv8n on raw images |
| `sbatch-yolo-finetuning-clahe` | YOLOv8n on CLAHE images |
| `sbatch-yolo-finetuning-snr` | YOLOv8n on SNR-Aware images |
| `run_clahe_yolo.py` | foreground CLAHE fine-tune, for debugging the job |
| `run_yolo_compare.py` | side-by-side detections for a handful of frames |

## Environment

| Script | Purpose |
|---|---|
| `setup_arc_env.sh` | creates the conda env with a torch build that has Volta kernels |

## Module checks

Small scripts used while integrating each method. They print, they do not
assert, and they are not part of the test suite.

| Script | Checks |
|---|---|
| `test_clahe.py` | CLAHE output shape, dtype and brightness on a sample frame |
| `test_retinexformer.py` | RetinexFormer forward pass and weight loading |
| `test_yolo.py` | detector wrapper on a sample frame |
| `smoke_snr_aware.py` | SNR-Aware end to end, writes before/after comparisons |
| `verify_snr_mask_fix.py` | confirms the SNR mask is now required rather than an all-ones placeholder |

The automated suite lives in `tests/`, not here:

```bash
python tests/run_all_tests.py
```
