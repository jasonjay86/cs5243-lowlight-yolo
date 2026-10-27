"""
run_evaluation.py — CLI for the evaluation harness.

Owner: Jason Johnson

A thin wrapper over evaluation/ so the pipeline can be run from the repo
root. The metrics themselves live in evaluation/metrics.py (factored out of
scripts/evaluate_enhancements.py, Krutin Patel, PR #8).

Examples
--------
Full run, four methods, 500 frames, GPU:

    python scripts/run_evaluation.py \
        --images data/cct20/eccv_18_all_images_sm \
        --annotations data/cct20/caltech_bboxes_20200316.json \
        --image-labels data/cct20/caltech_camera_traps.json.zip \
        --methods raw clahe snr-aware retinexformer \
        --max-images 500 --device cuda

Re-score from cache (no inference, seconds instead of minutes):

    python scripts/run_evaluation.py --config experiments/<run>/config.yaml \
        --rescore

Regenerate the report only:

    python scripts/run_evaluation.py --config experiments/<run>/config.yaml \
        --rescore --no-write-results

The script writes its run directory as experiments/<run-id>/ with:
    config.yaml      full provenance of the numbers
    frames.json      the exact frame list and the sample composition
    predictions/     cached detections per method
    breakdown.json   per-stratum (day/night, empty/animal, species) numbers
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# OpenMP-on-Windows: must be set before torch is imported anywhere. See the
# note in evaluation/__init__.py. Set here too so the CLI is safe even if
# something else imports torch first.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.config import RunConfig, StrataConfig  # noqa: E402
from evaluation.harness import run_evaluation          # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--images", help="folder with the extracted CCT images")
    ap.add_argument("--annotations", help="CCT bbox JSON")
    ap.add_argument("--image-labels",
                    help="LILA image-level annotations (JSON or .json.zip); "
                         "the only source of real empty frames")
    ap.add_argument("--methods", nargs="+",
                    default=["raw", "clahe", "snr-aware"],
                    help="registered enhancement names; FIRST is the baseline")
    ap.add_argument("--max-images", type=int, default=500)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="threshold for precision/recall/FP rate")
    ap.add_argument("--conf-floor", type=float, default=0.001,
                    help="YOLO confidence floor for the PR curve")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0"])
    ap.add_argument("--yolo-weights", default="detection/yolov8n.pt")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--empty-fraction", type=float, default=0.25,
                    help="share of the sample that is animal-free frames")
    ap.add_argument("--day-fraction", type=float, default=0.25,
                    help="share of the ANIMAL frames that are daytime")
    ap.add_argument("--species-balance", default="natural",
                    choices=["natural", "uniform"],
                    help="'natural' keeps the dataset distribution; "
                         "'uniform' spreads across species")
    ap.add_argument("--min-per-species", type=int, default=1)

    ap.add_argument("--config", type=Path,
                    help="load a run config from YAML (for --rescore)")
    ap.add_argument("--rescore", action="store_true",
                    help="score from the cache; do not run inference")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore any existing cache and recompute")
    ap.add_argument("--no-write-results", action="store_true",
                    help="do not write the CSVs and figure under results/")
    ap.add_argument("--run-id", default="", help="override the run id")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    if args.config:
        cfg = RunConfig.from_yaml(args.config)
        # CLI overrides win over the stored config, for the flags that were
        # explicitly passed.
        if args.max_images != 500:
            cfg.max_images = args.max_images
        if args.methods != ["raw", "clahe", "snr-aware"]:
            cfg.methods = args.methods
        if args.device != "cpu":
            cfg.device = args.device
        if args.seed != 0:
            cfg.seed = args.seed
    else:
        if not args.images or not args.annotations:
            build_parser().error("--images and --annotations are required "
                                 "unless --config is given")
        strata = StrataConfig(
            empty_fraction=args.empty_fraction,
            day_fraction=args.day_fraction,
            species_balance=args.species_balance,
            min_per_species=args.min_per_species,
        )
        cfg = RunConfig(
            images_dir=args.images,
            annotations=args.annotations,
            image_labels=args.image_labels,
            methods=args.methods,
            max_images=args.max_images,
            imgsz=args.imgsz,
            conf=args.conf,
            conf_floor=args.conf_floor,
            device=args.device,
            yolo_weights=args.yolo_weights,
            bootstrap=args.bootstrap,
            seed=args.seed,
            strata=strata,
            run_id=args.run_id,
        )

    if args.rescore:
        # Point at the run directory that holds the cached predictions.
        if args.config:
            from evaluation.config import run_dir
            cfg.run_id = cfg.run_id or Path(args.config).parent.name
            experiments_root = Path(args.config).parent.parent
        else:
            experiments_root = REPO_ROOT / "experiments"

        # The frame list must match the cached run exactly.
        frames_json = experiments_root / cfg.run_id / "frames.json"
        if not frames_json.is_file():
            print(f"ERROR: no frames.json in "
                  f"{experiments_root / cfg.run_id}", file=sys.stderr)
            return 1

    out = run_evaluation(
        cfg,
        repo_root=REPO_ROOT,
        use_cache=not args.no_cache,
        write_results=not args.no_write_results,
        skip_predict=args.rescore,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
