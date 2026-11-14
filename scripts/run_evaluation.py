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


def _flatten(groups: list[list[str]]) -> list[str]:
    """Flatten repeated nargs="+" flags into one list."""
    return [item for group in groups for item in group]


def _parse_pairs(items: list[str], flag: str) -> dict[str, str]:
    """Parse repeated ``NAME=PATH`` flags into an order-preserving dict.

    Order matters: the first name is the baseline every delta is measured
    against.
    """
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"ERROR: {flag} expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise SystemExit(f"ERROR: {flag} expects NAME=PATH, got {item!r}")
        if name in out:
            raise SystemExit(f"ERROR: {flag} given twice for {name!r}")
        out[name] = path
    return out


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
    ap.add_argument("--split-manifest",
                    help="frozen split_manifest.csv from create_cct_split.py; "
                         "required when --split is used")
    ap.add_argument("--split",
                    help="score EVERY frame in this split instead of a "
                         "stratified sample (e.g. 'test'). Held-out "
                         "evaluation of fine-tuned checkpoints uses this.")
    ap.add_argument("--pixel-source", default=None,
                    choices=["on_the_fly", "precomputed"],
                    help="on_the_fly: raw frames + enhancement here. "
                         "precomputed: frames already enhanced on disk "
                         "(point --images at the enhanced folder); the "
                         "enhancement is NOT re-applied")
    ap.add_argument("--weights", action="append", nargs="+", default=[],
                    metavar="NAME=CHECKPOINT",
                    help="fine-tuned checkpoint per condition; repeat, or pass "
                         "several after one flag; the FIRST is the baseline. "
                         "Implies grid mode.")
    ap.add_argument("--condition", action="append", nargs="+", default=[],
                    metavar="NAME=MANIFEST",
                    help="that condition's split_manifest.csv; repeat, or pass "
                         "several after one flag; must name the same "
                         "conditions as --weights, same order")
    ap.add_argument("--cells", default=None, choices=["minimal", "full"],
                    help="minimal: baseline row + diagonal (2N-1 cells). "
                         "full: every model x condition cell (N^2)")
    ap.add_argument("--methods", nargs="+",
                    default=["raw", "clahe", "snr-aware"],
                    help="registered enhancement names; FIRST is the baseline")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=None,
                    help="threshold for precision/recall/FP rate")
    ap.add_argument("--conf-floor", type=float, default=None,
                    help="YOLO confidence floor for the PR curve")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "cuda:0"])
    ap.add_argument("--yolo-weights", default="detection/yolov8n.pt")
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)

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
    ap.add_argument("--redraw", metavar="CSV", default=None,
                    help="redraw the grid figure from an existing grid.csv "
                         "and exit; needs no GPU and runs no inference")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path for --redraw (default: grid.png "
                         "beside the csv)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore any existing cache and recompute")
    ap.add_argument("--no-write-results", action="store_true",
                    help="do not write the CSVs and figure under results/")
    ap.add_argument("--run-id", default="", help="override the run id")
    ap.add_argument("--experiments-root", default=None,
                    help="where run directories live "
                         "(default: <repo>/experiments)")
    return ap


def _find_latest_run(experiments_root: Path) -> Path | None:
    """The most recently modified run directory that has a config.yaml.

    Run ids carry a timestamp, so the exact directory name is not
    predictable. This lets --rescore work without being told which run to
    rescore.

    Sorted by modification time rather than by name: a name sort would
    order "run-100f-s0-..." before "run-50f-s1-...", because the frame
    count comes first in the name, so the newest run is not the
    lexically-last one.
    """
    if not experiments_root.is_dir():
        return None
    candidates = [d for d in experiments_root.iterdir()
                  if d.is_dir() and (d / "config.yaml").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / "config.yaml").stat().st_mtime)


def _resolve_rescore_run(args, cfg=None) -> tuple[Path, Path]:
    """Work out which run directory to rescore, and where experiments live.

    Returns (run_dir, experiments_root).

    Three ways to point at a run:
      1. --run-id <name>              explicit directory name
      2. --config experiments/<run>/config.yaml   a config you point at
      3. neither                       the most recent run

    --experiments-root overrides where runs are looked for, which is what
    makes it possible to rescore a run that lives outside this checkout.
    """
    experiments_root = (Path(args.experiments_root)
                        if args.experiments_root
                        else REPO_ROOT / "experiments")

    if args.run_id:
        run = experiments_root / args.run_id
        if not (run / "config.yaml").is_file():
            print(f"ERROR: no run called '{args.run_id}' in {experiments_root}",
                  file=sys.stderr)
            raise SystemExit(1)
        return run, experiments_root

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            print(f"ERROR: config not found: {config_path}", file=sys.stderr)
            raise SystemExit(1)
        # The run directory is the config's parent. The experiments root
        # is its grandparent, unless the config was given as an absolute
        # path outside the repo.
        run = config_path.parent
        if run.parent.name == "experiments":
            experiments_root = run.parent
        elif args.experiments_root:
            experiments_root = Path(args.experiments_root)
        else:
            experiments_root = REPO_ROOT / "experiments"
        if cfg is not None:
            cfg.run_id = cfg.run_id or run.name
        return run, experiments_root

    # No explicit target: use the most recent run.
    latest = _find_latest_run(experiments_root)
    if latest is None:
        print(f"ERROR: no runs found in {experiments_root}.", file=sys.stderr)
        print("       Run a full evaluation first, or pass --run-id.",
              file=sys.stderr)
        raise SystemExit(1)
    return latest, experiments_root


def main() -> int:
    args = build_parser().parse_args()

    # Redraw short-circuits everything: it reads a CSV and draws. No data, no
    # GPU, no inference, so it works on a login node.
    if args.redraw:
        from evaluation.grid_figure import draw_grid_figure, load_rows
        csv_path = Path(args.redraw)
        out = args.out or csv_path.with_suffix(".png")
        rows = load_rows(csv_path)
        written = draw_grid_figure(rows, out, split=args.split)
        print(f"[figure] wrote {written}  ({len(rows)} cells)")
        return 0

    # A rescore needs to know WHICH run to rescore before it can build a
    # config. Resolve that first: --run-id, then --config, then the most
    # recent run.
    run_dir_override = None
    experiments_root = None
    if args.rescore:
        run_dir_override, experiments_root = _resolve_rescore_run(args, None)
        cfg = RunConfig.from_yaml(run_dir_override / "config.yaml")
        cfg.run_id = run_dir_override.name

        # The frame list must match the cached run exactly.
        frames_json = run_dir_override / "frames.json"
        if not frames_json.is_file():
            print(f"ERROR: no frames.json in {run_dir_override}", file=sys.stderr)
            print("       That run has no cached predictions to rescore.",
                  file=sys.stderr)
            return 1
        print(f"[rescore] using run: {run_dir_override}")

        # CLI overrides still win over the stored config.
        if args.conf is not None:
            cfg.conf = args.conf
        if args.bootstrap is not None:
            cfg.bootstrap = args.bootstrap
        if args.imgsz is not None:
            cfg.imgsz = args.imgsz
        if args.conf_floor is not None:
            cfg.conf_floor = args.conf_floor
        if args.device is not None:
            cfg.device = args.device
    elif args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            print(f"ERROR: config not found: {config_path}", file=sys.stderr)
            return 1
        cfg = RunConfig.from_yaml(config_path)
        # CLI overrides win over the stored config, for the flags that were
        # explicitly passed.
        if args.max_images is not None:
            cfg.max_images = args.max_images
        if args.methods != ["raw", "clahe", "snr-aware"]:
            cfg.methods = args.methods
        if args.device is not None:
            cfg.device = args.device
        if args.seed is not None:
            cfg.seed = args.seed
        if args.conf is not None:
            cfg.conf = args.conf
        if args.bootstrap is not None:
            cfg.bootstrap = args.bootstrap
        if args.imgsz is not None:
            cfg.imgsz = args.imgsz
        if args.conf_floor is not None:
            cfg.conf_floor = args.conf_floor
        if args.split_manifest is not None:
            cfg.split_manifest = args.split_manifest
        if args.split is not None:
            cfg.split = args.split
        if args.pixel_source is not None:
            cfg.pixel_source = args.pixel_source
    else:
        if not args.images or not args.annotations:
            build_parser().error("--images and --annotations are required "
                                 "unless --config or --rescore is given")
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
            split_manifest=args.split_manifest or "",
            split=args.split or "",
            methods=args.methods,
            max_images=args.max_images if args.max_images is not None else 500,
            pixel_source=args.pixel_source or "on_the_fly",
            imgsz=args.imgsz if args.imgsz is not None else 640,
            conf=args.conf if args.conf is not None else 0.25,
            conf_floor=args.conf_floor if args.conf_floor is not None else 0.001,
            device=args.device if args.device is not None else "cpu",
            yolo_weights=args.yolo_weights,
            bootstrap=args.bootstrap if args.bootstrap is not None else 1000,
            seed=args.seed if args.seed is not None else 0,
            strata=strata,
            run_id=args.run_id,
        )

        # Grid mode: one checkpoint and one manifest per condition.
        if args.weights:
            cfg.weights = _parse_pairs(_flatten(args.weights), "--weights")
            cfg.conditions = _parse_pairs(_flatten(args.condition), "--condition")
            if not cfg.conditions:
                raise SystemExit(
                    "ERROR: grid mode needs at least one --condition.")
            # Conditions and models need not be the same set: a condition can
            # have test images without a fine-tuned model of its own. Only the
            # baseline must agree, because every delta is measured against it.
            if list(cfg.weights)[0] != list(cfg.conditions)[0]:
                raise SystemExit(
                    "ERROR: the first --weights and the first --condition must "
                    "be the same name (the baseline cell).\n"
                    f"  first weight:    {list(cfg.weights)[0]}\n"
                    f"  first condition: {list(cfg.conditions)[0]}")
            if not cfg.split:
                raise SystemExit(
                    "ERROR: grid mode needs --split (usually 'test') so every "
                    "cell is scored on the held-out frames.")
        if args.cells:
            cfg.cell_mode = args.cells

    if cfg.weights:
        from evaluation.grid import run_grid
        run_grid(cfg, REPO_ROOT, experiments_root=experiments_root)
        return 0

    out = run_evaluation(
        cfg,
        repo_root=REPO_ROOT,
        experiments_root=experiments_root,
        use_cache=not args.no_cache,
        write_results=not args.no_write_results,
        skip_predict=args.rescore,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
