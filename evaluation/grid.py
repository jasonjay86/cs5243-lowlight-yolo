"""
Fine-tuned grid evaluation.

Owner: Jason Johnson

Scores every (model trained on) x (tested on) cell on the held-out test
split, so Q1 (test-time enhancement) and Q2 (train-time enhancement) come out
of one invocation and one table. See evaluation/cells.py for the cell
semantics.

Why this is a separate path from the zero-shot run in evaluation/harness.py:

  * the zero-shot arm applies enhancement in process, from raw frames
  * the grid reads pixels that were already enhanced on disk, and each
    condition keeps them in a different directory
  * the grid varies the CHECKPOINT as well as the condition, which the
    zero-shot loop does not

Trying to force both through one loop produced a worse design. What IS shared
is everything that produces the numbers: build_stats and summarize, so the
metrics, the bootstrap intervals, the paired delta CI and the empty-frame
false-positive rate are the same code for both arms.
"""
from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

from evaluation.cache import PredictionCache
from evaluation.cells import cell_kind, cells_to_run, validate_cells
from evaluation.config import RunConfig, run_dir
from evaluation.dataset import (build_dataset, filter_by_split,
                                load_manifest_paths, load_split_map,
                                resolve_frame_path, scale_boxes_to_disk)
from evaluation.metrics import (breakdown, build_stats, per_species,
                                summarize)
from evaluation.predict import predictions_to_per_image, run_method

FIELDS = [
    "model", "condition", "kind", "n_frames",
    "map50", "map50_lo", "map50_hi",
    "d_map50", "d_lo", "d_hi",
    "map50_95", "precision", "recall",
    "empty_fp", "fp_lo", "fp_hi",
    # These four were computed on every cell and then dropped here, which is
    # why the latency and FLOPs slides had nothing to cite.
    "enh_ms", "yolo_ms", "gflops_mp", "n_detections",
]


def canonical_frames(cfg: RunConfig, repo_root: Path) -> list:
    """One frame list for the whole grid, with boxes scaled exactly once.

    Boxes are scaled against the RAW on-disk images. Every enhancement here
    is required to preserve image dimensions, so the annotation-to-pixel
    transform is identical for every condition and applying it per condition
    would be the same arithmetic repeated N times.
    """
    import cv2

    ds = build_dataset(Path(cfg.images_dir), Path(cfg.annotations),
                       Path(cfg.image_labels) if cfg.image_labels else None,
                       cfg.strata)
    split_map = load_split_map(Path(cfg.split_manifest))
    frames = filter_by_split(ds.frames, split_map, cfg.split)
    if not frames:
        raise SystemExit(
            f"ERROR: no frames matched split={cfg.split!r}. Check that "
            f"--images, --annotations and --split-manifest describe the "
            f"same image set.")

    rescaled, unreadable, shapes = 0, 0, {}
    for fr in frames:
        img = cv2.imread(fr.path, cv2.IMREAD_COLOR)
        if img is None:
            unreadable += 1
            continue
        shapes[fr.image_id] = img.shape
        before = fr.boxes.copy()
        scale_boxes_to_disk(fr, img.shape)
        if fr.boxes.size:
            rescaled += 1 if not _same(before, fr.boxes) else 0
    print(f"[grid] split={cfg.split}: {len(frames)} frames, "
          f"rescaled {rescaled}, unreadable {unreadable}")
    if unreadable:
        print(f"[grid] WARNING: {unreadable} baseline frame(s) could not be read")
    return frames


def _same(a, b) -> bool:
    import numpy as np
    return bool(np.allclose(a, b))


def condition_frames(canonical: list, name: str, manifest: Path,
                     repo_root: Path) -> list:
    """Clone the canonical frames with one condition's image paths.

    Paths come from the manifest's ``image_path`` column. A path that does
    not exist (a manifest written in someone else's ``/work`` tree) falls
    back to that condition's own directory by filename, so one person's
    manifest stays usable in another person's checkout.
    """
    paths = load_manifest_paths(Path(manifest))

    fallback = None
    for p in paths.values():
        resolved = resolve_frame_path(p, None, repo_root)
        if resolved is not None:
            fallback = Path(resolved).parent
            break
    if fallback is None:
        print(f"[grid] {name}: WARNING no manifest path resolved; "
              f"relying on filenames only")

    out, missing = [], 0
    for fr in canonical:
        p = resolve_frame_path(paths.get(fr.image_id, ""), fallback, repo_root)
        if p is None:
            missing += 1
            continue
        out.append(dataclasses.replace(fr, path=str(p)))
    if missing:
        print(f"[grid] {name}: WARNING {missing} frame(s) not found on disk")
    return out


def run_grid(cfg: RunConfig, repo_root: Path,
             experiments_root: Path | None = None) -> dict:
    """Evaluate every selected cell and write the comparison table."""
    repo_root = Path(repo_root)
    # Resolve repo-relative paths and fill in the run id, so the rest of this
    # function can assume absolute paths and a concrete run directory.
    cfg = cfg.resolved(repo_root)
    experiments_root = (Path(experiments_root) if experiments_root
                        else repo_root / "experiments")

    names = list(cfg.weights)
    cells = cells_to_run(names, cfg.cell_mode, conditions=list(cfg.conditions))
    validate_cells(cells, cfg.weights, cfg.conditions)

    # A condition may legitimately have test images and no model of its own
    # (an enhancement run only over the test split). A model with no matching
    # condition is different: it can only ever appear as a row in --cells full,
    # so warn rather than refuse.
    base_model, base_cond = names[0], list(cfg.conditions)[0]
    if base_model != base_cond:
        raise SystemExit(
            f"ERROR: the baseline model must be the baseline condition, or "
            f"every delta is measured against the wrong cell.\n"
            f"       first --weights:   {base_model}\n"
            f"       first --condition: {base_cond}")
    unmatched = [m for m in names if m not in cfg.conditions]
    if unmatched:
        print(f"[grid] NOTE model(s) with no matching condition: {unmatched}; "
              f"they appear only as rows against other conditions")

    rdir = run_dir(cfg, experiments_root)
    rdir.mkdir(parents=True, exist_ok=True)
    # Provenance: without this the CSV records numbers but not which
    # checkpoints or manifests produced them, which is exactly the gap that
    # makes a table unauditable three months later.
    cfg.to_yaml(rdir / "config.yaml")

    canonical = canonical_frames(cfg, repo_root)
    base_ids = [f.image_id for f in canonical]

    frames_by_condition: dict[str, list] = {}
    for name in names:
        frs = condition_frames(canonical, name, cfg.conditions[name], repo_root)
        got = [f.image_id for f in frs]
        # A paired delta is only meaningful when every cell is scored on the
        # same frames. All conditions use the frozen split, so this must
        # hold; if it does not, something upstream is wrong and the table
        # would be quietly incomparable.
        if got != base_ids:
            raise SystemExit(
                f"ERROR: condition {name!r} resolved {len(got)} frame(s) but "
                f"the baseline has {len(base_ids)}. Every cell must be "
                f"scored on the same frames.")
        frames_by_condition[name] = frs

    from evaluation.harness import _ground_truth_digest
    gt_digest = _ground_truth_digest(canonical)
    fingerprint = cfg.analysis_fingerprint(base_ids, gt_digest)
    print(f"[grid] fingerprint: {fingerprint}")
    cache = PredictionCache(rdir, fingerprint)

    from evaluation.predict import _load_model

    all_stats: dict[str, object] = {}
    models: dict[str, object] = {}
    for model_name, cond in cells:
        key = f"{model_name}__{cond}"
        frames = frames_by_condition[cond]
        preds = cache.load(key)
        if preds is None:
            if model_name not in models:
                models[model_name] = _load_model(cfg.weights[model_name],
                                                 cfg.device)
            print(f"[grid] {key}: scoring {len(frames)} frames")
            preds = run_method(cond, frames, models[model_name], cfg.imgsz,
                               cfg.conf_floor, repo_root, cfg.device,
                               cfg.weights[model_name],
                               pixel_source=cfg.pixel_source)
            preds.method = key
            cache.save(preds)
        else:
            print(f"[grid] {key}: cached")
        all_stats[key] = build_stats(frames, predictions_to_per_image(preds),
                                     preds.enh_ms, preds.yolo_ms, None)

    if not all_stats:
        raise SystemExit("[grid] no cell produced predictions.")

    base = names[0]
    results = summarize(all_stats, cfg.conf, cfg.bootstrap)

    rows = []
    for res in results:
        model_name, condition = res.method.split("__", 1)
        rows.append({
            "model": model_name,
            "condition": condition,
            "kind": cell_kind(model_name, condition, base),
            "n_frames": len(frames_by_condition[condition]),
            "map50": round(res.ap50, 4),
            "map50_lo": round(res.ap50_ci[0], 4),
            "map50_hi": round(res.ap50_ci[1], 4),
            "d_map50": round(res.d_ap50, 4),
            "d_lo": round(res.d_ci[0], 4),
            "d_hi": round(res.d_ci[1], 4),
            "map50_95": round(res.map5095, 4),
            "precision": round(res.precision, 4),
            "recall": round(res.recall, 4),
            "empty_fp": round(res.fp_rate, 4),
            "fp_lo": round(res.fp_ci[0], 4),
            "fp_hi": round(res.fp_ci[1], 4),
            "enh_ms": round(res.enh_ms, 3),
            "yolo_ms": round(res.yolo_ms, 3),
            "gflops_mp": (round(res.gflops_mp, 2)
                          if res.gflops_mp is not None else ""),
            "n_detections": res.n_detections,
        })

    csv_path = rdir / "grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Stratified views, per cell: night/day, animal/empty, and per species.
    # Nested, so a sidecar rather than more flat columns. Each one is
    # best-effort: a frame set without a usable night flag should not lose
    # the headline table.
    strat = {}
    for key, stats in all_stats.items():
        entry = {}
        for name, kwargs in (("by_light", {"by": "night"}),
                             ("by_presence", {"by": "empty"})):
            try:
                entry[name] = breakdown(stats, cfg.conf, **kwargs)
            except Exception as exc:  # noqa: BLE001
                entry[name] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            entry["by_species"] = per_species(stats, cfg.conf, min_frames=5)
        except Exception as exc:  # noqa: BLE001
            entry["by_species"] = {"error": f"{type(exc).__name__}: {exc}"}
        strat[key] = entry
    strat_path = rdir / "grid_breakdown.json"
    strat_path.write_text(json.dumps(strat, indent=2, default=str),
                          encoding="utf-8")

    _print_table(rows, cfg)
    print(f"[grid] wrote {csv_path}")

    # Drawn from the CSV, not from the in-memory rows, so the figure and the
    # committed numbers cannot drift apart. A plotting failure must not throw
    # away metrics that are already computed and cached.
    try:
        from evaluation.grid_figure import (draw_grid_figure,
                                           draw_summary_figure, load_rows)
        loaded = load_rows(csv_path)
        fig_path = draw_grid_figure(loaded, rdir / "grid.png",
                                    split=cfg.split)
        print(f"[grid] figure: {fig_path}")
        sum_path = draw_summary_figure(loaded, rdir / "grid-summary.png",
                                       split=cfg.split)
        print(f"[grid] summary: {sum_path}")
        # Slide-sized copies, so nothing has to be rescaled in the deck.
        draw_grid_figure(loaded, rdir / "grid-slide.png", split=cfg.split,
                         slide=True)
        draw_summary_figure(loaded, rdir / "grid-summary-slide.png",
                            split=cfg.split, slide=True)
        print(f"[grid] slide figures: {rdir / 'grid-slide.png'}, "
              f"{rdir / 'grid-summary-slide.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[grid] WARNING: figure failed ({type(exc).__name__}: {exc})")
        print("[grid] numbers and csv are unaffected; redraw with "
              f"--redraw {csv_path}")

    print(f"[grid] config: {rdir / 'config.yaml'}")
    return {"rows": rows, "csv": str(csv_path), "run_dir": str(rdir),
            "fingerprint": fingerprint}


def _print_table(rows: list[dict], cfg: RunConfig) -> None:
    print()
    print(f"{'model':<12} {'tested on':<12} {'Q':<6} {'mAP50':>7} "
          f"{'d':>7} {'95% CI on d':>17} {'mAP50-95':>8} {'emptyFP':>8}")
    for r in rows:
        ci = f"[{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]"
        print(f"{r['model']:<12} {r['condition']:<12} {r['kind']:<6} "
              f"{r['map50']:>7.3f} {r['d_map50']:>+7.3f} {ci:>17} "
              f"{r['map50_95']:>8.3f} {r['empty_fp']:>8.3f}")
    print()
    print(f"  confidence {cfg.conf}, bootstrap {cfg.bootstrap}, "
          f"cells={cfg.cell_mode}, split={cfg.split}")
    print("  delta is against the first cell; CI is the paired bootstrap "
          "interval over test frames.")
