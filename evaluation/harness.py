"""
Evaluation harness orchestration.

Owner: Jason Johnson

The pipeline is:

    select frames (stratified)
        -> run each method through the detector (cached)
            -> score from the cache (metrics)
                -> write CSVs and figures (report)

The cache sits in the middle on purpose. The selection and the metrics are
cheap and get iterated on a lot; the detection is expensive and does not.
Splitting them means a change to the AP implementation or a new threshold
costs seconds instead of a full re-run.

Every run writes a config.yaml with the full provenance of the numbers:
the frame list, the annotation file hashes, the YOLO weights, the library
versions, the device, and the git revision.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evaluation.cache import PredictionCache
from evaluation.config import RunConfig, env_fingerprint, git_sha, run_dir
from evaluation.dataset import (
    build_dataset,
    scale_boxes_to_disk,
    select_frames,
    summarise_sample,
)
from evaluation.metrics import build_stats, breakdown, per_species, summarize
from evaluation.predict import predictions_to_per_image, run_method
from evaluation.report import (
    draw_figure,
    print_summary,
    write_false_positives,
    write_latency,
    write_method_comparison,
    write_sample_manifest,
)


def _relative_weights_path(weights_path: str, repo_root) -> str:
    """The weights path as it should appear in a figure caption.

    An absolute path leaks the local username and the machine's directory
    layout into an artefact that gets committed and shared. If the weights
    live under the repo root, print the repo-relative path. Otherwise fall
    back to the file name alone.
    """
    try:
        return str(Path(weights_path).resolve().relative_to(
            Path(repo_root).resolve()))
    except (ValueError, OSError):
        return Path(weights_path).name


def _figure_title(n_day: int, n_total: int, n_night: int | None = None) -> str:
    """The figure title, which must not misdescribe the sample.

    The title names the corpus, not the lighting, unless the lighting is
    actually known.

    The CCT-20 box annotation file carries no timestamps, so the day/night
    split cannot be measured from it: every frame falls back to "night".
    Asserting "night frames" on that basis claims a measurement the data
    does not support, and it is wrong the moment someone supplies real
    timestamps and finds a mixed sample.

    So the title only mentions lighting when the sample genuinely is all
    one kind AND the classification was real rather than defaulted. The
    caller passes `n_night` so the mixed case can be spelled out.
    """
    base = "Raw vs enhanced + YOLO on CCT-20"
    if n_total and n_day == 0 and n_night is not None and n_night == n_total:
        return base + " night frames"
    if n_total and n_day == n_total:
        return base + " day frames"
    return base


def _ground_truth_digest(frames) -> str:
    """Hash of the resolved ground-truth boxes, for the cache fingerprint."""
    import hashlib
    h = hashlib.sha256()
    for fr in frames:
        h.update(fr.path.encode("utf-8"))
        h.update(b"\n")
        if fr.boxes.size:
            # Round so float noise does not invalidate the cache.
            for row in np.round(np.asarray(fr.boxes, dtype=float), 2).reshape(-1, 4):
                h.update(",".join(f"{v:.2f}" for v in row).encode("utf-8"))
                h.update(b";")
    return h.hexdigest()[:16]


def run_evaluation(cfg: RunConfig,
                   repo_root: Path,
                   experiments_root: Path | None = None,
                   results_root: Path | None = None,
                   use_cache: bool = True,
                   write_results: bool = True,
                   skip_predict: bool = False) -> dict:
    """Run one evaluation end to end and return the scored results."""
    repo_root = Path(repo_root)
    cfg = cfg.resolved(repo_root)
    experiments_root = Path(experiments_root or (repo_root / "experiments"))
    results_root = Path(results_root or (repo_root / "results"))

    rdir = run_dir(cfg, experiments_root)
    rdir.mkdir(parents=True, exist_ok=True)

    # ---- 1. select frames -------------------------------------------------
    print(f"[harness] loading annotations from {cfg.annotations}")
    dataset = build_dataset(
        Path(cfg.images_dir), Path(cfg.annotations),
        Path(cfg.image_labels) if cfg.image_labels else None,
        cfg.strata,
    )
    print(f"[harness] {len(dataset.frames)} usable frames "
          f"({dataset.n_skipped_car} skipped for containing a car, "
          f"{dataset.n_empty_from_labels} empty frames from the image-level file)")

    frames = select_frames(dataset, cfg.max_images, cfg.strata, cfg.seed)

    # On a rescore, the cached predictions belong to a SPECIFIC frame list.
    # Re-selecting the sample here could produce a different list (a
    # changed seed, a changed stratification, a changed dataset), and the
    # cached boxes would then be scored against the wrong frames. So the
    # stored frame list is authoritative and the selection is replaced by it.
    if skip_predict:
        stored = rdir / "frames.json"
        if not stored.is_file():
            raise SystemExit(
                f"ERROR: {stored} not found. A rescore needs the frame list "
                f"from the original run."
            )
        _stored = json.loads(stored.read_text(encoding="utf-8"))
        wanted = list(_stored.get("frames", []))
        if not wanted:
            raise SystemExit(f"ERROR: {stored} has no frame list.")
        by_path = {f.path: f for f in dataset.frames}
        missing = [p for p in wanted if p not in by_path]
        if missing:
            raise SystemExit(
                f"ERROR: {len(missing)} frame(s) from the stored run are not "
                f"in the dataset now, e.g. {missing[0]}. The dataset or the "
                f"annotations changed since that run, so its cached "
                f"predictions can no longer be scored."
            )
        frames = [by_path[p] for p in wanted]
        print(f"[harness] rescore: using the {len(frames)} frames from "
              f"{stored.name}")

    # Scale the ground-truth boxes into the on-disk pixel space. The CCT
    # benchmark images are downsized to a maximum of 1024 px per side while
    # the bbox JSON carries original-resolution coordinates. Without this,
    # every box lands in the wrong place and recall is exactly zero even
    # though the detector is firing normally.
    import cv2
    n_scaled, n_unreadable = 0, 0
    for fr in frames:
        img = cv2.imread(fr.path, cv2.IMREAD_COLOR)
        if img is None:
            n_unreadable += 1
            continue
        before = fr.boxes.copy()
        scale_boxes_to_disk(fr, img.shape)
        if fr.boxes.size and not np.allclose(before, fr.boxes):
            n_scaled += 1
    if n_scaled:
        print(f"[harness] rescaled ground-truth boxes for {n_scaled} frame(s) "
              f"from annotation space to on-disk pixel space")
    if n_unreadable:
        print(f"[harness] WARNING: {n_unreadable} frame(s) could not be read; "
              f"their ground-truth boxes were left unscaled")

    # Sanity check: the boxes must sit inside the image. Boxes that lie
    # outside the frame are the signature of a coordinate-space mismatch,
    # and they silently destroy recall.
    oob = 0
    for fr in frames:
        if not fr.boxes.size:
            continue
        img = cv2.imread(fr.path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        b = fr.boxes
        if (b[:, 0] >= w).any() or (b[:, 1] >= h).any() or \
           (b[:, 2] <= 0).any() or (b[:, 3] <= 0).any():
            oob += 1
    if oob:
        print(f"[harness] WARNING: {oob} frame(s) have ground-truth boxes "
              f"outside the image bounds. The annotation and the image "
              f"dimensions disagree, so the boxes are probably in a "
              f"different coordinate space. Recall will be ~0 until that "
              f"is resolved.")

    sample = summarise_sample(frames)
    print(f"[harness] selected {sample['n_frames']} frames: "
          f"{sample['n_empty']} empty, {sample['n_night']} night, "
          f"{sample['n_day']} day, {sample['n_boxes']} ground-truth boxes")

    frame_ids = [f.path for f in frames]
    gt_digest = _ground_truth_digest(frames)
    fingerprint = cfg.analysis_fingerprint(frame_ids, gt_digest)
    cfg.run_id = cfg.run_id or f"run-{cfg.max_images}f-s{cfg.seed}"
    cfg.to_yaml(rdir / "config.yaml")
    (rdir / "frames.json").write_text(
        json.dumps({"fingerprint": fingerprint, "frames": frame_ids,
                    "sample": sample}, indent=2), encoding="utf-8")
    print(f"[harness] run dir: {rdir}")
    print(f"[harness] fingerprint: {fingerprint}")

    # ---- 2. predict (cached) ---------------------------------------------
    # Caches written before the fingerprint scheme was split into
    # "inference inputs" and "scoring inputs" are still valid: the
    # detections did not change, only the hash that guards them. Accept
    # the old scheme so a cache is not thrown away over a bookkeeping
    # change.
    legacy = []
    try:
        legacy.append(cfg.legacy_analysis_fingerprint(frame_ids, gt_digest))
    except Exception:
        pass

    cache = PredictionCache(rdir, fingerprint, legacy_fingerprints=legacy)
    model = None
    if not skip_predict:
        for method in cfg.methods:
            if use_cache and cache.has(method):
                print(f"[harness] {method}: using cached predictions")
                continue
            if model is None:
                from evaluation.predict import _load_model
                model = _load_model(cfg.yolo_weights, cfg.device)
            print(f"[harness] {method}: running on {len(frames)} frames")
            preds = run_method(method, frames, model, cfg.imgsz, cfg.conf_floor,
                               repo_root, cfg.device, cfg.yolo_weights)
            cache.save(preds)
            print(f"[harness] {method}: {preds.n_detections} detections, "
                  f"{preds.enh_ms:.1f} ms/img enhance, {preds.yolo_ms:.1f} ms/img YOLO")

    # ---- 3. score ---------------------------------------------------------
    all_stats = {}
    for method in cfg.methods:
        preds = cache.load(method)
        if preds is None:
            why = cache.why_stale(method)
            print(f"[harness] SKIPPING {method}: no cached predictions"
                  + (f" ({why})" if why else ""))
            continue
        per_image = predictions_to_per_image(preds)
        gflops = None
        try:
            from evaluation.predict import _get_enhancement
            enh = _get_enhancement(method, repo_root)
            gflops = enh.flops_per_megapixel()
        except Exception:
            pass
        all_stats[method] = build_stats(frames, per_image, preds.enh_ms,
                                         preds.yolo_ms, gflops)

    if not all_stats:
        raise SystemExit("[harness] no method had usable predictions.")

    results = summarize(all_stats, cfg.conf, cfg.bootstrap)

    # ---- 4. report --------------------------------------------------------
    n_empty = sample["n_empty"]
    n_boxes = sample["n_boxes"]
    dev = "GPU" if env_fingerprint().get("cuda_available") else "CPU"

    # Print the weights path relative to the repo root. The absolute path
    # leaks the local username and the machine's directory layout into a
    # figure that gets committed and shared.
    weights_path = _relative_weights_path(cfg.yolo_weights, repo_root)

    caption = (f"{len(frames)} frames ({len(frames) - n_empty} with animals, "
               f"{n_empty} empty), {n_boxes} ground-truth boxes. "
               f"Same YOLO ({weights_path}), imgsz {cfg.imgsz}, "
               f"P/R/FP at conf {cfg.conf}, IoU 0.5. "
               f"Latency measured on {dev}. CIs: paired bootstrap over frames "
               f"({cfg.bootstrap} resamples).")

    # The day/night split is only meaningful if the timestamps are real.
    # The CCT-20 box annotations carry none, so the split is a default.
    if not sample.get("day_night_measured", False):
        caption += (" Day/night was not measured: the annotation file has no "
                    "timestamps, so the split is a default and is not "
                    "reported here.")

    # The title must not misdescribe the sample. The CCT-20 box
    # annotations carry no timestamps, so the day/night counts come from a
    # default rather than a measurement; only claim lighting when the
    # sample really is one kind AND the classification was real.
    title = _figure_title(sample.get("n_day", 0), len(frames),
                          sample.get("n_night")
                          if sample.get("day_night_measured", False)
                          else None)

    if write_results:
        write_method_comparison(results, results_root / "method_comparison.csv")
        write_false_positives(results, results_root / "false_positives.csv")
        write_latency(results, results_root / "latency.csv")
        write_sample_manifest(sample, results_root / "sample_manifest.csv")
        try:
            draw_figure(results, all_stats, cfg.conf, caption, title,
                        results_root / "figures" / "comparison.png")
        except ImportError:
            # matplotlib is an optional dependency for the CSV outputs. The
            # report still works without the figure.
            print("[harness] matplotlib not installed; skipping the figure")
        print(f"[harness] wrote summaries to {results_root}")

    # Per-stratum breakdown, saved next to the run.
    strata_out = {}
    for method, s in all_stats.items():
        strata_out[method] = {
            "by_light": breakdown(s, cfg.conf, by="night"),
            "by_presence": breakdown(s, cfg.conf, by="empty"),
            "by_species": per_species(s, cfg.conf, min_frames=5),
        }
    (rdir / "breakdown.json").write_text(json.dumps(strata_out, indent=2,
                                                   default=float),
                                          encoding="utf-8")

    print_summary(results, cfg.conf)
    print(f"\n[harness] done. artefacts in {rdir}")

    return {
        "config": cfg,
        "results": results,
        "all_stats": all_stats,
        "sample": sample,
        "run_dir": rdir,
        "fingerprint": fingerprint,
    }


def compare_runs(run_dirs: list[Path]) -> list[dict]:
    """Read several runs' method_comparison.csv into one table.

    Used to put several seeds or several sample sizes side by side in the
    report.
    """
    rows = []
    for rd in run_dirs:
        rd = Path(rd)
        cfg_path = rd / "config.yaml"
        meta = {}
        if cfg_path.is_file():
            from evaluation.config import RunConfig
            cfg = RunConfig.from_yaml(cfg_path)
            meta = {
                "run_id": cfg.run_id,
                "max_images": cfg.max_images,
                "seed": cfg.seed,
                "conf": cfg.conf,
                "methods": ",".join(cfg.methods),
            }
        import csv
        with (rd / "metrics" / "method_comparison.csv").open(
                newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({**meta, **row})
    return rows
