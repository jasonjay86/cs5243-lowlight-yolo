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
    cache = PredictionCache(rdir, fingerprint)
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
            print(f"[harness] SKIPPING {method}: no cached predictions")
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
    caption = (f"{len(frames)} frames ({len(frames) - n_empty} with animals, "
               f"{n_empty} empty), {n_boxes} ground-truth boxes. "
               f"Same YOLO ({cfg.yolo_weights}), imgsz {cfg.imgsz}, "
               f"P/R/FP at conf {cfg.conf}, IoU 0.5. "
               f"Latency measured on {dev}. CIs: paired bootstrap over frames "
               f"({cfg.bootstrap} resamples).")

    if write_results:
        write_method_comparison(results, results_root / "method_comparison.csv")
        write_false_positives(results, results_root / "false_positives.csv")
        write_latency(results, results_root / "latency.csv")
        write_sample_manifest(sample, results_root / "sample_manifest.csv")
        try:
            draw_figure(results, all_stats, cfg.conf, caption,
                        "Raw vs enhanced + YOLO on CCT-20 night frames",
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
