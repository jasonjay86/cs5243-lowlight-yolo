"""
Report writers for the evaluation harness.

Owner: Jason Johnson

Produces the artefacts the project spec calls for:

  results/method_comparison.csv   one row per method, the headline table
  results/false_positives.csv      empty-frame FP rate per method
  results/latency.csv              per-method, per-stage timing
  results/figures/comparison.png   PR curves + delta-mAP bars

Small summaries are written to results/ for committing. Large artefacts stay
under experiments/<run-id>/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Categorical colour slots, fixed order per method. Text never wears these;
# they only mark lines, bars and swatches.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7d5fff"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{d}f}"


def write_method_comparison(results, path: Path) -> Path:
    """The headline table: one row per method."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "mAP@0.5", "mAP@0.5 CI low", "mAP@0.5 CI high",
        "mAP@0.5:0.95", "precision", "recall",
        "d_mAP@0.5 vs raw", "d CI low", "d CI high",
        "empty-frame FP rate", "FP CI low", "FP CI high",
        "n_detections", "enhance ms/img", "YOLO ms/img", "GFLOPs/MP",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "method": r.method,
                "mAP@0.5": fmt(r.ap50),
                "mAP@0.5 CI low": fmt(r.ap50_ci[0]),
                "mAP@0.5 CI high": fmt(r.ap50_ci[1]),
                "mAP@0.5:0.95": fmt(r.map5095),
                "precision": fmt(r.precision),
                "recall": fmt(r.recall),
                "d_mAP@0.5 vs raw": fmt(r.d_ap50, 4),
                "d CI low": fmt(r.d_ci[0], 4),
                "d CI high": fmt(r.d_ci[1], 4),
                "empty-frame FP rate": fmt(r.fp_rate, 4),
                "FP CI low": fmt(r.fp_ci[0], 4),
                "FP CI high": fmt(r.fp_ci[1], 4),
                "n_detections": r.n_detections,
                "enhance ms/img": fmt(r.enh_ms, 1),
                "YOLO ms/img": fmt(r.yolo_ms, 1),
                "GFLOPs/MP": fmt(r.gflops_mp, 1) if r.gflops_mp is not None else "n/a",
            })
    return path


def write_false_positives(results, path: Path) -> Path:
    """Empty-frame false positives, per method."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "empty-frame FP rate", "CI low", "CI high",
            "FP rate (all detections, conf floor)", "precision",
        ])
        w.writeheader()
        for r in results:
            w.writerow({
                "method": r.method,
                "empty-frame FP rate": fmt(r.fp_rate, 4),
                "CI low": fmt(r.fp_ci[0], 4),
                "CI high": fmt(r.fp_ci[1], 4),
                "FP rate (all detections, conf floor)": "",
                "precision": fmt(r.precision),
            })
    return path


def write_latency(results, path: Path) -> Path:
    """Per-method, per-stage timing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "enhance ms/img", "YOLO ms/img", "total ms/img",
            "enhance share of total", "GFLOPs/MP",
        ])
        w.writeheader()
        for r in results:
            total = r.enh_ms + r.yolo_ms
            share = (r.enh_ms / total) if total > 0 else float("nan")
            w.writerow({
                "method": r.method,
                "enhance ms/img": fmt(r.enh_ms, 1),
                "YOLO ms/img": fmt(r.yolo_ms, 1),
                "total ms/img": fmt(total, 1),
                "enhance share of total": fmt(share, 3),
                "GFLOPs/MP": fmt(r.gflops_mp, 1) if r.gflops_mp is not None else "n/a",
            })
    return path


def write_sample_manifest(sample: dict, path: Path) -> Path:
    """How the frame sample is composed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerow(["n_frames", sample.get("n_frames", "")])
        w.writerow(["n_empty", sample.get("n_empty", "")])
        w.writerow(["n_night", sample.get("n_night", "")])
        w.writerow(["n_day", sample.get("n_day", "")])
        w.writerow(["n_ground_truth_boxes", sample.get("n_boxes", "")])
        w.writerow([])
        w.writerow(["species", "frames"])
        for sp, n in (sample.get("species") or {}).items():
            w.writerow([sp, n])
    return path


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def draw_figure(results, all_stats, conf: float, caption: str, title: str,
                out_path: Path, demo: bool = False) -> Path:
    """Results table on top, PR curves and delta-mAP bars below.

    The layout matches scripts/evaluate_enhancements.py (Krutin Patel,
    PR #8) so the two figures are directly comparable: same table columns,
    same two panels, same palette, same caption placement.

    `demo=True` stamps a red banner over the title. It exists so a layout
    check can never be mistaken for a real result.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.method for r in results]
    color = {m: SERIES[i % len(SERIES)] for i, m in enumerate(names)}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
    })

    fig = plt.figure(figsize=(15, 10), dpi=200, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.55, 1.25], hspace=0.32,
                          wspace=0.18, left=0.05, right=0.97, top=0.90,
                          bottom=0.10)
    fig.text(0.05, 0.955, title, fontsize=17, fontweight="bold", ha="left")
    if demo:
        fig.text(0.05, 0.925,
                 "SYNTHETIC DEMO DATA: layout check only, not real results",
                 fontsize=13, fontweight="bold", color="#b3261e", ha="left")

    # ---- table ----
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    head = ["", "Condition", "mAP@0.5\n[95% CI]", "mAP@\n0.5:0.95",
            f"Precision\n(conf>={conf})", f"Recall\n(conf>={conf})",
            "Δ mAP@0.5 vs raw\n[95% CI]", "Empty-frame\nFP rate [95% CI]",
            "Enhance\n(ms/img)", "YOLO\n(ms/img)", "Enhance\nGFLOPs/MP"]
    cells = []
    for r in results:
        base = r.method == names[0]
        cells.append([
            "", r.method,
            f"{fmt(r.ap50)}\n[{fmt(r.ap50_ci[0])}, {fmt(r.ap50_ci[1])}]",
            fmt(r.map5095), fmt(r.precision), fmt(r.recall),
            "baseline" if base else
            f"{fmt(r.d_ap50, 4)}\n[{fmt(r.d_ci[0], 4)}, {fmt(r.d_ci[1], 4)}]",
            f"{fmt(r.fp_rate, 2)}\n[{fmt(r.fp_ci[0], 2)}, {fmt(r.fp_ci[1], 2)}]",
            fmt(r.enh_ms, 1), fmt(r.yolo_ms, 1),
            fmt(r.gflops_mp, 1) if r.gflops_mp is not None else "n/a",
        ])
    # The condition column holds the enhancement name alone. The "+ YOLO"
    # suffix that used to sit on every row is redundant: the title, the
    # axis labels and the legend all say that the detector is YOLO. It
    # also made the longest name ("retinexformer + YOLO") overflow its
    # cell into the neighbouring column.
    widths = [0.015, 0.155, 0.11, 0.07, 0.08, 0.08, 0.145, 0.125, 0.07,
              0.07, 0.08]
    tb = ax.table(cellText=cells, colLabels=head, colWidths=widths,
                  loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(11)
    tb.scale(1, 2.9)
    for (ri, ci), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        c.set_linewidth(0.8)
        if ri == 0:
            c.set_facecolor("#efeeea")
            c.set_text_props(fontweight="bold", color=INK, fontsize=10)
        else:
            c.set_facecolor(SURFACE)
            c.set_text_props(color=INK)
            if ci == 0:
                c.set_facecolor(color[names[ri - 1]])
            if ci == 1:
                c.set_text_props(fontweight="bold", ha="left")

    # The footnote goes UNDER the table, not at a fixed axes coordinate.
    # tb.scale(1, 2.9) makes the table taller than the axes it sits in, so
    # it grows downward past the axes boundary. A footnote placed at a fixed
    # axes-relative y lands inside the table's last row and overlaps it.
    # Measure the table's real extent instead and place the footnote just
    # below it in figure coordinates.
    fig.canvas.draw()
    _renderer = fig.canvas.get_renderer()
    _tb_box = tb.get_window_extent(_renderer)
    _fig_box = fig.get_window_extent(_renderer)
    _foot_y = (_tb_box.y0 / _fig_box.height) - 0.022
    fig.text(0.05, _foot_y,
             "Class-agnostic: every ground-truth box is one 'animal' class; "
             "a detection counts if it boxes the right place, whatever its "
             "COCO label. Higher is better except FP rate and latency.",
             fontsize=10, color=INK2, va="top", ha="left")

    # ---- panel A: PR curves ----
    from evaluation.metrics import pr_curve
    a1 = fig.add_subplot(gs[1, 0])
    a1.set_facecolor(SURFACE)
    n_img = len(next(iter(all_stats.values())).n_gt)
    ones = np.ones(n_img)
    for m in names:
        rec, prec, _ = pr_curve(all_stats[m], 0, ones)
        if len(prec):
            prec = np.maximum.accumulate(prec[::-1])[::-1]
        a1.plot(rec, prec, color=color[m], lw=2, label=m + " + YOLO")
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1.02)
    a1.set_xlabel("Recall (IoU 0.5)")
    a1.set_ylabel("Precision")
    a1.set_title("A. Precision-recall curve", loc="left", fontsize=13,
                 fontweight="bold")
    a1.grid(color=GRID, lw=0.8)
    a1.set_axisbelow(True)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a1.legend(frameon=False, loc="upper right", fontsize=11)

    # ---- panel B: delta mAP ----
    a2 = fig.add_subplot(gs[1, 1])
    others = [r for r in results if r.method != names[0]]
    xs = np.arange(len(others))
    if others:
        lo_all = min([0.0] + [r.d_ci[0] for r in others])
        hi_all = max([0.0] + [r.d_ci[1] for r in others])
        pad = (hi_all - lo_all) * 0.18 or 0.01
        a2.set_ylim(lo_all - pad, hi_all + pad * 1.6)
        a2.set_xlim(-0.7, len(others) - 0.3)
        for x, r in zip(xs, others):
            lo, hi = r.d_ci
            a2.bar(x, r.d_ap50, width=0.38, color=color[r.method])
            a2.errorbar(x, r.d_ap50,
                        yerr=[[r.d_ap50 - lo], [hi - r.d_ap50]],
                        color=INK2, capsize=5, lw=1.6)
            a2.text(x, hi + pad * 0.25, f"{r.d_ap50:+.3f}", ha="center",
                    va="bottom", fontsize=12, fontweight="bold")
    a2.axhline(0, color=INK2, lw=1.2)
    a2.set_xticks(xs)
    a2.set_xticklabels([r.method + " + YOLO" for r in others], fontsize=11)
    a2.set_xlabel("below 0 = the enhancement hurt detection; "
                   "a CI that crosses 0 = no clear difference")
    a2.set_ylabel("Δ mAP@0.5 vs raw baseline (95% bootstrap CI)")
    a2.set_title("B. Change in mAP@0.5 vs raw baseline", loc="left",
                 fontsize=13, fontweight="bold")
    a2.grid(axis="y", color=GRID, lw=0.8)
    a2.set_axisbelow(True)
    for s_ in ("top", "right", "bottom"):
        a2.spines[s_].set_visible(False)
    a2.tick_params(axis="x", length=0)

    fig.text(0.05, 0.03, caption, fontsize=10, color=INK2, ha="left",
             va="center", wrap=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def print_summary(results, conf: float) -> None:
    """Console table, matching the upstream script's format."""
    print("\n" + f"{'condition':<16}{'mAP50':>9}{'mAP50-95':>11}"
          f"{'P':>8}{'R':>8}{'dAP50':>9}{'emptyFP':>10}"
          f"{'enh ms':>10}{'yolo ms':>10}")
    for r in results:
        print(f"{r.method:<16}{fmt(r.ap50):>9}{fmt(r.map5095):>11}"
              f"{fmt(r.precision):>8}{fmt(r.recall):>8}"
              f"{fmt(r.d_ap50, 4):>9}{fmt(r.fp_rate, 3):>10}"
              f"{fmt(r.enh_ms, 1):>10}{fmt(r.yolo_ms, 1):>10}")
