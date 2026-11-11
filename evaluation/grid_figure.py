"""
Grid figure: render a (model x condition) comparison from ``grid.csv``.

Owner: Jason Johnson

Two panels, no embedded table:

  A. mAP50 heatmap over the full model x condition grid. Cells that were not
     scored (``--cells minimal`` leaves most of the grid empty) are drawn as
     hatched grey rather than blank, so a reader can tell "not evaluated"
     from "evaluated and scored zero".

  B. Delta from the baseline with its 95% paired bootstrap interval, one row
     per non-baseline cell. This is the panel that answers "is the
     difference real", and it is what a table of point estimates cannot show.

Deliberately no table panel: earlier figures in this project sank a lot of
time into table cells overflowing their column and footnotes colliding with
the rendered text. The numbers already live in grid.csv, so the figure only
carries what a figure is for.

Everything here reads plain rows, so it works for any number of conditions,
including one.
"""
from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402,F401  -- matplotlib.text.Text for font scaling
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Scoring-cell kinds that carry a meaningful delta.
_BASELINE_KIND = "Q1+Q2"


def _num(row: dict, key: str, default: float | None = None):
    """Read a numeric cell, tolerating blank or absent columns."""
    val = row.get(key)
    if val is None or str(val).strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _fit_panel_titles(fig) -> None:
    """Shrink any axes title wider than its own panel.

    Panel titles are drawn left-aligned on the axes edge and matplotlib does
    not wrap or clip them, so an over-long one simply runs off the figure.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        title = ax.title
        if not title.get_text():
            continue
        tb = title.get_window_extent(renderer=renderer)
        ab = ax.get_window_extent(renderer=renderer)
        if ab.width <= 0 or tb.width <= ab.width:
            continue
        factor = max(0.5, ab.width / tb.width)
        title.set_fontsize(title.get_fontsize() * factor)


def _wrap_footnote(fig, width_in: float) -> None:
    """Wrap the supxlabel to the canvas width.

    supxlabel is a single Text, so a long note silently runs off the right
    edge instead of wrapping: measured at 7,735 px wide on a 4,000 px slide.
    """
    t = getattr(fig, "_supxlabel", None)
    if t is None:
        return
    fontsize = t.get_fontsize()
    # ~0.58 em average advance for this sans face, scaled to points.
    max_chars = max(40, int(width_in * 72.0 / (fontsize * 0.58)))
    t.set_text(textwrap.fill(t.get_text(), max_chars))
    t.set_verticalalignment("bottom")
    t.set_y(0.012)


def load_rows(csv_path: Path) -> list[dict]:
    """Read grid.csv into typed rows, preserving file order.

    Columns added after a run was produced are treated as absent rather than
    fatal, so an older csv still plots.
    """
    with Path(csv_path).open("r", newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    if not raw:
        raise SystemExit(f"ERROR: {csv_path} has no rows")
    rows = []
    for r in raw:
        fp = _num(r, "empty_fp", 0.0)
        map50 = _num(r, "map50", 0.0)
        rows.append({
            "model": r["model"],
            "condition": r["condition"],
            "kind": r["kind"],
            "n_frames": int(_num(r, "n_frames", 0) or 0),
            "map50": map50,
            "map50_lo": _num(r, "map50_lo", map50),
            "map50_hi": _num(r, "map50_hi", map50),
            "d_map50": _num(r, "d_map50", 0.0),
            "d_lo": _num(r, "d_lo", 0.0),
            "d_hi": _num(r, "d_hi", 0.0),
            "map50_95": _num(r, "map50_95", 0.0),
            "precision": _num(r, "precision", 0.0),
            "recall": _num(r, "recall", 0.0),
            "empty_fp": fp,
            "fp_lo": _num(r, "fp_lo", fp),
            "fp_hi": _num(r, "fp_hi", fp),
            "enh_ms": _num(r, "enh_ms"),
            "yolo_ms": _num(r, "yolo_ms"),
            "gflops_mp": _num(r, "gflops_mp"),
            "n_detections": _num(r, "n_detections"),
        })
    return rows


def _ordered(values: list[str]) -> list[str]:
    """Unique values in first-appearance order (baseline first by design)."""
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def draw_grid_figure(rows: list[dict], out_path: Path, *,
                     title: str | None = None,
                     split: str | None = None,
                     dpi: int = 200,
                     slide: bool = False,
                     return_fig: bool = False):
    """Draw the grid figure and write it to ``out_path``.

    ``return_fig`` is for tests: it keeps the figure open and returns
    ``(path, fig)`` so layout geometry can be asserted. The figure is not
    closed in that case, so the caller must close it.
    """
    if not rows:
        raise SystemExit("ERROR: no rows to draw")

    models = _ordered([r["model"] for r in rows])
    conds = _ordered([r["condition"] for r in rows])
    n_m, n_c = len(models), len(conds)

    lookup = {(r["model"], r["condition"]): r for r in rows}
    deltas = [r for r in rows if r["kind"] != _BASELINE_KIND]
    baseline = next((r for r in rows if r["kind"] == _BASELINE_KIND), rows[0])

    # Size the canvas from the data. A 2-method study and a 5-method study
    # should both look deliberate rather than stretched or cramped.
    rows_needed = max(n_m, len(deltas), 3)
    label_rotation = 30 if n_c > 3 else 0
    if slide:
        width, height = 13.333, 7.5          # 16:9, exactly one slide
        dpi = max(dpi, 300)
        font_scale = 1.55
        need = 2.6 + 0.46 * rows_needed
        if need > height:
            # More cells than a 16:9 canvas holds legibly. Grow rather than
            # cram: an unreadable slide is worse than an unusual ratio.
            height = need
            print(f"[figure] NOTE {rows_needed} rows need {need:.1f}in; "
                  f"this slide is taller than 16:9")
    else:
        width = 7.6 + 1.15 * n_c
        height = 2.6 + 0.46 * rows_needed
        font_scale = 1.0

    # Constrained layout, not hand-computed margins. Long row labels like
    # "retinexformer -> retinexformer" and rotated condition names have
    # widths that depend on the font, and every attempt to budget for them
    # with fixed fractions collided with something. This lets matplotlib
    # measure the text and place the panels around it.
    fig = plt.figure(figsize=(width, height), dpi=dpi, layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    _draw_heatmap(ax_a, models, conds, lookup, label_rotation)
    _draw_deltas(ax_b, deltas, baseline)

    if title is None:
        title = "Fine-tuned models across test-image conditions"
    fig.suptitle(title, fontsize=13, fontweight="bold",
                 horizontalalignment="left", x=0.012)

    note = (f"Baseline: {baseline['model']} on {baseline['condition']} "
            f"(mAP50 {baseline['map50']:.3f}).  Delta is paired on the same "
            f"frames; bars are 95% bootstrap intervals.")
    if split:
        note = f"Split: {split}.  " + note
    # supxlabel rather than fig.text: constrained layout reserves room for it,
    # so it cannot land on top of the x-axis labels.
    fig.supxlabel(note, fontsize=8, color="#444444",
                  horizontalalignment="left", x=0.012)

    if font_scale != 1.0:
        for text in fig.findobj(matplotlib.text.Text):
            try:
                text.set_fontsize(text.get_fontsize() * font_scale)
            except (TypeError, ValueError):
                pass
        # The enlarged title straddled the top edge and clipped. Anchor it to
        # the canvas edge so its full height is inside the image.
        sup = getattr(fig, "_suptitle", None)
        if sup is not None:
            sup.set_verticalalignment("top")
            sup.set_y(1.0)
        # The footnote is the one line a reader in the back row must be able
        # to read, and it starts smallest. Give it a floor of its own.
        supx = getattr(fig, "_supxlabel", None)
        if supx is not None:
            supx.set_fontsize(14)
        _fit_panel_titles(fig)
        _wrap_footnote(fig, width)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    if return_fig:
        return out_path, fig
    plt.close(fig)
    return out_path


def _draw_heatmap(ax, models, conds, lookup, rotation: int = 0) -> None:
    import numpy as np

    n_m, n_c = len(models), len(conds)
    grid = np.full((n_m, n_c), np.nan)
    for i, m in enumerate(models):
        for j, c in enumerate(conds):
            row = lookup.get((m, c))
            if row is not None:
                grid[i, j] = row["map50"]

    scored = grid[~np.isnan(grid)]
    vmin = 0.0
    vmax = max(0.05, float(np.nanmax(scored))) if scored.size else 1.0

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")

    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=vmin,
                   vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_c))
    ax.set_yticks(range(n_m))
    ax.set_xticklabels(conds, rotation=rotation,
                       ha="right" if rotation else "center", fontsize=9)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("tested on", fontsize=9)
    ax.set_ylabel("trained on", fontsize=9)

    # Annotate every cell: the value if scored, an em dash if not. Better to
    # say "not evaluated" than to leave a hole the reader must interpret.
    for i in range(n_m):
        for j in range(n_c):
            row = lookup.get((models[i], conds[j]))
            if row is None:
                ax.text(j, i, "—", ha="center", va="center", fontsize=13,
                        color="#4d4d4d", fontweight="bold")
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, hatch="///",
                                           edgecolor="#bbbbbb", linewidth=0))
            else:
                v = row["map50"]
                ink = "white" if v < (vmin + vmax) / 2 else "#11202b"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=10, color=ink, fontweight="bold")
                if row["kind"] == _BASELINE_KIND:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, edgecolor="#111111",
                                               linewidth=1.6))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("mAP@0.5 by cell", fontsize=10, loc="left", pad=6)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("mAP50", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _draw_deltas(ax, deltas, baseline) -> None:
    if not deltas:
        ax.axis("off")
        ax.text(0.5, 0.5, "Only the baseline cell was scored.\n"
                          "Add conditions or use --cells full.",
                ha="center", va="center", fontsize=9, color="#666666")
        return

    labels = [f"{r['model']} → {r['condition']}" for r in deltas]
    ys = list(range(len(deltas)))[::-1]

    # Pass 1: plot. Pass 2: label, once the axis limits are final. Labelling
    # inside the plotting loop would offset each label against a limit that
    # is still moving as later intervals are added.
    lo_x = hi_x = 0.0
    for y, r in zip(ys, deltas):
        d, lo, hi = r["d_map50"], r["d_lo"], r["d_hi"]
        color = "#b2182b" if d < 0 else "#1b7837"
        ax.plot([lo, hi], [y, y], color=color, linewidth=2.2,
                solid_capstyle="round", alpha=0.85)
        ax.plot([d], [y], "o", color=color, markersize=7,
                markeredgecolor="white", markeredgewidth=1.0)
        lo_x, hi_x = min(lo_x, lo), max(hi_x, hi)

    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
    span = max(hi_x - lo_x, 1e-3)
    ax.set_xlim(lo_x - 0.05 * span, hi_x + 0.26 * span)

    for y, r in zip(ys, deltas):
        d, lo, hi = r["d_map50"], r["d_lo"], r["d_hi"]
        # A difference only earns an unqualified label if the interval
        # clears zero; otherwise it is marked not significant.
        significant = lo > 0 or hi < 0
        tag = "" if significant else " (n.s.)"
        ax.text(hi + 0.03 * span, y, f"{d:+.3f}{tag}", va="center",
                fontsize=8.5, color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Δ mAP50 vs baseline", fontsize=9)
    ax.set_title("Paired change vs baseline", fontsize=10,
                 loc="left", pad=6)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.set_ylim(-0.7, len(deltas) - 0.3)

    handles = [
        Line2D([0], [0], marker="o", color="#b2182b", linestyle="-",
               label="lower than baseline", markersize=6),
        Line2D([0], [0], marker="o", color="#1b7837", linestyle="-",
               label="higher than baseline", markersize=6),
        Patch(facecolor="#d9d9d9", hatch="///", edgecolor="#bbbbbb",
              label="not evaluated"),
    ]
    ax.legend(handles=handles, fontsize=7.5, frameon=False,
              loc="lower right", handlelength=1.6)


def draw_summary_figure(rows: list[dict], out_path: Path, *,
                        title: str | None = None,
                        split: str | None = None,
                        dpi: int = 200,
                        slide: bool = False,
                        return_fig: bool = False):
    """Table plus bars, in the shape Krutin's comparison figure used.

    Top: every column of grid.csv that belongs on a slide, including the
    intervals. Bottom left: mAP@0.5 per cell with its 95% interval and the
    baseline dashed across. Bottom right: empty-frame false-positive rate
    with its interval, because that number is a result in its own right and
    a table cell undersells it.

    ``slide=True`` switches to a 16:9 canvas at 300 dpi with larger type and
    the delta merged with its interval, so the file can be dropped onto a
    slide at full bleed without rescaling or a second layout pass.
    """
    if not rows:
        raise SystemExit("ERROR: no rows to draw")

    n = len(rows)
    baseline = next((r for r in rows if r["kind"] == _BASELINE_KIND), rows[0])
    conds = _ordered([r["condition"] for r in rows])
    palette = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(conds)}

    if slide:
        width, height = 13.333, 7.5          # 16:9, exactly one slide
        dpi = max(dpi, 300)
        font_scale = 1.15
        table_ratio = max(n * 0.38 + 1.1, 2.4)
        panels_ratio = 2.5
        need = table_ratio + panels_ratio + 1.15
        if need > height:
            height = need
            print(f"[figure] NOTE {n} rows need {need:.1f}in; "
                  f"this slide is taller than 16:9")
    else:
        width = 15.0
        table_ratio = 0.34 * n + 0.55
        panels_ratio = 2.7
        height = table_ratio + panels_ratio + 1.15
        font_scale = 1.0

    fig = plt.figure(figsize=(width, height), dpi=dpi, layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[table_ratio, panels_ratio],
                          width_ratios=[1.0, 1.0])
    ax_t = fig.add_subplot(gs[0, :])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_fp = fig.add_subplot(gs[1, 1])
    ax_t.axis("off")

    _draw_table(ax_t, rows, compact=slide)
    labels = [f"{r['model']}\non {r['condition']}" for r in rows]
    _draw_metric_bars(ax_map, rows, labels, palette, baseline,
                      key="map50", lo_key="map50_lo", hi_key="map50_hi",
                      ylabel="mAP@0.5 (test split)",
                      panel="mAP@0.5 by trained model and test input")
    _draw_metric_bars(ax_fp, rows, labels, palette, baseline,
                      key="empty_fp", lo_key="fp_lo", hi_key="fp_hi",
                      ylabel="false-positive rate on empty frames",
                      panel="Empty-frame false positives (lower is better)")

    handles = [Patch(facecolor=palette[c], label=f"tested on {c}")
               for c in conds]
    ax_map.legend(handles=handles, fontsize=7.5, frameon=False,
                  loc="upper right", handlelength=1.2)

    if title is None:
        title = "Fine-tuned YOLOv8n on CCT-20: trained-on × tested-on"
    fig.suptitle(title, fontsize=13, fontweight="bold",
                 horizontalalignment="left", x=0.012)

    note = (f"Q1 = raw-trained model on an enhanced input.  Q2 = model "
            f"trained on the same enhanced data it is tested on.  "
            f"Baseline: {baseline['model']} on {baseline['condition']} "
            f"(mAP@0.5 {baseline['map50']:.3f}).")
    if split:
        note = f"Split: {split}, {baseline['n_frames']:,} frames.  " + note
    note += ("  Bars are 95% paired bootstrap intervals; where an interval "
             "crosses the baseline the difference is not resolved.")
    fig.supxlabel(note, fontsize=8, color="#444444",
                  horizontalalignment="left", x=0.012)

    # Slide variant: scale every text artist at once, after layout, so the
    # type is legible on a projector without a second set of font constants
    # scattered through the drawing code.
    if font_scale != 1.0:
        for text in fig.findobj(matplotlib.text.Text):
            try:
                text.set_fontsize(text.get_fontsize() * font_scale)
            except (TypeError, ValueError):
                pass
        # The enlarged title straddled the top edge and clipped. Anchor it to
        # the canvas edge so its full height is inside the image.
        sup = getattr(fig, "_suptitle", None)
        if sup is not None:
            sup.set_verticalalignment("top")
            sup.set_y(1.0)
        supx = getattr(fig, "_supxlabel", None)
        if supx is not None:
            supx.set_fontsize(14)
        _fit_panel_titles(fig)
        _wrap_footnote(fig, width)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    if return_fig:
        return out_path, fig
    plt.close(fig)
    return out_path


def _draw_table(ax, rows: list[dict], compact: bool = False) -> None:
    if compact:
        # Merge the delta with its interval so nine columns fit a 16:9 slide
        # at legible type. Same information, one fewer column.
        cols = ["Trained on", "Tested on", "Q", "mAP@0.5",
                "d mAP@0.5 [95% CI]", "mAP@0.5:0.95", "Precision", "Recall",
                "Empty FP"]
        cell = []
        for r in rows:
            cell.append([
                r["model"], r["condition"], r["kind"],
                f"{r['map50']:.3f}",
                f"{r['d_map50']:+.3f} [{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]",
                f"{r['map50_95']:.3f}",
                f"{r['precision']:.3f}",
                f"{r['recall']:.3f}",
                f"{r['empty_fp']:.3f}",
            ])
    else:
        cols = ["Trained on", "Tested on", "Q", "mAP@0.5", "d mAP@0.5",
                "95% CI on d", "mAP@0.5:0.95", "Precision", "Recall",
                "Empty FP"]
        cell = []
        for r in rows:
            cell.append([
                r["model"], r["condition"], r["kind"],
                f"{r['map50']:.3f}",
                f"{r['d_map50']:+.3f}",
                f"[{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]",
                f"{r['map50_95']:.3f}",
                f"{r['precision']:.3f}",
                f"{r['recall']:.3f}",
                f"{r['empty_fp']:.3f}",
            ])

    # Column widths from the longest string in each column. Auto-sizing needs
    # a draw pass and behaves differently across backends; this is
    # deterministic and cannot overflow its axes.
    widths = []
    for i, head in enumerate(cols):
        widths.append(max(len(head), max((len(row[i]) for row in cell),
                                         default=0)))
    total = sum(widths) or 1
    col_widths = [w / total for w in widths]

    table = ax.table(cellText=cell, colLabels=cols, loc="center",
                     cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)
    for (row_i, _col), cellobj in table.get_celld().items():
        cellobj.set_edgecolor("#cccccc")
        if row_i == 0:
            cellobj.set_facecolor("#eef1f4")
            cellobj.set_text_props(fontweight="bold")
        elif rows[row_i - 1]["kind"] == _BASELINE_KIND:
            cellobj.set_facecolor("#fff8e1")
    ax.set_title("Every scored cell, all metrics", fontsize=10, loc="left",
                 pad=4)


def _draw_metric_bars(ax, rows, labels, palette, baseline, *, key,
                      lo_key, hi_key, ylabel, panel) -> None:
    n = len(rows)
    xs = range(n)
    colours = [palette[r["condition"]] for r in rows]
    vals = [r[key] for r in rows]
    lower = [max(0.0, r[key] - (r[lo_key] if r[lo_key] is not None
                                else r[key])) for r in rows]
    upper = [max(0.0, (r[hi_key] if r[hi_key] is not None else r[key])
                  - r[key]) for r in rows]

    ax.bar(xs, vals, color=colours, width=0.62, edgecolor="white",
           linewidth=0.8)
    # Only draw an interval where it actually has width: older csvs have
    # none, and a zero-length bar cap just looks like an artefact.
    if any(lower) or any(upper):
        ax.errorbar(xs, vals, yerr=[lower, upper], fmt="none",
                    ecolor="#333333", elinewidth=1.1, capsize=3)

    ax.axhline(baseline[key], color="#333333", linestyle="--", linewidth=1.0,
               alpha=0.75)

    top = max([v + u for v, u in zip(vals, upper)] + [baseline[key], 0.05])
    for x, v, u in zip(xs, vals, upper):
        ax.text(x, v + u + 0.02 * top, f"{v:.3f}", ha="center",
                va="bottom", fontsize=8)

    rotation = 0
    if n > 4:
        rotation = 20
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=rotation,
                       ha="right" if rotation else "center")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, top * 1.18)
    ax.set_title(panel, fontsize=10, loc="left", pad=6)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Draw the grid figure from an existing grid.csv. "
                    "Needs no GPU and no torch.")
    ap.add_argument("csv", help="path to grid.csv")
    ap.add_argument("--out", default=None, help="output png (default: beside "
                                                "the csv as grid.png)")
    ap.add_argument("--figure", default="grid",
                    choices=["grid", "summary", "both"],
                    help="grid: heatmap + delta intervals. summary: table + "
                         "bars with every metric. both: write each.")
    ap.add_argument("--slide", action="store_true",
                    help="16:9 canvas at 300 dpi with larger type, sized to "
                         "paste straight onto a slide")
    ap.add_argument("--title", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)

    rows = load_rows(Path(args.csv))
    out = Path(args.out) if args.out else Path(args.csv).with_suffix(".png")
    base = out.parent / out.stem
    tag = "-slide" if args.slide else ""

    # Naming is fixed here rather than derived twice, so a --slide run writes
    # grid-slide.png / grid-summary-slide.png regardless of which figure was
    # asked for. An explicit --out is honoured verbatim for a single figure.
    if args.figure == "grid":
        targets = [Path(args.out) if args.out else Path(f"{base}{tag}.png")]
    elif args.figure == "summary":
        targets = [Path(args.out) if args.out
                   else Path(f"{base}-summary{tag}.png")]
    else:
        targets = [Path(f"{base}{tag}.png"),
                   Path(f"{base}-summary{tag}.png")]

    written = []
    if args.figure in ("grid", "both"):
        written.append(draw_grid_figure(rows, targets[0], title=args.title,
                                        split=args.split, dpi=args.dpi,
                                        slide=args.slide))
    if args.figure in ("summary", "both"):
        written.append(draw_summary_figure(rows, targets[-1],
                                           title=args.title,
                                           split=args.split, dpi=args.dpi,
                                           slide=args.slide))
    for path in written:
        # Absolute, because "wrote grid-slide.png" is useless when the file
        # lands in a run directory rather than the working directory.
        print(f"[figure] wrote {Path(path).resolve()}  ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
