"""
Cell selection for the fine-tuned comparison grid.

Owner: Jason Johnson

Two questions are being asked at once, and they are different questions:

  Q1, test-time enhancement
      One model, trained on raw, evaluated on each condition's test images.
      Does enhancing the input help a detector that never saw enhanced data?

  Q2, train-time enhancement
      Each condition's own model evaluated on that condition's own test
      images. Does training on enhanced data help?

Both are cells of a (model trained on) x (tested on) grid:

    rows    = the checkpoint that scores the images
    columns = which condition's test images it scores

The diagonal is Q2, because model and test condition match. The baseline
rows's off-diagonal cells are Q1, because the raw-trained model is scored on
enhanced input. The (raw, raw) cell belongs to both: it is the shared
reference for everything else.

Duplicating the logic of ``scripts/plot_finetune_eval.py`` is deliberate. It
is eight lines, it is the one part of that script worth keeping identical,
and sharing it through an import would couple the harness to a plotting
script.
"""
from __future__ import annotations

CELL_MODES = ("minimal", "full")


def cells_to_run(names: list[str], mode: str = "minimal") -> list[tuple[str, str]]:
    """Return the (model, test-condition) cells to evaluate.

    The first name is the baseline and must be the raw/no-enhancement
    condition, because every delta is measured against it.

    ``minimal`` scores 2N-1 cells: the baseline's whole row (Q1, plus the
    baseline cell itself) and each condition's diagonal cell (Q2).
    ``full`` scores all N*N cells, which is only worth doing once the models
    are final.

    Raises ValueError rather than silently scoring a partial grid, because a
    missing cell is easy to miss in a table and impossible to notice in a
    figure.
    """
    if mode not in CELL_MODES:
        raise ValueError(f"cell mode must be one of {CELL_MODES}, got {mode!r}")
    if not names:
        raise ValueError("cells_to_run needs at least one condition name")

    base = names[0]
    if mode == "full":
        return [(model, test) for model in names for test in names]

    cells = [(base, test) for test in names]
    cells += [(name, name) for name in names if name != base]
    return cells


def cell_kind(model: str, test_condition: str, base: str) -> str:
    """Label a cell as Q1, Q2, both, or neither.

    Used for the table so a reader can tell which question each row answers
    without working it out from the axes.
    """
    is_q1 = model == base
    is_q2 = model == test_condition
    if is_q1 and is_q2:
        return "Q1+Q2"
    if is_q1:
        return "Q1"
    if is_q2:
        return "Q2"
    return "cross"


def validate_cells(cells: list[tuple[str, str]], models: dict[str, str],
                   conditions: dict[str, str]) -> None:
    """Fail early if a cell names a model or condition that was not supplied.

    A missing weight path or manifest produces an empty or missing column
    later, which is far harder to diagnose than a message at startup.
    """
    missing_models = sorted({m for m, _ in cells if m not in models})
    missing_conditions = sorted({c for _, c in cells if c not in conditions})
    if missing_models:
        raise ValueError(
            f"cell references unknown model(s): {missing_models}; "
            f"supplied: {sorted(models)}")
    if missing_conditions:
        raise ValueError(
            f"cell references unknown condition(s): {missing_conditions}; "
            f"supplied: {sorted(conditions)}")
