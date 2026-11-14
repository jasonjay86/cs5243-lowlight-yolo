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


def cells_to_run(models: list[str], mode: str = "minimal", *,
                 conditions: list[str] | None = None) -> list[tuple[str, str]]:
    """Return the (model, test-condition) cells to evaluate.

    ``models`` and ``conditions`` are separate sets on purpose: a condition
    can have test images without having a fine-tuned model of its own. That
    happens whenever an enhancement was only run over the test split, and it
    is exactly the case that makes the Q1 cell worth having — a raw-trained
    detector scored on that condition's images.

    The first model is the baseline and must also be the first condition,
    because every delta is measured against the (baseline, baseline) cell.

    ``minimal`` scores the baseline's whole row (Q1, plus the baseline cell
    itself) and each model's diagonal cell where that model also has a test
    condition (Q2). When the two sets are equal that is 2N-1 cells.
    ``full`` scores every model against every condition.
    """
    if mode not in CELL_MODES:
        raise ValueError(f"cell mode must be one of {CELL_MODES}, got {mode!r}")
    if not models:
        raise ValueError("cells_to_run needs at least one model")

    conds = list(conditions) if conditions is not None else list(models)
    if not conds:
        raise ValueError("cells_to_run needs at least one condition")

    base = models[0]
    if conds[0] != base:
        raise ValueError(
            f"the first model ({base!r}) must also be the first condition "
            f"({conds[0]!r}); every delta is measured against that cell")

    if mode == "full":
        return [(model, test) for model in models for test in conds]

    cells = [(base, test) for test in conds]
    cells += [(model, model) for model in models
              if model != base and model in conds]
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
