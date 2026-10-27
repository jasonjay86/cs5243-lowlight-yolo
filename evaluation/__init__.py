"""
Evaluation package.

Compares detection results across enhancement methods and produces the
method comparison table (delta mAP, FPR, latency) under results/.

Public API:
    from evaluation import RunConfig, StrataConfig, run_evaluation
"""
from __future__ import annotations

import os

# OpenMP-on-Windows: PyTorch and Ultralytics both link against an OpenMP
# runtime, and on a stock conda+pip install the second one to initialise
# refuses to load (OMP Error #15: "Initializing libomp.dll, but found
# libiomp5md.dll already initialized"). The crash happens when the first
# enhancement module imports torch, which is mid-run rather than at import
# time, so the flag has to be set before anything pulls torch in.
#
# KMP_DUPLICATE_LIB_OK=TRUE is Intel's documented workaround for the
# inference path (as opposed to training). setdefault() leaves an
# externally-set value alone, so anyone who prefers to set it in their
# shell keeps their choice.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from evaluation.config import RunConfig, StrataConfig
from evaluation.harness import compare_runs, run_evaluation

__all__ = [
    "RunConfig",
    "StrataConfig",
    "run_evaluation",
    "compare_runs",
]
