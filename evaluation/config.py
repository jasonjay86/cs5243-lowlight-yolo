"""
Run configuration for the evaluation harness.

Owner: Jason Johnson

Every run must be reproducible. That means the configuration that produced a
number has to travel with the number: which frames, which enhancement, which
YOLO weights, which thresholds, which code revision, and on what hardware.

This module defines the RunConfig, its serialisation to YAML, and the
fingerprint used to invalidate a prediction cache when any of those inputs
change.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StrataConfig:
    """How the frame sample is stratified.

    Attributes:
        empty_fraction: share of the sample that must be animal-free frames.
            Needed for the empty-frame false-positive metric, which has no
            power without them.
        day_fraction: share of the ANIMAL frames that should be daytime.
            A daytime control is a useful sanity check on the enhancement
            itself: a good enhancer should barely change a well-lit frame.
        species_balance: "natural" keeps the dataset's own species
            distribution (right for a dataset-level claim); "uniform" spreads
            the sample evenly across species (better per-species power, but
            then the sample no longer reflects the dataset).
        min_per_species: floor on frames per species when species_balance is
            "uniform". Prevents a species with 1 available frame from being
            over-weighted.
        night_start_hour / night_end_hour: hour range treated as night.
            19 -> 6 means "19:00 through 05:59". Used to derive the day/night
            stratum from the dataset's own timestamp field.
    """
    empty_fraction: float = 0.25
    day_fraction: float = 0.25
    species_balance: str = "natural"
    min_per_species: int = 1
    night_start_hour: int = 19
    night_end_hour: int = 6


@dataclass
class RunConfig:
    """Everything needed to reproduce one evaluation run.

    Only `images_dir`, `annotations` and `methods` are truly required. The
    rest have defaults chosen to match the upstream evaluation script so the
    two produce comparable numbers.
    """
    # ---- data -------------------------------------------------------------
    images_dir: str = ""
    annotations: str = ""
    image_labels: str | None = None

    # ---- what to run ------------------------------------------------------
    methods: list[str] = field(default_factory=lambda: ["raw", "clahe", "snr-aware"])
    max_images: int = 500

    # ---- detector settings ------------------------------------------------
    imgsz: int = 640
    conf: float = 0.25           # threshold for P/R and the FP rate
    conf_floor: float = 0.001    # floor for the PR curve (full curve)
    device: str = "cpu"          # "cpu" or "cuda"
    yolo_weights: str = "detection/yolov8n.pt"

    # ---- statistics -------------------------------------------------------
    bootstrap: int = 1000
    seed: int = 0

    # ---- sample -----------------------------------------------------------
    strata: StrataConfig = field(default_factory=StrataConfig)

    # ---- bookkeeping ------------------------------------------------------
    run_id: str = ""
    repo_root: str = ""

    # ------------------------------------------------------------------
    def resolved(self, repo_root: Path) -> "RunConfig":
        """Return a copy with repo-relative paths resolved to absolute."""
        root = Path(repo_root)
        # asdict() turns the nested StrataConfig dataclass into a plain dict,
        # which would break every `cfg.strata.some_field` access downstream.
        # Rebuild the nested dataclass explicitly.
        out = RunConfig(
            **{f.name: getattr(self, f.name) for f in fields(self)
               if f.name != "strata"},
            strata=StrataConfig(**asdict(self.strata)),
        )
        out.repo_root = str(root)
        for field_name in ("images_dir", "annotations", "image_labels",
                           "yolo_weights"):
            val = getattr(out, field_name)
            if val and not Path(val).is_absolute():
                setattr(out, field_name, str(root / val))
        if not out.run_id:
            out.run_id = default_run_id(out)
        return out

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        # asdict() already flattens the nested dataclass into a dict, which is
        # what yaml wants. The env fingerprint is added under a reserved key.
        data["_env"] = env_fingerprint()
        path.write_text(yaml.safe_dump(data, sort_keys=True, default_flow_style=False),
                        encoding="utf-8")

    @staticmethod
    def from_yaml(path: Path) -> "RunConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        strata = StrataConfig(**(data.pop("strata", None) or {}))
        data.pop("_env", None)
        known = {f.name for f in fields(RunConfig)}
        data = {k: v for k, v in data.items() if k in known}
        return RunConfig(strata=strata, **data)

    # ------------------------------------------------------------------
    def analysis_fingerprint(self, frame_ids: list[str],
                             gt_digest: str = "") -> str:
        """Hash of every input that can change the numbers.

        Used to decide whether a cached prediction file is still valid.
        Includes the frame list, the annotation file's contents, the
        enhancement/detector settings, and the environment.

        `gt_digest` is a hash of the RESOLVED ground-truth boxes. The
        annotation file's hash is not enough on its own: the boxes depend
        on the on-disk image dimensions as well, so a change in how the
        boxes are resolved must invalidate the cache too. Cached
        detections are still valid in that case, but the cached *scores*
        are not, so the digest is part of the fingerprint.
        """
        h = hashlib.sha256()

        def feed(label: str, value: Any) -> None:
            h.update(label.encode("utf-8"))
            h.update(b"\x00")
            h.update(str(value).encode("utf-8"))
            h.update(b"\x00")

        feed("methods", ",".join(self.methods))
        feed("max_images", self.max_images)
        feed("imgsz", self.imgsz)
        feed("conf", self.conf)
        feed("conf_floor", self.conf_floor)
        feed("device", self.device)
        feed("yolo_weights", self.yolo_weights)
        feed("seed", self.seed)
        feed("strata", json.dumps(asdict(self.strata), sort_keys=True))
        feed("n_frames", len(frame_ids))
        for fid in frame_ids:
            h.update(fid.encode("utf-8"))
            h.update(b"\n")
        feed("gt_digest", gt_digest)

        for label, p in (("annotations", self.annotations),
                         ("image_labels", self.image_labels),
                         ("yolo_weights_file", self.yolo_weights)):
            if p:
                path = Path(p)
                if path.is_file():
                    feed(label, file_sha256(path))
                elif label == "yolo_weights_file":
                    feed(label, "MISSING")
        feed("env", json.dumps(env_fingerprint(), sort_keys=True))
        return h.hexdigest()[:16]


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed so a 100 MB annotation file is fine."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_sha(repo_root: Path) -> str:
    """Short SHA of the checked-out revision, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        sha = out.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def env_fingerprint() -> dict:
    """Record the software and hardware a run happened on.

    Latency numbers are meaningless without the device and the library
    versions, so these go into every run's config.yaml.
    """
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod in ("torch", "ultralytics", "numpy", "opencv"):
        try:
            m = __import__(mod)
            # str() is required: torch.__version__ is a TorchVersion object,
            # which yaml.safe_dump cannot represent. It would crash the run
            # when the config is written.
            info[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            info[mod] = "not installed"
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        info["cuda_available"] = False
    return info


def default_run_id(cfg: RunConfig) -> str:
    """A sortable run id: method-count, frame count, seed, short timestamp."""
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"run-{cfg.max_images}f-s{cfg.seed}-{stamp}"


def run_dir(cfg: RunConfig, experiments_root: Path) -> Path:
    """Where a run's artefacts live: experiments/<run_id>/."""
    return Path(experiments_root) / cfg.run_id
