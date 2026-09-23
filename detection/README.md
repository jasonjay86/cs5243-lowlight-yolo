# detection

**Owner:** Rich Zanni

Shared YOLO wrapper. Every enhancement method passes its output through the
same detection code so the comparison is fair.

## Interface (proposed)

```python
class Detector:
    def __init__(self, weights_path: str, device: str = "cuda"):
        ...

    def detect(self, image: np.ndarray) -> list[Detection]:
        ...

    @property
    def name(self) -> str:  # e.g. "yolov8s"
        ...
```

The evaluation harness will:
1. Build a `Detector`.
2. Build a list of `Enhancement` instances.
3. For each (image, enhancement): `enhance → detect → save predictions`.
4. Compute mAP per enhancement, then Δ mAP vs raw.

## Notes

- Pin the YOLO weights (`*.pt`) — store under `weights/` (gitignored) or via Git LFS.
- Record YOLO weights, version, and inference settings in
  `experiments/<run>/config.yaml`.
