# raw

**Owner:** (unassigned — grab it, it's a 30-minute task)

The raw enhancement is a no-op — it returns the input image unchanged. It
serves as the **baseline** that the other methods are compared against via Δ mAP.

The whole point of the project is `mAP(method) − mAP(raw)`, so this module
must exist and be registered before any eval runs.

## Files

- `__init__.py` — exports a registered `RawEnhancement` class implementing
  `Enhancement.enhance()` as a passthrough.

## Implementation hint

```python
import numpy as np
from enhancements.base import Enhancement, register

@register
class RawEnhancement(Enhancement):
    def name(self) -> str:
        return "raw"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        return image  # baseline
```
