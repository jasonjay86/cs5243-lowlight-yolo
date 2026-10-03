"""
SNR-Aware low-light image enhancement.

Owner: Jason Johnson

See enhancements/snr_aware/README.md for the architecture, paper reference,
and setup instructions. This file is the entry point — it registers an
SNRAwareEnhancement class via the @register decorator from base.py.

Suggested layout for the full implementation:
  - model.py      : transformer model definition (vendored from upstream)
  - enhance.py    : inference wrapper (load weights, forward, uint8 output)
  - __init__.py   : this file — registers the Enhancement subclass
"""

from __future__ import annotations

from enhancements.base import Enhancement, register
from enhancements.snr_aware.enhance import SNRAwareEnhancement


@register
class SNRAwareEnhancementRegistered(SNRAwareEnhancement, Enhancement):
    """Registered adapter for the SNR-Aware inference wrapper.

    The actual implementation lives in enhance.py.SNRAwareEnhancement. This
    subclass only exists to attach the @register decorator and declare the
    Enhancement ABC contract — same pattern as the raw module's
    RawEnhancement, which is also a single class with @register + Enhancement.

    All keyword args (weights_path, device) are forwarded to SNRAwareEnhancement's
    __init__ via *args/**kwargs — no extra constructor needed here.
    """

    def __init__(self, *args, **kwargs):
        # SNRAwareEnhancement.__init__ already takes (weights_path, device).
        # We don't override it; this subclass is just the registered handle.
        super().__init__(*args, **kwargs)