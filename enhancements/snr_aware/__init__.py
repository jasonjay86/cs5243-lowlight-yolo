"""
SNR-Aware low-light image enhancement.

Owner: Jayson Jay

See enhancements/snr_aware/README.md for the architecture, paper reference,
and setup instructions. This file is the entry point — it must register an
SNRAwareEnhancement class via the @register decorator from base.py.

Suggested layout for the full implementation:
  - model.py      : transformer model definition
  - enhance.py    : inference wrapper (load weights, forward, uint8 output)
  - __init__.py   : this file — registers the Enhancement subclass
"""
