"""Enhancements package.

Provides the Enhancement interface (base.py) and a registry of concrete
implementations. Subpackages (raw, clahe, snr_aware) each register a single
class via the @register decorator in base.py.

To discover all registered enhancements:

    from enhancements.base import available_enhancements
    print(available_enhancements())

To use a specific enhancement:

    from enhancements.base import get_enhancement
    enh = get_enhancement("raw")
    out = enh.enhance(img)

Subpackages must be imported once for their @register decorator to run.
Either import them explicitly (e.g. ``import enhancements.raw``) or rely on
get_enhancement("name") to trigger the registry lookup.
"""
