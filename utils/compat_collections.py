"""Compatibility helpers for legacy dependencies on Python 3.10+."""

import collections
import collections.abc


for name in (
    "Mapping",
    "MutableMapping",
    "Sequence",
):
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))
