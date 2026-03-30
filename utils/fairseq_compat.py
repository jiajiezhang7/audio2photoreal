"""Compatibility helpers for fairseq on newer PyTorch versions."""

from contextlib import contextmanager
from typing import Sequence

import fairseq
import torch


@contextmanager
def _legacy_torch_load():
    original_load = torch.load

    def wrapped_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = wrapped_load
    try:
        yield
    finally:
        torch.load = original_load


def load_model_ensemble_and_task(paths: Sequence[str]):
    with _legacy_torch_load():
        return fairseq.checkpoint_utils.load_model_ensemble_and_task(list(paths))
