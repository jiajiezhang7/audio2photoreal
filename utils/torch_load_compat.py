"""Helpers for loading trusted local checkpoints across PyTorch versions."""

import torch


def load_trusted(path, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return torch.load(path, *args, **kwargs)
