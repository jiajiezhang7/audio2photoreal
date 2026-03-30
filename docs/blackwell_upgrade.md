# Blackwell GPU Upgrade Notes

## Overview

This repository now maintains two separate local environment tracks:

- `a2p-demo`: conservative reproduction environment close to the original repo guidance
- `a2p-blackwell`: Blackwell-targeted GPU runtime environment

The goal of `a2p-blackwell` is practical compatibility on NVIDIA Blackwell GPUs while keeping the project code and model checkpoints unchanged.

## Version Strategy

`a2p-blackwell` uses:

- Python 3.10
- PyTorch 2.7.1
- torchvision 0.22.1
- torchaudio 2.7.1
- official `cu128` wheels
- PyTorch3D built from source from the `facebookresearch/pytorch3d` main branch

The exact PyTorch3D commit installed by `scripts/setup_blackwell_env.sh` is written to:

- `docs/blackwell_pytorch3d_commit.txt`

## Why This Exists

The original repository states it was tested with:

- CUDA 11.7
- Python 3.9
- GCC/G++ 9

That stack is suitable as a reproduction baseline, but it does not support Blackwell-class GPUs in practice.

## Install

```bash
bash scripts/setup_blackwell_env.sh a2p-blackwell
```

## Run

```bash
conda run -n a2p-blackwell python -m demo.demo
```

## Validation Targets

The Blackwell environment is considered healthy when:

- `torch.cuda.is_available()` returns `True`
- the detected device capability is `(12, 0)`
- `pytorch3d` imports successfully
- the PyTorch3D rasterization smoke test runs on GPU
- the demo path runs without triggering the CPU fallback path

## Environment Roles

Use `a2p-demo` when you want:

- minimal drift from the original repo assumptions
- a fallback environment for CPU or older GPU experiments

Use `a2p-blackwell` when you want:

- actual GPU execution on Blackwell hardware
- faster rendering and full-frame output without the CPU fallback stride compromise
