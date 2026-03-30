#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-a2p-blackwell}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
TORCHAUDIO_VERSION="2.7.1"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
PYTORCH3D_REPO="https://github.com/facebookresearch/pytorch3d.git"
PYTORCH3D_CLONE_DIR="${REPO_ROOT}/.cache/pytorch3d"
SIDECAR_PATH="${REPO_ROOT}/docs/blackwell_pytorch3d_commit.txt"

cd "${REPO_ROOT}"

run_in_env() {
    conda run --no-capture-output -n "${ENV_NAME}" "$@"
}

env_prefix() {
    conda env list | awk -v env="${ENV_NAME}" '$1 == env {print $2}'
}

env_exists() {
    conda env list | awk 'NR > 2 {print $1}' | grep -Fxq "${ENV_NAME}"
}

require_files() {
    local missing=0
    for path in "$@"; do
        if [ ! -f "${path}" ]; then
            echo "Missing required file: ${path}" >&2
            missing=1
        fi
    done
    if [ "${missing}" -ne 0 ]; then
        return 1
    fi
}

pip_modules_ready() {
    run_in_env python - <<'PY'
import importlib.util
required = [
    "attrdict",
    "colorama",
    "einops",
    "fairseq",
    "fastapi",
    "fvcore",
    "gradio",
    "gradio_client",
    "huggingface_hub",
    "hydra",
    "iopath",
    "mediapy",
    "omegaconf",
    "cv2",
    "starlette",
    "tensorboard",
    "tensorboardX",
    "tqdm",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
}

if env_exists; then
    echo "Updating conda environment: ${ENV_NAME}"
    conda env update -n "${ENV_NAME}" -f environment.blackwell.yml --prune
else
    echo "Creating conda environment: ${ENV_NAME}"
    conda env create -n "${ENV_NAME}" -f environment.blackwell.yml
fi

run_in_env python -m pip install --upgrade "pip<24" "setuptools<70" "wheel" "Cython<3"

run_in_env python -m pip uninstall -y torch torchvision torchaudio pytorch3d || true
run_in_env python -m pip install \
    "torch==${PYTORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${PYTORCH_INDEX_URL}"

if pip_modules_ready; then
    echo "Pinned project pip dependencies already exist."
else
    run_in_env python -m pip install -r requirements.blackwell.lock.txt
    run_in_env python -m pip install iopath fvcore
fi

mkdir -p "$(dirname "${SIDECAR_PATH}")"
mkdir -p "${PYTORCH3D_CLONE_DIR}"

if [ ! -d "${PYTORCH3D_CLONE_DIR}/.git" ]; then
    git clone --depth 1 "${PYTORCH3D_REPO}" "${PYTORCH3D_CLONE_DIR}"
fi

if [ -f "${SIDECAR_PATH}" ] && [ -s "${SIDECAR_PATH}" ]; then
    PYTORCH3D_COMMIT="$(cat "${SIDECAR_PATH}")"
    git -C "${PYTORCH3D_CLONE_DIR}" fetch origin
    git -C "${PYTORCH3D_CLONE_DIR}" checkout "${PYTORCH3D_COMMIT}"
else
    git -C "${PYTORCH3D_CLONE_DIR}" fetch --depth 1 origin main
    git -C "${PYTORCH3D_CLONE_DIR}" reset --hard origin/main
    PYTORCH3D_COMMIT="$(git -C "${PYTORCH3D_CLONE_DIR}" rev-parse HEAD)"
fi

printf '%s\n' "${PYTORCH3D_COMMIT}" > "${SIDECAR_PATH}"
echo "PyTorch3D commit: ${PYTORCH3D_COMMIT}"

CUDA_PREFIX="$(env_prefix)"
if [ -z "${CUDA_PREFIX}" ]; then
    echo "Could not resolve conda prefix for environment ${ENV_NAME}" >&2
    exit 1
fi

MAX_JOBS_VALUE="$(python - <<'PY'
import os
cpu = os.cpu_count() or 8
print(min(8, cpu))
PY
)"

run_in_env env \
    CUDA_HOME="${CUDA_PREFIX}" \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    MAX_JOBS="${MAX_JOBS_VALUE}" \
    python -m pip install --no-build-isolation --no-deps --force-reinstall "${PYTORCH3D_CLONE_DIR}"

require_files \
    dataset/PXB184/data_stats.pth \
    checkpoints/diffusion/c1_face/model000155000.pt \
    checkpoints/diffusion/c1_pose/model000340000.pt \
    checkpoints/guide/c1_pose/checkpoints/iter-0100000.pt \
    checkpoints/ca_body/data/PXB184/body_dec.ckpt \
    checkpoints/ca_body/data/PXB184/config.yml \
    assets/wav2vec_large.pt \
    assets/vq-wav2vec.pt \
    assets/iter-0200000.pt

run_in_env python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
run_in_env python -c "import pytorch3d"
run_in_env python - <<'PY'
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.textures import TexturesVertex

device = torch.device("cuda:0")
verts = torch.tensor(
    [[[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]],
    device=device,
    dtype=torch.float32,
)
faces = torch.tensor([[[0, 1, 2]]], device=device, dtype=torch.int64)
verts_rgb = torch.ones_like(verts)
mesh = Meshes(verts=verts, faces=faces, textures=TexturesVertex(verts_features=verts_rgb))
cameras = FoVPerspectiveCameras(device=device)
raster_settings = RasterizationSettings(image_size=16)
rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
fragments = rasterizer(mesh)
assert fragments.zbuf.is_cuda
print("pytorch3d-gpu-smoke-ok", tuple(fragments.zbuf.shape))
PY
run_in_env python -c "import fairseq, gradio, torchaudio, torchvision"
run_in_env ffmpeg -version

echo "Blackwell environment is ready."
echo "Launch with: conda run -n ${ENV_NAME} python -m demo.demo"
