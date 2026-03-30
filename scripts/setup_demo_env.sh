#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-a2p-demo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

run_in_env() {
    conda run --no-capture-output -n "${ENV_NAME}" "$@"
}

pip_deps_ready() {
    run_in_env python -c "
import importlib.util
required = [
    'attrdict',
    'colorama',
    'einops',
    'fairseq',
    'gradio',
    'gradio_client',
    'huggingface_hub',
    'hydra',
    'mediapy',
    'omegaconf',
    'cv2',
    'tensorboard',
    'tensorboardX',
    'tqdm',
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
"
}

env_exists() {
    conda env list | awk 'NR > 2 {print $1}' | grep -Fxq "${ENV_NAME}"
}

download_file() {
    local url="$1"
    local output_path="$2"

    if [ -f "${output_path}" ]; then
        echo "Removing stale download: ${output_path}"
        rm -f "${output_path}"
    fi

    if command -v wget >/dev/null 2>&1; then
        wget --tries=3 --waitretry=5 -O "${output_path}" "${url}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --retry-delay 5 "${url}" -o "${output_path}"
    else
        echo "Neither wget nor curl is available on PATH." >&2
        return 1
    fi
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

if env_exists; then
    echo "Updating conda environment: ${ENV_NAME}"
    conda env update -n "${ENV_NAME}" -f environment.demo.yml --prune
else
    echo "Creating conda environment: ${ENV_NAME}"
    conda env create -n "${ENV_NAME}" -f environment.demo.yml
fi

run_in_env python -m pip install --upgrade "pip<24" "setuptools<70" "wheel" "Cython<3"
if pip_deps_ready; then
    echo "Pinned pip dependencies already exist."
else
    run_in_env python -m pip install -r requirements.demo.lock.txt
fi

if require_files assets/wav2vec_large.pt assets/vq-wav2vec.pt assets/iter-0200000.pt; then
    echo "Prerequisite audio assets already exist."
else
    bash scripts/download_prereq.sh
fi

if require_files \
    checkpoints/diffusion/c1_face/model000155000.pt \
    checkpoints/diffusion/c1_pose/model000340000.pt \
    checkpoints/guide/c1_pose/checkpoints/iter-0100000.pt; then
    echo "PXB184 motion checkpoints already exist."
else
    download_file \
        "http://audio2photoreal_models.berkeleyvision.org/PXB184_models.tar" \
        "PXB184_models.tar"
    tar xf PXB184_models.tar
    rm -f PXB184_models.tar
fi

if require_files checkpoints/ca_body/data/PXB184/body_dec.ckpt checkpoints/ca_body/data/PXB184/config.yml; then
    echo "PXB184 rendering checkpoints already exist."
else
    mkdir -p checkpoints/ca_body/data
    download_file \
        "https://github.com/facebookresearch/ca_body/releases/download/v0.0.1-alpha/PXB184.tar.gz" \
        "PXB184.tar.gz"
    tar xf PXB184.tar.gz --directory checkpoints/ca_body/data
    rm -f PXB184.tar.gz
fi

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

run_in_env python -c "import fairseq, gradio, pytorch3d, torch, torchaudio, torchvision"
run_in_env python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.is_available())"
run_in_env ffmpeg -version

echo "Demo environment is ready."
echo "Launch with: conda run -n ${ENV_NAME} python -m demo.demo"
