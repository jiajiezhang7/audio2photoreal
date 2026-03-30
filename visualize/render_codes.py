"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import glob
import os
import re
import subprocess
from collections import OrderedDict
from typing import Dict, List

import mediapy

import numpy as np
from scipy.io import wavfile

import utils.compat_collections  # noqa: F401
import torch
import torch as th
import torchaudio
from attrdict import AttrDict

from omegaconf import OmegaConf
from tqdm import tqdm
from utils.model_util import get_person_num
from visualize.ca_body.utils.image import linear2displayBatch
from visualize.ca_body.utils.torch import to_device
from visualize.ca_body.utils.train import load_checkpoint, load_from_config

ffmpeg_header = "ffmpeg -y "  # -hide_banner -loglevel error "


def _runtime_device(config) -> th.device:
    requested = os.getenv("A2P_DEVICE", "auto").strip().lower()
    if requested != "auto":
        return th.device(requested)
    if th.cuda.is_available():
        gpu = config.get("gpu", 0)
        capability = th.cuda.get_device_capability(gpu)
        arch = f"sm_{capability[0]}{capability[1]}"
        if arch in th.cuda.get_arch_list():
            return th.device(f"cuda:{gpu}")
        print(f"CUDA architecture {arch} is unsupported by this PyTorch build; using CPU.")
    return th.device("cpu")


def _render_stride(device: th.device) -> int:
    requested = os.getenv("A2P_RENDER_STRIDE")
    if requested is not None and requested.strip() != "":
        return max(1, int(requested))
    # CPU rendering is prohibitively slow for full-frame videos in this repo.
    # Keep enough temporal samples so motion remains visible in the exported mp4.
    return 20 if device.type == "cpu" else 1


def filter_params(params, ignore_names):
    return OrderedDict(
        [
            (k, v)
            for k, v in params.items()
            if not any([re.match(n, k) is not None for n in ignore_names])
        ]
    )


def call_ffmpeg(command: str) -> None:
    print(command, "-" * 100)
    e = subprocess.call(command, shell=True)
    if e != 0:
        assert False, e


def save_audio_file(path: str, audio: np.ndarray, sample_rate: int) -> None:
    tensor = torch.tensor(audio)
    try:
        torchaudio.save(path, tensor, sample_rate)
        return
    except RuntimeError:
        pass

    audio_np = tensor.detach().cpu().numpy()
    if np.issubdtype(audio_np.dtype, np.floating):
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_np = (audio_np.T * np.iinfo(np.int16).max).astype(np.int16)
    else:
        audio_np = audio_np.T
    wavfile.write(path, sample_rate, audio_np)


class BodyRenderer(th.nn.Module):
    def __init__(
        self,
        config_base: str,
        render_rgb: bool,
    ):
        super().__init__()
        self.config_base = config_base
        ckpt_path = f"{config_base}/body_dec.ckpt"
        config_path = f"{config_base}/config.yml"
        assets_path = f"{config_base}/static_assets.pt"
        # config
        config = OmegaConf.load(config_path)
        self.device = _runtime_device(config)
        self.render_stride = _render_stride(self.device)
        # assets
        static_assets = AttrDict(torch.load(assets_path, map_location="cpu"))
        # build model
        self.model = load_from_config(config.model, assets=static_assets).to(
            self.device
        )
        self.model.cal_enabled = False
        self.model.pixel_cal_enabled = False
        self.model.learn_blur_enabled = False
        self.render_rgb = render_rgb
        if not self.render_rgb:
            self.model.rendering_enabled = None
        # load model checkpoints
        print("loading...", ckpt_path)
        load_checkpoint(
            ckpt_path,
            modules={"model": self.model},
            ignore_names={"model": ["lbs_fn.*"]},
        )
        self.model.eval()
        self.model.to(self.device)
        # load default parameters for renderer
        person = get_person_num(config_path)
        self.default_inputs = to_device(
            th.load(f"assets/render_defaults_{person}.pth", map_location="cpu"),
            self.device,
        )

    def _write_video_stream(
        self, motion: np.ndarray, face: np.ndarray, save_name: str
    ) -> None:
        out = self._render_loop(motion, face)
        mediapy.write_video(save_name, out, fps=30)

    def _render_loop(self, body_pose: np.ndarray, face: np.ndarray) -> List[np.ndarray]:
        all_rgb = []
        default_inputs_copy = to_device(copy.deepcopy(self.default_inputs), self.device)
        frame_indices = list(range(0, len(body_pose), self.render_stride)) or [0]
        for idx, b in enumerate(tqdm(frame_indices)):
            B = default_inputs_copy["K"].shape[0]
            default_inputs_copy["lbs_motion"] = (
                th.tensor(body_pose[b : b + 1, :], device=self.device, dtype=th.float)
                .tile(B, 1)
                .to(self.device)
            )
            geom = (
                self.model.lbs_fn.lbs_fn(
                    default_inputs_copy["lbs_motion"],
                    self.model.lbs_fn.lbs_scale.unsqueeze(0).tile(B, 1),
                    self.model.lbs_fn.lbs_template_verts.unsqueeze(0).tile(B, 1, 1),
                )
                * self.model.lbs_fn.global_scaling
            )
            default_inputs_copy["geom"] = geom
            face_codes = (
                th.from_numpy(face).float().to(self.device)
                if not th.is_tensor(face)
                else face.to(self.device)
            )
            curr_face = th.tile(face_codes[b : b + 1, ...], (2, 1))
            default_inputs_copy["face_embs"] = curr_face
            preds = self.model(**default_inputs_copy)
            rgb0 = linear2displayBatch(preds["rgb"])[0]
            rgb1 = linear2displayBatch(preds["rgb"])[1]
            rgb = th.cat((rgb0, rgb1), axis=-1).permute(1, 2, 0)
            rgb = rgb.clip(0, 255).to(th.uint8)
            rgb_np = rgb.contiguous().detach().byte().cpu().numpy()
            next_idx = frame_indices[idx + 1] if idx + 1 < len(frame_indices) else len(body_pose)
            repeat = max(1, next_idx - b)
            all_rgb.extend([rgb_np] * repeat)
        return all_rgb[: len(body_pose)]

    def render_full_video(
        self,
        data_block: Dict[str, np.ndarray],
        animation_save_path: str,
        audio_sr: int = None,
        render_gt: bool = False,
    ) -> None:
        tag = os.path.basename(os.path.dirname(animation_save_path))
        save_name = os.path.splitext(os.path.basename(animation_save_path))[0]
        save_name = f"{tag}_{save_name}"
        save_audio_file(
            f"/tmp/audio_{save_name}.wav",
            data_block["audio"],
            audio_sr,
        )
        if render_gt:
            tag = "gt"
            self._write_video_stream(
                data_block["gt_body"],
                data_block["gt_face"],
                f"/tmp/{tag}_{save_name}.mp4",
            )
        else:
            tag = "pred"
            self._write_video_stream(
                data_block["body_motion"],
                data_block["face_motion"],
                f"/tmp/{tag}_{save_name}.mp4",
            )
        command = f"{ffmpeg_header} -i /tmp/{tag}_{save_name}.mp4 -i /tmp/audio_{save_name}.wav -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k -pix_fmt yuva420p {animation_save_path}_{tag}.mp4"
        call_ffmpeg(command)
        subprocess.call(
            f"rm /tmp/audio_{save_name}.wav && rm /tmp/{tag}_{save_name}.mp4",
            shell=True,
        )
