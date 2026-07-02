from __future__ import annotations

import argparse
import contextlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from datasets import create_dataloader

from models.configuration_bridge_wa import BridgeWAConfig
from models.configuration_world_teacher import WorldTeacherConfig
from models.lora_utils import mark_only_lora_trainable
from models.modeling_bridge_wa import BridgeWA
from models.modeling_world_teacher import WorldTeacher
from models.processing_vision_action import VisionActionProcessor
from scripts.train_world_teacher import build_tokenizer, denorm_views, maybe_resize


_CACHE_FILE_LRU_MAX = 16
_CACHE_FILE_LRU: "OrderedDict[str, object]" = OrderedDict()


def get_logger(name="train_bridge_wa", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


class _SimpleLogger:
    def __init__(self, logger):
        self.logger = logger

    def info(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.warning(msg, *args, **kwargs)


def _load_local_wan_checkpoint(
    model: WorldTeacher,
    ckpt_dir: Path,
    logger,
    ignore_shape_mismatch_prefixes: tuple[str, ...] = (),
) -> None:
    index_path = ckpt_dir / "model.safetensors.index.json"
    bin_index_path = ckpt_dir / "pytorch_model.bin.index.json"
    single_safe = ckpt_dir / "model.safetensors"
    single_bin = ckpt_dir / "pytorch_model.bin"
    if not (index_path.exists() or bin_index_path.exists() or single_safe.exists() or single_bin.exists()):
        raise FileNotFoundError(f"No WorldTeacher weights found under {ckpt_dir}")

    # Materialize the frozen Wan backbone from config.wan_model_id, then overlay
    # the released World Teacher checkpoint tensors.
    model._ensure_wan_modules()
    model_state = model.state_dict()
    ignore_shape_mismatch_prefixes = tuple(ignore_shape_mismatch_prefixes or ())

    def _filter_state(
        state_dict: Dict[str, torch.Tensor],
        *,
        allow_partial: bool,
    ) -> Dict[str, torch.Tensor]:
        filtered_state = {}
        ignored_shape = []
        unexpected = []
        for key, value in state_dict.items():
            target = model_state.get(key)
            if target is None:
                unexpected.append(key)
                if allow_partial:
                    continue
                filtered_state[key] = value
                continue
            if tuple(value.shape) != tuple(target.shape):
                if any(key.startswith(prefix) for prefix in ignore_shape_mismatch_prefixes):
                    ignored_shape.append((key, tuple(value.shape), tuple(target.shape)))
                    continue
            filtered_state[key] = value
        if unexpected and allow_partial:
            logger.warning(
                "World Teacher trainable-only checkpoint has %d unexpected keys (show 8): %s",
                len(unexpected),
                unexpected[:8],
            )
        if ignored_shape:
            preview = [
                f"{name}: ckpt{src_shape} -> model{dst_shape}"
                for name, src_shape, dst_shape in ignored_shape[:8]
            ]
            logger.warning(
                "Ignoring %d World Teacher keys with shape mismatch: %s",
                len(ignored_shape),
                "; ".join(preview),
            )
        return filtered_state

    def _apply_state(
        state_dict: Dict[str, torch.Tensor],
        *,
        log_incompatibility: bool = True,
        allow_partial: bool = False,
    ) -> int:
        state_dict = _filter_state(state_dict, allow_partial=allow_partial)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if log_incompatibility:
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            if missing and not allow_partial:
                logger.warning("World Teacher checkpoint missing keys (show 8): %s", missing[:8])
            if unexpected:
                logger.warning("World Teacher checkpoint unexpected keys (show 8): %s", unexpected[:8])
        return len(state_dict)

    def _read_index(path: Path) -> tuple[dict, dict]:
        with path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        metadata = index.get("metadata", {}) or {}
        weight_map = index.get("weight_map", {}) or {}
        if not weight_map:
            raise RuntimeError(f"Checkpoint index has no weight_map: {path}")
        return metadata, weight_map

    loaded_tensors = 0
    if index_path.exists():
        from safetensors.torch import load_file as safe_load

        _, weight_map = _read_index(index_path)
        shard_files = sorted(set(weight_map.values()))
        for shard in shard_files:
            shard_path = ckpt_dir / shard
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing World Teacher checkpoint shard: {shard_path}")
            state = safe_load(str(shard_path))
            loaded_tensors += _apply_state(state, log_incompatibility=False)
            del state
        logger.info("Loaded World Teacher safetensors checkpoint from %s (%d tensors).", ckpt_dir, loaded_tensors)
    elif bin_index_path.exists():
        metadata, weight_map = _read_index(bin_index_path)
        checkpoint_type = str(metadata.get("checkpoint_type", "") or "")
        is_trainable_only = checkpoint_type == "trainable_only"
        if is_trainable_only:
            logger.info(
                "Loading World Teacher trainable-only checkpoint overlay from %s; frozen/base weights come from wan_model_id.",
                ckpt_dir,
            )
        shard_files = sorted(set(weight_map.values()))
        for shard in shard_files:
            shard_path = ckpt_dir / shard
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing World Teacher checkpoint shard: {shard_path}")
            state = torch.load(str(shard_path), map_location="cpu")
            if not isinstance(state, dict):
                raise RuntimeError(f"Unexpected WorldTeacher checkpoint shard format in {shard_path}: {type(state)}")
            loaded_tensors += _apply_state(
                state,
                log_incompatibility=False,
                allow_partial=is_trainable_only,
            )
            del state
        expected_tensors = int(metadata.get("num_tensors", loaded_tensors) or loaded_tensors)
        if is_trainable_only and loaded_tensors != expected_tensors:
            logger.warning(
                "Loaded %d/%d trainable World Teacher tensors after filtering unexpected/mismatched keys.",
                loaded_tensors,
                expected_tensors,
            )
        logger.info(
            "Loaded World Teacher %scheckpoint from %s (%d tensors).",
            "trainable-only " if is_trainable_only else "",
            ckpt_dir,
            loaded_tensors,
        )
    elif single_safe.exists():
        from safetensors.torch import load_file as safe_load

        state = safe_load(str(single_safe))
        loaded_tensors = _apply_state(state)
        logger.info("Loaded World Teacher safetensors checkpoint from %s (%d tensors).", single_safe, loaded_tensors)
    else:
        state = torch.load(str(single_bin), map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError(f"Unexpected WorldTeacher checkpoint format: {type(state)}")
        loaded_tensors = _apply_state(state)
        logger.info("Loaded World Teacher torch checkpoint from %s (%d tensors).", single_bin, loaded_tensors)


def load_world_teacher(
    checkpoint: str,
    device: torch.device,
    logger,
    ignore_shape_mismatch_prefixes: tuple[str, ...] = (),
) -> WorldTeacher:
    ckpt_dir = Path(checkpoint)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"World Teacher checkpoint not found: {checkpoint}")
    cfg = WorldTeacherConfig.from_pretrained(checkpoint)
    model = WorldTeacher(cfg)
    logger.info("Bootstrapping World Teacher from wan_model_id=%s", cfg.wan_model_id)
    _load_local_wan_checkpoint(
        model,
        ckpt_dir,
        logger,
        ignore_shape_mismatch_prefixes=ignore_shape_mismatch_prefixes,
    )
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model


def maybe_get_wrist(image_input: torch.Tensor, image_mask: Optional[torch.Tensor], height: int, width: int):
    if image_input.shape[1] <= 1:
        return None, None
    wrist = image_input[:, 1]
    wrist = maybe_resize(wrist, height, width)
    wrist_valid_mask = None
    if image_mask is not None and image_mask.dim() >= 2 and image_mask.shape[1] > 1:
        wrist_valid_mask = image_mask[:, 1].bool()
        if bool(wrist_valid_mask.any()):
            wrist = wrist * wrist_valid_mask.view(-1, 1, 1, 1).to(dtype=wrist.dtype, device=wrist.device)
    return wrist, wrist_valid_mask


def _sample_key_to_cache_path(cache_dir: Path, sample_key: str) -> Path:
    safe_key = str(sample_key).replace(os.sep, "__").replace("/", "__")
    return cache_dir / f"{safe_key}.pt"


def _sample_key_to_episode_key(sample_key: str) -> str:
    key = str(sample_key)
    marker = "_frame"
    pos = key.rfind(marker)
    if pos <= 0:
        raise ValueError(f"Cannot derive episode key from sample_key: {sample_key}")
    return key[:pos]


def _episode_key_to_cache_path(cache_dir: Path, episode_key: str) -> Path:
    safe_key = str(episode_key).replace(os.sep, "__").replace("/", "__")
    return cache_dir / f"{safe_key}.pt"


def _sample_key_to_episode_cache_path(cache_dir: Path, sample_key: str) -> Path:
    return _episode_key_to_cache_path(cache_dir, _sample_key_to_episode_key(sample_key))


def _load_cache_file_cached(cache_path: Path):
    key = str(cache_path)
    payload = _CACHE_FILE_LRU.get(key)
    if payload is not None:
        _CACHE_FILE_LRU.move_to_end(key)
        return payload

    payload = torch.load(key, map_location="cpu")
    _CACHE_FILE_LRU[key] = payload
    if len(_CACHE_FILE_LRU) > _CACHE_FILE_LRU_MAX:
        _CACHE_FILE_LRU.popitem(last=False)
    return payload


def _load_cache_payload_for_sample(cache_dir: Path, sample_key: str):
    sample_key = str(sample_key)
    sample_cache_path = _sample_key_to_cache_path(cache_dir, sample_key)
    if sample_cache_path.exists():
        return _load_cache_file_cached(sample_cache_path), sample_cache_path

    episode_cache_path = _sample_key_to_episode_cache_path(cache_dir, sample_key)
    if not episode_cache_path.exists():
        raise FileNotFoundError(
            f"Missing teacher cache for sample {sample_key}: checked {sample_cache_path} and {episode_cache_path}"
        )
    payload = _load_cache_file_cached(episode_cache_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported episode cache payload type {type(payload)} in {episode_cache_path}")
    sample_to_index = payload.get("sample_to_index")
    if not isinstance(sample_to_index, dict):
        raise KeyError(f"sample_to_index missing in episode cache file {episode_cache_path}")
    if sample_key not in sample_to_index:
        raise KeyError(f"sample_key {sample_key} missing in episode cache file {episode_cache_path}")
    index = int(sample_to_index[sample_key])
    sample_payload = {"sample_key": sample_key}
    for field in ("future_latents", "future_latents_pooled", "change_map", "flow_map"):
        tensor = payload.get(field)
        if tensor is not None:
            sample_payload[field] = tensor[index]
    for field in ("teacher_future_steps", "world_teacher_path", "cache_version", "cache_granularity", "episode_key"):
        if field in payload:
            sample_payload[field] = payload[field]
    return sample_payload, episode_cache_path


def build_teacher_future_latents(
    world_teacher: WorldTeacher,
    world_teacher_tokenizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    teacher_future_steps: int,
    return_front_latents: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    image_input = denorm_views(batch["image_input"].to(device))
    image_mask = batch.get("image_mask")
    if image_mask is not None:
        image_mask = image_mask.to(device).bool()
    front = maybe_resize(image_input[:, 0], world_teacher.config.wan_height, world_teacher.config.wan_width)
    wrist, wrist_valid_mask = maybe_get_wrist(
        image_input=image_input,
        image_mask=image_mask,
        height=world_teacher.config.wan_height,
        width=world_teacher.config.wan_width,
    )
    lang = batch["language_instruction"]
    input_ids, attention_mask = world_teacher_tokenizer(lang, return_mask=True, add_special_tokens=True)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    front_latents_full = world_teacher._encode_image_latents(front)
    front_latents = front_latents_full[:, :, 0]
    text_context = world_teacher._encode_text(input_ids, attention_mask)
    wrist_tokens = None
    if wrist is not None:
        wrist_tokens = world_teacher._encode_wrist_tokens(wrist, wrist_valid_mask=wrist_valid_mask)
    context_img = world_teacher._build_context(
        text_context=text_context,
        wrist_tokens=wrist_tokens,
        front_latents=front_latents,
        future_latents=None,
        proprio=None,
    )
    future_latents = world_teacher.sample_future_latents(
        front_latents=front_latents,
        context=context_img,
        num_inference_steps=int(teacher_future_steps),
        reference_latents=front_latents_full,
    )
    if return_front_latents:
        return future_latents, front_latents
    return future_latents


def _decode_future_images_batch(world_teacher, future_latents: torch.Tensor) -> torch.Tensor:
    world_teacher._ensure_wan_modules()
    assert world_teacher.wan_vae is not None
    if future_latents.dim() == 4:
        future_latents = future_latents.unsqueeze(2)
    if future_latents.dim() != 5:
        raise ValueError(f"Unsupported future_latents shape: {tuple(future_latents.shape)}")

    target_device = next(world_teacher.wan_vae.parameters()).device
    target_dtype = next(world_teacher.wan_vae.parameters()).dtype
    hidden_states = [future_latents[i].to(device=target_device, dtype=target_dtype) for i in range(future_latents.shape[0])]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        videos = world_teacher.wan_vae.decode(
            hidden_states,
            device=target_device,
            tiled=world_teacher.config.wan_tiled,
            tile_size=world_teacher.config.wan_tile_size,
            tile_stride=world_teacher.config.wan_tile_stride,
        )

    images = []
    for video in videos:
        decoded = ((video.clamp(-1, 1) + 1) * 0.5).detach().float().cpu()
        if decoded.dim() == 4:
            if decoded.shape[0] in (1, 3):
                decoded = decoded[:, 0]
            elif decoded.shape[1] in (1, 3):
                decoded = decoded[0]
        if decoded.dim() != 3:
            raise ValueError(f"Unexpected decoded future image shape: {tuple(decoded.shape)}")
        images.append(decoded)
    return torch.stack(images, dim=0)


def _compute_optical_flow_map(front: torch.Tensor, future: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    front = front.detach().float().cpu()
    future = future.detach().float().cpu()
    front_np = (front.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    future_np = (future.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    front_gray = cv2.cvtColor(front_np, cv2.COLOR_RGB2GRAY)
    future_gray = cv2.cvtColor(future_np, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        front_gray,
        future_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    if flow.shape[:2] != target_hw:
        flow = cv2.resize(flow, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] /= max(front_gray.shape[1], 1)
    flow[..., 1] /= max(front_gray.shape[0], 1)
    flow = np.clip(flow, -1.0, 1.0)
    return torch.from_numpy(flow.transpose(2, 0, 1)).float()


def _compute_optical_flow_map_task(args) -> torch.Tensor:
    return _compute_optical_flow_map(*args)


def _compute_optical_flow_maps_batch(
    front_batch: torch.Tensor,
    future_images: torch.Tensor,
    target_hw: tuple[int, int],
    flow_workers: int = 0,
) -> torch.Tensor:
    tasks = [
        (front_batch[i].detach().cpu(), future_images[i], target_hw)
        for i in range(front_batch.shape[0])
    ]
    if not tasks:
        return torch.empty((0, 2, target_hw[0], target_hw[1]), dtype=torch.float32)

    if flow_workers <= 0:
        cpu_count = os.cpu_count() or 1
        flow_workers = min(len(tasks), max(1, min(cpu_count, 8)))
    if flow_workers <= 1:
        return torch.stack([_compute_optical_flow_map(front, future, hw) for front, future, hw in tasks], dim=0)

    prev_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=flow_workers) as executor:
            outputs = list(executor.map(_compute_optical_flow_map_task, tasks))
        return torch.stack(outputs, dim=0)
    finally:
        cv2.setNumThreads(prev_threads)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("BridgeWA training")
    parser.add_argument("--models", type=str, required=True, help="Base VisionAction or BridgeWA checkpoint")
    parser.add_argument("--world_teacher_path", type=str, default="", help="Frozen WorldTeacher image-only checkpoint")
    parser.add_argument("--output_dir", type=str, default="./models/train/BridgeWA_libero")
    parser.add_argument("--train_metas_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--future_lr_scale", type=float, default=2.0)
    parser.add_argument("--change_lr_scale", type=float, default=2.0)
    parser.add_argument("--flow_lr_scale", type=float, default=2.0)
    parser.add_argument("--action_lr_scale", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=50000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", choices=("constant", "linear", "cosine"))
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help=(
            "ckpt-* directory to resume from. New-format checkpoints must contain "
            "training_state/ for optimizer, LR, scaler, and RNG restoration."
        ),
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help=(
            "Load model weights from --resume_from_checkpoint but reset optimizer, LR schedule, "
            "RNG, and global_step. Use this for old checkpoints without training_state/."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--tokenizer_path", type=str, default="google/umt5-xxl")
    parser.add_argument("--num_actions", type=int, default=None)
    parser.add_argument("--action_mode", type=str, default=None)
    parser.add_argument("--future_token_count", type=int, default=4)
    parser.add_argument("--future_pool_hw", type=int, default=2)
    parser.add_argument("--future_token_dim", type=int, default=None)
    parser.add_argument("--change_token_count", type=int, default=16)
    parser.add_argument("--change_map_pool_hw", type=int, default=4)
    parser.add_argument("--change_token_dim", type=int, default=None)
    parser.add_argument("--flow_token_count", type=int, default=16)
    parser.add_argument("--flow_map_pool_hw", type=int, default=4)
    parser.add_argument("--flow_token_dim", type=int, default=None)
    parser.add_argument("--bridge_num_injected_layers", type=int, default=4)
    parser.add_argument("--bridge_injection_start_layer", type=int, default=None)
    parser.add_argument("--bridge_inject_future", action="store_true", default=None)
    parser.add_argument("--no_bridge_inject_future", action="store_false", dest="bridge_inject_future")
    parser.add_argument("--bridge_inject_change", action="store_true", default=None)
    parser.add_argument("--no_bridge_inject_change", action="store_false", dest="bridge_inject_change")
    parser.add_argument("--bridge_inject_flow", action="store_true", default=None)
    parser.add_argument("--no_bridge_inject_flow", action="store_false", dest="bridge_inject_flow")
    parser.add_argument("--bridge_future_num_injected_layers", type=int, default=None)
    parser.add_argument("--bridge_future_injection_start_layer", type=int, default=None)
    parser.add_argument("--bridge_change_num_injected_layers", type=int, default=None)
    parser.add_argument("--bridge_change_injection_start_layer", type=int, default=None)
    parser.add_argument("--bridge_flow_num_injected_layers", type=int, default=None)
    parser.add_argument("--bridge_flow_injection_start_layer", type=int, default=None)
    parser.add_argument("--bridge_use_future_gate", action="store_true", default=None)
    parser.add_argument("--no_bridge_use_future_gate", action="store_false", dest="bridge_use_future_gate")
    parser.add_argument("--bridge_use_change_gate", action="store_true", default=None)
    parser.add_argument("--no_bridge_use_change_gate", action="store_false", dest="bridge_use_change_gate")
    parser.add_argument("--bridge_use_flow_gate", action="store_true", default=None)
    parser.add_argument("--no_bridge_use_flow_gate", action="store_false", dest="bridge_use_flow_gate")
    parser.add_argument("--bridge_modulation_scale", type=float, default=0.1)
    parser.add_argument("--bridge_change_bias_scale", type=float, default=1.0)
    parser.add_argument("--bridge_flow_bias_scale", type=float, default=1.0)
    parser.add_argument("--bridge_lora_rank", type=int, default=None)
    parser.add_argument("--bridge_lora_alpha", type=float, default=None)
    parser.add_argument("--bridge_lora_dropout", type=float, default=None)
    parser.add_argument("--bridge_lora_last_n_blocks", type=int, default=None)
    parser.add_argument("--bridge_lora_target_modules", type=str, nargs="*", default=None)
    parser.add_argument("--bridge_lora_only_bridge_layers", action="store_true", default=None)
    parser.add_argument("--no_bridge_lora_only_bridge_layers", action="store_false", dest="bridge_lora_only_bridge_layers")
    parser.add_argument("--future_distill_weight", type=float, default=0.1)
    parser.add_argument("--future_distill_cosine_weight", type=float, default=0.1)
    parser.add_argument("--change_distill_weight", type=float, default=0.1)
    parser.add_argument("--change_distill_cosine_weight", type=float, default=0.1)
    parser.add_argument("--flow_distill_weight", type=float, default=0.1)
    parser.add_argument("--flow_distill_cosine_weight", type=float, default=0.1)
    parser.add_argument(
        "--bridge_guidance_source",
        type=str,
        default="predicted",
        choices=("predicted", "teacher", "blend"),
        help="Which world guidance tensors to inject into BridgeWA blocks during training.",
    )
    parser.add_argument(
        "--bridge_guidance_blend_ratio",
        type=float,
        default=0.5,
        help="Teacher guidance weight when --bridge_guidance_source=blend.",
    )
    parser.add_argument("--teacher_future_steps", type=int, default=50)
    parser.add_argument("--teacher_cache_dir", type=str, default="")
    parser.add_argument("--libero_abs_cache_dir", type=str, default="")
    parser.add_argument(
        "--vlabench_camera_order",
        type=str,
        default="",
        help="Optional comma-separated VLABench view order, e.g. image,second_image,wrist_image.",
    )
    parser.add_argument(
        "--dobot_camera_order",
        type=str,
        default="",
        help="Optional comma-separated Dobot view order, e.g. "
             "observation.images.third_person,observation.images.wrist.",
    )
    parser.add_argument(
        "--dobot_action_offset",
        type=int,
        default=None,
        help="Frame offset between Dobot observation index and action target start. "
             "0 aligns state/image t to command action t from run_control_lerobot21.py.",
    )
    parser.add_argument(
        "--dobot_ee6d_arm_slot",
        type=str,
        default="",
        choices=("", "first", "second", "left", "right", "active", "single", "0", "1"),
        help="Which 10D slot receives the Dobot single-arm ee6d state/action in 20D ee6d mode. "
             "Default is first, matching the VLABench single-arm convention.",
    )
    parser.add_argument("--flow_workers", type=int, default=0)
    parser.add_argument("--train_last_n_blocks", type=int, default=4)
    parser.add_argument("--freeze_vlm", action="store_true", default=True)
    parser.add_argument("--no_freeze_vlm", action="store_false", dest="freeze_vlm")
    parser.add_argument("--vision_action_domain_id", type=int, default=None)
    parser.add_argument(
        "--log_with",
        type=str,
        default="tensorboard",
        help="Accelerate tracker backend. Use 'none' to disable tracker initialization.",
    )
    parser.add_argument("--wandb_project", type=str, default="BridgeWA-Training")
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="")
    parser.add_argument("--wandb_dir", type=str, default="")
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Enable DDP unused-parameter detection for modality/layer ablations.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    return parser.parse_args()


def build_teacher_guidance_online(
    world_teacher,
    world_teacher_tokenizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    teacher_future_steps: int,
    flow_workers: int,
    future_pool_hw: int,
) -> Dict[str, torch.Tensor]:
    image_input = denorm_views(batch["image_input"].to(device))
    front = maybe_resize(image_input[:, 0], world_teacher.config.wan_height, world_teacher.config.wan_width)

    future_latents, front_latents = build_teacher_future_latents(
        world_teacher,
        world_teacher_tokenizer,
        batch,
        device,
        teacher_future_steps,
        return_front_latents=True,
    )
    if future_latents.dim() == 5:
        future_latents_4d = future_latents[:, :, 0]
    else:
        future_latents_4d = future_latents

    future_pool_hw = max(1, int(future_pool_hw))
    future_latents_pooled = F.adaptive_avg_pool2d(future_latents_4d, (future_pool_hw, future_pool_hw))
    delta = future_latents_4d.float() - front_latents.float()
    change_map = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    change_map = change_map / (change_map.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    future_images = _decode_future_images_batch(world_teacher, future_latents_4d)
    flow_map = _compute_optical_flow_maps_batch(
        front,
        future_images,
        future_latents_4d.shape[-2:],
        flow_workers=flow_workers,
    ).to(device)

    return {
        "future_latents": future_latents_pooled,
        "change_map": change_map,
        "flow_map": flow_map,
    }


def _pool_future_cache_tensor(tensor: torch.Tensor, future_pool_hw: int, cache_path: Path) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.dim() == 5:
        tensor = tensor[:, 0]
    if tensor.dim() != 3:
        raise ValueError(f"Expected cached future latent [C,H,W], got {tuple(tensor.shape)} in {cache_path}")
    source_hw = tensor.shape[-2:]
    future_pool_hw = max(1, int(future_pool_hw))
    if min(source_hw) < future_pool_hw:
        raise ValueError(
            f"Cached future_latents_pooled in {cache_path} has spatial size {source_hw}, "
            f"but --future_pool_hw={future_pool_hw}. Recompute the cache with raw future_latents "
            "or disable --teacher_cache_dir for this size ablation."
        )
    return F.adaptive_avg_pool2d(tensor.unsqueeze(0), (future_pool_hw, future_pool_hw)).squeeze(0)


def load_teacher_guidance_from_cache(
    batch: Dict[str, torch.Tensor],
    cache_dir: Path,
    device: torch.device,
    future_pool_hw: int,
) -> Dict[str, torch.Tensor]:
    sample_keys = batch.get("sample_key")
    if sample_keys is None:
        raise KeyError("sample_key is required in batch when using --teacher_cache_dir.")

    future_latents = []
    change_maps = []
    flow_maps = []
    for sample_key in sample_keys:
        payload, cache_path = _load_cache_payload_for_sample(cache_dir, str(sample_key))
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported cache payload type {type(payload)} in {cache_path}")
        future_tensor = payload.get("future_latents")
        if future_tensor is None:
            future_tensor = payload.get("future_latents_pooled")
        change_tensor = payload.get("change_map")
        flow_tensor = payload.get("flow_map")
        if future_tensor is None:
            raise KeyError(f"future_latents or future_latents_pooled missing in cache file {cache_path}")
        if change_tensor is None:
            raise KeyError(f"change_map missing in cache file {cache_path}")
        if flow_tensor is None:
            raise KeyError(f"flow_map missing in cache file {cache_path}")
        future_latents.append(_pool_future_cache_tensor(future_tensor, future_pool_hw, cache_path))
        change_maps.append(change_tensor.float())
        flow_maps.append(flow_tensor.float())
    return {
        "future_latents": torch.stack(future_latents, dim=0).to(device),
        "change_map": torch.stack(change_maps, dim=0).to(device),
        "flow_map": torch.stack(flow_maps, dim=0).to(device),
    }


def build_optimizer(model: BridgeWA, args) -> AdamW:
    future_prefixes = ("future_token_projector.", "future_token_predictor.")
    change_prefixes = ("change_map_projector.", "change_map_predictor.")
    flow_prefixes = ("flow_map_projector.", "flow_map_predictor.")
    action_prefixes = ("transformer.action_decoder.", "transformer.action_encoder.", "transformer.soft_prompt_hub.")
    groups = {
        "base": {"params": [], "lr": args.learning_rate},
        "future": {"params": [], "lr": args.learning_rate * args.future_lr_scale},
        "change": {"params": [], "lr": args.learning_rate * args.change_lr_scale},
        "flow": {"params": [], "lr": args.learning_rate * args.flow_lr_scale},
        "action": {"params": [], "lr": args.learning_rate * args.action_lr_scale},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(future_prefixes):
            bucket = "future"
        elif name.startswith(change_prefixes):
            bucket = "change"
        elif name.startswith(flow_prefixes):
            bucket = "flow"
        elif name.startswith(action_prefixes):
            bucket = "action"
        else:
            bucket = "base"
        groups[bucket]["params"].append(param)
    param_groups = [
        {"name": name, "params": group["params"], "lr": group["lr"], "weight_decay": args.weight_decay}
        for name, group in groups.items()
        if group["params"]
    ]
    return AdamW(param_groups, betas=tuple(args.betas))


def update_group_lrs(optim: torch.optim.Optimizer, step: int, args):
    total_steps = max(1, int(args.iters))
    warmup_steps = max(0, min(int(args.warmup_steps), total_steps))
    current_step = min(step + 1, total_steps)

    if warmup_steps > 0 and current_step <= warmup_steps:
        scale = float(current_step) / float(max(1, warmup_steps))
    else:
        sched = str(getattr(args, "lr_scheduler_type", "constant")).lower()
        min_lr_ratio = float(max(0.0, min(1.0, getattr(args, "min_lr_ratio", 0.0))))
        if sched == "constant" or total_steps <= warmup_steps:
            scale = 1.0
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = max(0.0, min(1.0, progress))
            if sched == "linear":
                decay = 1.0 - progress
            elif sched == "cosine":
                decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
            else:
                raise ValueError(f"Unsupported lr scheduler type: {sched}")
            scale = max(min_lr_ratio, decay)

    for group in optim.param_groups:
        base_lr = group.get("initial_lr", group["lr"])
        group["lr"] = base_lr * scale


def _build_bridge_wa_overrides(args) -> dict:
    overrides = {
        "future_token_count": args.future_token_count,
        "future_token_pool_hw": args.future_pool_hw,
        "future_token_dim": args.future_token_dim,
        "change_token_count": args.change_token_count,
        "change_map_pool_hw": args.change_map_pool_hw,
        "change_token_dim": args.change_token_dim,
        "flow_token_count": args.flow_token_count,
        "flow_map_pool_hw": args.flow_map_pool_hw,
        "flow_token_dim": args.flow_token_dim,
        "bridge_num_injected_layers": args.bridge_num_injected_layers,
        "bridge_injection_start_layer": args.bridge_injection_start_layer,
        "bridge_modulation_scale": args.bridge_modulation_scale,
        "bridge_change_bias_scale": args.bridge_change_bias_scale,
        "bridge_flow_bias_scale": args.bridge_flow_bias_scale,
        "future_distill_weight": args.future_distill_weight,
        "future_distill_cosine_weight": args.future_distill_cosine_weight,
        "change_distill_weight": args.change_distill_weight,
        "change_distill_cosine_weight": args.change_distill_cosine_weight,
        "flow_distill_weight": args.flow_distill_weight,
        "flow_distill_cosine_weight": args.flow_distill_cosine_weight,
    }
    optional_overrides = {
        "num_actions": args.num_actions,
        "action_mode": args.action_mode,
        "bridge_inject_future": args.bridge_inject_future,
        "bridge_inject_change": args.bridge_inject_change,
        "bridge_inject_flow": args.bridge_inject_flow,
        "bridge_future_num_injected_layers": args.bridge_future_num_injected_layers,
        "bridge_future_injection_start_layer": args.bridge_future_injection_start_layer,
        "bridge_change_num_injected_layers": args.bridge_change_num_injected_layers,
        "bridge_change_injection_start_layer": args.bridge_change_injection_start_layer,
        "bridge_flow_num_injected_layers": args.bridge_flow_num_injected_layers,
        "bridge_flow_injection_start_layer": args.bridge_flow_injection_start_layer,
        "bridge_use_future_gate": args.bridge_use_future_gate,
        "bridge_use_change_gate": args.bridge_use_change_gate,
        "bridge_use_flow_gate": args.bridge_use_flow_gate,
        "bridge_lora_rank": args.bridge_lora_rank,
        "bridge_lora_alpha": args.bridge_lora_alpha,
        "bridge_lora_dropout": args.bridge_lora_dropout,
        "bridge_lora_last_n_blocks": args.bridge_lora_last_n_blocks,
        "bridge_lora_target_modules": args.bridge_lora_target_modules,
        "bridge_lora_only_bridge_layers": args.bridge_lora_only_bridge_layers,
    }
    for key, value in optional_overrides.items():
        if value is not None:
            overrides[key] = value
    return overrides


def freeze_for_bridge_wa_training(model: BridgeWA, args, logger):
    if args.freeze_vlm:
        model.vlm.requires_grad_(False)
        logger.info("Freezing VisionAction Florence backbone.")
    total_blocks = len(model.transformer.blocks)
    train_last_n = max(0, min(total_blocks, int(args.train_last_n_blocks)))
    lora_rank = int(getattr(model.config, "bridge_lora_rank", 0) or 0)
    if lora_rank > 0:
        trainable_lora_modules = 0
        for block in model.transformer.blocks:
            block.requires_grad_(False)
            trainable_lora_modules += mark_only_lora_trainable(block)
        lora_last_n = int(getattr(model.config, "bridge_lora_last_n_blocks", 0) or 0)
        if lora_last_n <= 0 or lora_last_n > total_blocks:
            lora_last_n = total_blocks
        logger.info(
            "LoRA stage2 enabled: rank=%d last_n=%d only_bridge_layers=%s trainable_lora_modules=%d. "
            "Transformer base weights remain frozen.",
            lora_rank,
            lora_last_n,
            bool(getattr(model.config, "bridge_lora_only_bridge_layers", False)),
            trainable_lora_modules,
        )
        return
    freeze_until = max(0, total_blocks - train_last_n)
    for idx, block in enumerate(model.transformer.blocks):
        block.requires_grad_(idx >= freeze_until)
    logger.info("Training only the last %d transformer blocks (total=%d).", train_last_n, total_blocks)


def _resolve_resume_checkpoint(path_value: str) -> Path | None:
    if not path_value:
        return None
    checkpoint_dir = Path(path_value)
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"--resume_from_checkpoint does not exist or is not a directory: {checkpoint_dir}")
    return checkpoint_dir


def _infer_checkpoint_step(checkpoint_dir: Path | None) -> int:
    if checkpoint_dir is None:
        return 0
    state_path = checkpoint_dir / "state.json"
    if state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as f:
                return int(json.load(f).get("global_step", 0))
        except Exception:
            pass
    if checkpoint_dir.name.startswith("ckpt-"):
        try:
            return int(checkpoint_dir.name.split("ckpt-", 1)[1])
        except ValueError:
            pass
    return 0


def _training_state_dir(checkpoint_dir: Path | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    state_dir = checkpoint_dir / "training_state"
    if state_dir.exists() and state_dir.is_dir():
        return state_dir
    return None


def _optimizer_group_snapshot(optim: torch.optim.Optimizer) -> list[dict]:
    groups = []
    for group in optim.param_groups:
        groups.append(
            {
                key: value
                for key, value in group.items()
                if key != "params" and isinstance(value, (str, int, float, bool, type(None), list, tuple))
            }
        )
    return groups


def load_bridge_wa_model(args, logger, model_path: str | None = None) -> BridgeWA:
    source = model_path or args.models
    config_path = Path(source) / "config.json"
    overrides = _build_bridge_wa_overrides(args)
    is_bridge_wa_ckpt = False
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg_json = json.load(f)
            is_bridge_wa_ckpt = cfg_json.get("model_type") == "bridge_wa"
        except Exception:
            is_bridge_wa_ckpt = False

    if is_bridge_wa_ckpt:
        logger.info("Loading existing BridgeWA checkpoint from %s", source)
        cfg = BridgeWAConfig.from_pretrained(source)
        for key, value in overrides.items():
            setattr(cfg, key, value)
        model = BridgeWA.from_pretrained(source, config=cfg, ignore_mismatched_sizes=True)
    else:
        logger.info("Initializing BridgeWA from base VisionAction checkpoint %s", source)
        model = BridgeWA.from_vision_action_pretrained(source, **overrides)
    if args.vision_action_domain_id is not None:
        model.config.vision_action_domain_id = int(args.vision_action_domain_id)
    return model


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_with = None if str(args.log_with).lower() in {"", "none", "false", "0"} else args.log_with
    ddp_kwargs = []
    if args.ddp_find_unused_parameters:
        ddp_kwargs.append(DistributedDataParallelKwargs(find_unused_parameters=True))
    accelerator = Accelerator(
        log_with=log_with,
        project_dir=output_dir,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=ddp_kwargs,
        gradient_accumulation_steps=max(1, int(args.gradient_accumulation_steps)),
    )
    if log_with is not None:
        tracker_project = "BridgeWA-Training"
        tracker_kwargs = {}
        if str(log_with).lower() == "wandb":
            tracker_project = args.wandb_project or tracker_project
            wandb_kwargs = {}
            if args.wandb_name:
                wandb_kwargs["name"] = args.wandb_name
            if args.wandb_group:
                wandb_kwargs["group"] = args.wandb_group
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            if args.wandb_mode:
                wandb_kwargs["mode"] = args.wandb_mode
            if args.wandb_dir:
                wandb_kwargs["dir"] = args.wandb_dir
            if wandb_kwargs:
                tracker_kwargs["wandb"] = wandb_kwargs
        accelerator.init_trackers(tracker_project, config=vars(args), init_kwargs=tracker_kwargs)
    accelerator.wait_for_everyone()
    logger = get_logger(name="train_bridge_wa", output_dir=output_dir, accelerator=accelerator)
    set_seed(args.seed + accelerator.process_index)
    logger.info("Args: %s", args)
    resume_checkpoint_dir = _resolve_resume_checkpoint(args.resume_from_checkpoint)
    resume_step = _infer_checkpoint_step(resume_checkpoint_dir)
    resume_training_state_dir = _training_state_dir(resume_checkpoint_dir)
    if resume_checkpoint_dir is not None:
        logger.info(
            "Resume source: %s | step=%d | weights_only=%s | training_state=%s",
            str(resume_checkpoint_dir),
            resume_step,
            str(args.resume_weights_only),
            str(resume_training_state_dir) if resume_training_state_dir is not None else "",
        )
        if not args.resume_weights_only and resume_training_state_dir is None:
            raise FileNotFoundError(
                f"{resume_checkpoint_dir} does not contain training_state/. "
                "Cannot fully restore optimizer/LR/RNG state from this checkpoint. "
                "Use --resume_weights_only to intentionally reset training state."
            )
    logger.info(
        "World guidance injection mode: source=%s blend_ratio=%.3f",
        args.bridge_guidance_source,
        float(args.bridge_guidance_blend_ratio),
    )
    if args.bridge_lora_rank is not None and int(args.bridge_lora_rank) > 0:
        logger.info(
            "BridgeWA LoRA requested: rank=%s alpha=%s dropout=%s last_n=%s only_bridge_layers=%s",
            args.bridge_lora_rank,
            args.bridge_lora_alpha,
            args.bridge_lora_dropout,
            args.bridge_lora_last_n_blocks,
            args.bridge_lora_only_bridge_layers,
        )

    model_source = str(resume_checkpoint_dir) if resume_checkpoint_dir is not None else args.models
    model = load_bridge_wa_model(args, logger, model_path=model_source)
    logger.info(
        "BridgeWA injected layers: future=%s change=%s flow=%s union=%s",
        sorted(getattr(model.transformer, "future_injected_layers", set())),
        sorted(getattr(model.transformer, "change_injected_layers", set())),
        sorted(getattr(model.transformer, "flow_injected_layers", set())),
        sorted(getattr(model.transformer, "bridge_injected_layers", set())),
    )
    logger.info(
        "BridgeWA target sizes: future=%dx%d/%d dim=%s, change=%dx%d/%d dim=%s, flow=%dx%d/%d dim=%s",
        int(model.config.future_token_pool_hw),
        int(model.config.future_token_pool_hw),
        int(model.config.future_token_count),
        getattr(model.config, "future_token_dim", None) or "hidden",
        int(model.config.change_map_pool_hw),
        int(model.config.change_map_pool_hw),
        int(model.config.change_token_count),
        getattr(model.config, "change_token_dim", None) or "hidden",
        int(model.config.flow_map_pool_hw),
        int(model.config.flow_map_pool_hw),
        int(model.config.flow_token_count),
        getattr(model.config, "flow_token_dim", None) or "hidden",
    )
    freeze_for_bridge_wa_training(model, args, logger)
    processor_source = model_source if (Path(model_source) / "preprocessor_config.json").exists() else args.models
    processor = VisionActionProcessor.from_pretrained(processor_source)
    world_teacher_tokenizer = None
    world_teacher = None
    teacher_cache_dir = Path(args.teacher_cache_dir) if args.teacher_cache_dir else None
    if teacher_cache_dir is not None:
        teacher_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using offline combined cache from %s", str(teacher_cache_dir))
    else:
        if not args.world_teacher_path:
            raise ValueError("--world_teacher_path is required when --teacher_cache_dir is not set.")
        world_teacher_tokenizer = build_tokenizer(args.tokenizer_path)
        world_teacher = load_world_teacher(args.world_teacher_path, accelerator.device, _SimpleLogger(logger))

    handler_kwargs = {}
    if args.libero_abs_cache_dir:
        handler_kwargs["libero_abs_cache_dir"] = args.libero_abs_cache_dir
    if args.vlabench_camera_order:
        handler_kwargs["vlabench_camera_order"] = args.vlabench_camera_order
    if args.dobot_camera_order:
        handler_kwargs["dobot_camera_order"] = args.dobot_camera_order
    if args.dobot_action_offset is not None:
        handler_kwargs["dobot_action_offset"] = int(args.dobot_action_offset)
    if args.dobot_ee6d_arm_slot:
        handler_kwargs["dobot_ee6d_arm_slot"] = args.dobot_ee6d_arm_slot

    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=True,
        return_future=False,
        handler_kwargs=handler_kwargs or None,
    )

    optim = build_optimizer(model, args)
    for group in optim.param_groups:
        group["initial_lr"] = group["lr"]
    model, optim = accelerator.prepare(model, optim)
    model_config = accelerator.unwrap_model(model).config
    future_cache_pool_hw = int(model_config.future_token_pool_hw)

    model.train()
    if resume_training_state_dir is not None and not args.resume_weights_only:
        accelerator.load_state(str(resume_training_state_dir))
        global_step = resume_step
        logger.info("Loaded full accelerator training state from %s at global_step=%d", resume_training_state_dir, global_step)
    else:
        global_step = 0
        if resume_checkpoint_dir is not None and args.resume_weights_only:
            logger.info("Loaded checkpoint weights from %s and reset training state to global_step=0", resume_checkpoint_dir)
    t0 = time.time()
    logger.info("Start BridgeWA training for %d iterations", args.iters)

    def save_checkpoint(step: int):
        save_dir = output_dir / f"ckpt-{step}"
        training_state_dir = save_dir / "training_state"
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_dir.mkdir(parents=True, exist_ok=True)
            accelerator.print(f"Saving model to {save_dir}")
            accelerator.unwrap_model(model).save_pretrained(str(save_dir), safe_serialization=True)
            processor.save_pretrained(str(save_dir))
            state_payload = {
                "global_step": step,
                "complete_training_state": True,
                "training_state_dir": "training_state",
                "lr_scheduler_type": args.lr_scheduler_type,
                "warmup_steps": int(args.warmup_steps),
                "min_lr_ratio": float(args.min_lr_ratio),
                "iters": int(args.iters),
                "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
                "optimizer_param_groups": _optimizer_group_snapshot(optim),
            }
            with (save_dir / "state.json").open("w", encoding="utf-8") as f:
                json.dump(state_payload, f, indent=2)
        accelerator.wait_for_everyone()
        accelerator.save_state(str(training_state_dir), safe_serialization=True)
        accelerator.wait_for_everyone()

    for batch in train_dataloader:
        lang = {
            key: value.to(accelerator.device, non_blocking=True)
            for key, value in processor.encode_language(batch["language_instruction"]).items()
        }
        inputs = {**lang}
        for key in ("image_input", "image_mask", "domain_id", "proprio", "action"):
            inputs[key] = batch[key].to(accelerator.device, non_blocking=True)
        if teacher_cache_dir is not None:
            teacher_guidance = load_teacher_guidance_from_cache(
                batch,
                teacher_cache_dir,
                accelerator.device,
                future_pool_hw=future_cache_pool_hw,
            )
        else:
            teacher_guidance = build_teacher_guidance_online(
                world_teacher,
                world_teacher_tokenizer,
                batch,
                accelerator.device,
                args.teacher_future_steps,
                args.flow_workers,
                future_pool_hw=future_cache_pool_hw,
            )
        inputs.update(teacher_guidance)

        with accelerator.accumulate(model):
            update_group_lrs(optim, global_step, args)
            outputs = model(
                **inputs,
                guidance_source=args.bridge_guidance_source,
                guidance_blend_ratio=args.bridge_guidance_blend_ratio,
            )
            loss_terms = {}
            for key, value in outputs.items():
                if not isinstance(value, torch.Tensor) or value.ndim != 0:
                    continue
                if key in {
                    "future_token_mse",
                    "future_token_dir_loss",
                    "future_token_norm_loss",
                    "future_token_cosine",
                    "change_map_mse",
                    "change_map_cosine",
                    "flow_map_mse",
                    "flow_map_cosine",
                }:
                    continue
                loss_terms[key] = value
            loss = sum(loss_terms.values())

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if args.max_grad_norm:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                optim.zero_grad(set_to_none=True)

                if global_step % args.log_interval == 0:
                    logs = {k: float(v.detach().item()) for k, v in loss_terms.items()}
                    for name in (
                        "future_token_mse",
                        "future_token_dir_loss",
                        "future_token_norm_loss",
                        "future_token_cosine",
                        "change_map_mse",
                        "change_map_cosine",
                        "flow_map_mse",
                        "flow_map_cosine",
                    ):
                        if isinstance(outputs.get(name), torch.Tensor):
                            logs[name] = float(outputs[name].detach().item())
                    logs["loss_total"] = float(loss.detach().item())
                    for group in optim.param_groups:
                        logs[f"lr_{group['name']}"] = float(group["lr"])
                    if log_with is not None:
                        accelerator.log(logs, step=global_step)
                    if accelerator.is_main_process:
                        dt = (time.time() - t0) / max(1, args.log_interval)
                        t0 = time.time()
                        logger.info(
                            "[%d/%d] loss=%.4f future_mse=%.4f change_mse=%.4f flow_mse=%.4f (%.2fs/it)",
                            global_step,
                            args.iters,
                            logs["loss_total"],
                            logs.get("future_token_mse", 0.0),
                            logs.get("change_map_mse", 0.0),
                            logs.get("flow_map_mse", 0.0),
                            dt,
                        )

                global_step += 1
                if global_step == args.iters or global_step % args.save_interval == 0:
                    save_checkpoint(global_step)
        if global_step >= args.iters:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
