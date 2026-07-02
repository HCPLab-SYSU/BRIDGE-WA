# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import time
import json
import math
import random
import argparse
import inspect
import functools
import re
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from accelerate import Accelerator
try:
    from accelerate import DistributedDataParallelKwargs
except Exception:  # accelerate<0.23
    from accelerate.utils import DistributedDataParallelKwargs
try:
    from accelerate.utils import InitProcessGroupKwargs
except Exception:
    InitProcessGroupKwargs = None

from datasets import create_dataloader
from models.modeling_world_teacher import WorldTeacher

import logging
import sys
import psutil
from tqdm import tqdm


def get_logger(name="train", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
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


def get_args_parser():
    parser = argparse.ArgumentParser("WorldTeacher Training", add_help=False)
    parser.add_argument("--models", type=str, default="", help="Path or HF repo for pretrained WorldTeacher")
    parser.add_argument("--output_dir", type=str, default="runnings", help="Directory to save checkpoints")
    parser.add_argument("--train_metas_path", type=str, required=True, help="Path to training metadata (json or dir)")
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--wan_dit_lr_scale",
        type=float,
        default=1.0,
        help="LR scale applied to Wan DiT / VACE params relative to --learning_rate.",
    )
    parser.add_argument(
        "--action_head_lr_scale",
        type=float,
        default=1.0,
        help="LR scale applied to action-head / projection params relative to --learning_rate.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["constant", "linear", "cosine"],
        help="Learning rate schedule.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio used when --warmup_steps is 0.",
    )
    parser.add_argument("--warmup_steps", type=int, default=0, help="Absolute warmup steps.")
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.0,
        help="Floor ratio for LR decay, i.e. lr >= base_lr * min_lr_ratio.",
    )

    parser.add_argument("--iters", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=0, help="If >0, use epoch-based training (finite dataloader).")
    parser.add_argument("--steps_per_epoch", type=int, default=0, help="Optional steps per epoch for progress bar.")
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument(
        "--keep_checkpoint_interval",
        type=int,
        default=0,
        help=(
            "If >0, keep checkpoints whose step is divisible by this value and keep only the latest "
            "non-kept rolling checkpoint. The previous rolling checkpoint is deleted only after the "
            "next checkpoint finishes saving."
        ),
    )
    parser.add_argument(
        "--checkpoint_trainable_only",
        action="store_true",
        help="Save only trainable model tensors in checkpoints. Base/frozen weights must be loaded from --models.",
    )
    parser.add_argument(
        "--skip_optimizer_state",
        action="store_true",
        help="Do not save optimizer.pt. Scheduler and global_step are still saved for LR resume.",
    )
    parser.add_argument("--log_interval", type=int, default=20)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_from_wan_only", action="store_true", default=False)
    parser.add_argument("--wan_model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument(
        "--action_head",
        type=str,
        default=None,
        choices=["dit", "decoder", "separate", "independent", "vision_action"],
        help="Action head type. 'dit' shares Wan DiT; 'decoder' uses an independent transformer decoder.",
    )
    parser.add_argument(
        "--vision_action_model_id",
        type=str,
        default=None,
        help="When --action_head vision_action, load VisionAction backbone/action head from this checkpoint/repo.",
    )
    parser.add_argument(
        "--wan_backbone",
        type=str,
        default=None,
        choices=["auto", "ti2v", "vace", "vace_r2v", "r2v", "vace-r2v"],
        help="Wan backbone type. Use vace_r2v for Wan2.1 VACE Reference-to-Video.",
    )
    parser.add_argument(
        "--vace_scale",
        type=float,
        default=None,
        help="VACE control strength for R2V (default in config is 1.0).",
    )

    parser.add_argument("--tokenizer_path", type=str, default="google/umt5-xxl", help="Tokenizer name/path")
    parser.add_argument(
        "--data_proportions",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-meta sampling ratio list. Length must match train_metas_path entries.",
    )
    parser.add_argument("--action_mode", type=str, default=None, help="Action space name (e.g., ee6d, libero).")
    parser.add_argument("--sparse_t2v", action="store_true", default=False, help="Use T=2 with delta_t token.")
    parser.add_argument(
        "--joint_single_forward",
        action="store_true",
        default=False,
        help="DreamZero-style single Wan DiT pass for image+action diffusion (TI2V + DiT action head).",
    )
    parser.add_argument("--delta_t_scale", type=float, default=None, help="Scale factor for delta_t token.")
    parser.add_argument(
        "--use_future_token",
        type=int,
        choices=[0, 1],
        default=None,
        help="Whether to feed future-image token into action branch context (1=yes, 0=no).",
    )
    parser.add_argument(
        "--future_prompt_source",
        type=str,
        choices=["pred", "gt", "none"],
        default=None,
        help="Future prompt source for action branch: pred (model prediction), gt (teacher forcing), or none.",
    )
    parser.add_argument(
        "--scheduled_sampling",
        action="store_true",
        default=False,
        help="Enable scheduled sampling between GT and predicted future prompt for action branch.",
    )
    parser.add_argument(
        "--scheduled_sampling_strategy",
        type=str,
        default="linear",
        choices=["linear", "cosine", "exp"],
        help="Schedule strategy for GT prompt probability.",
    )
    parser.add_argument(
        "--scheduled_sampling_start",
        type=float,
        default=1.0,
        help="Initial GT probability for scheduled sampling.",
    )
    parser.add_argument(
        "--scheduled_sampling_end",
        type=float,
        default=0.0,
        help="Final GT probability for scheduled sampling.",
    )
    parser.add_argument(
        "--scheduled_sampling_warmup_steps",
        type=int,
        default=0,
        help="Keep GT probability at start value for the first N steps.",
    )
    parser.add_argument(
        "--scheduled_sampling_exp_k",
        type=float,
        default=5.0,
        help="Exponential decay factor for --scheduled_sampling_strategy exp.",
    )
    parser.add_argument("--future_index", type=int, default=None)
    parser.add_argument("--num_actions", type=int, default=None, help="Action horizon. Defaults to future_index when set.")
    parser.add_argument("--wan_height", type=int, default=None)
    parser.add_argument("--wan_width", type=int, default=None)
    parser.add_argument("--action_loss_weight", type=float, default=None)
    parser.add_argument("--action_supervised_weight", type=float, default=None)
    parser.add_argument("--image_loss_weight", type=float, default=None)
    parser.add_argument("--recon_loss_weight", type=float, default=None)
    parser.add_argument("--image_only_training", action="store_true", default=False)
    parser.add_argument("--return_future_image", action="store_true", default=False)
    parser.add_argument(
        "--vlabench_camera_order",
        type=str,
        default="",
        help="Optional comma-separated VLABench view order. For WorldTeacher keep wrist_image in slot 2, "
             "e.g. image,wrist_image,second_image or second_image,wrist_image,image.",
    )
    parser.add_argument(
        "--vlabench_future_view",
        type=str,
        default="",
        help="Optional VLABench view used for future_image supervision, e.g. image or second_image. "
             "Defaults to the first view from --vlabench_camera_order.",
    )
    parser.add_argument(
        "--dobot_camera_order",
        type=str,
        default="",
        help="Optional comma-separated Dobot view order, e.g. "
             "observation.images.third_person,observation.images.wrist.",
    )
    parser.add_argument(
        "--dobot_future_view",
        type=str,
        default="",
        help="Optional Dobot view used for future_image supervision. Defaults to the first Dobot view.",
    )
    parser.add_argument(
        "--dobot_action_offset",
        type=int,
        default=None,
        help="Frame offset between Dobot observation index and action target start. "
             "0 aligns state/image t to command action t from run_control_lerobot21.py.",
    )
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="WorldTeacher")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--deepspeed", action="store_true", default=False)
    parser.add_argument("--deepspeed_zero_stage", type=int, default=3, choices=[2, 3])
    parser.add_argument("--deepspeed_offload", action="store_true", default=False)
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Checkpoint dir to resume optimizer/scheduler/global_step from. "
             "If empty, auto-detect from --models when it points to ckpt-*.",
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        default=False,
        help="Resume only model weights/step; skip optimizer/scheduler state even if available.",
    )
    parser.add_argument(
        "--head_only_optim",
        action="store_true",
        default=False,
        help="Debug option: build optimizer before lazy Wan init. "
             "Default False means full-parameter optimization.",
    )
    parser.add_argument(
        "--freeze_wan_non_dit",
        action="store_true",
        default=False,
        help="Freeze Wan text encoder / VAE / image encoder (and vace branch); keep Wan DiT trainable.",
    )
    parser.add_argument(
        "--param_report_max_names",
        type=int,
        default=40,
        help="Max number of parameter names printed for trainable/frozen groups.",
    )
    parser.add_argument("--fsdp", action="store_true", default=False)
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="full_shard",
                        choices=["full_shard", "shard_grad_op", "no_shard", "hybrid_shard"])
    parser.add_argument("--fsdp_auto_wrap_policy", type=str, default="transformer",
                        choices=["transformer", "size", "none"])
    parser.add_argument("--fsdp_min_num_params", type=int, default=100000000)
    parser.add_argument("--fsdp_offload", action="store_true", default=False)
    parser.add_argument("--fsdp_use_orig_params", action="store_true", default=True)

    return parser


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def build_tokenizer(tokenizer_path: str):
    from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer
    return HuggingfaceTokenizer(name=tokenizer_path, seq_len=512, clean="whitespace")


def _format_numel(numel: int) -> str:
    if numel >= 1_000_000_000:
        return f"{numel / 1_000_000_000:.3f}B"
    if numel >= 1_000_000:
        return f"{numel / 1_000_000:.3f}M"
    if numel >= 1_000:
        return f"{numel / 1_000:.3f}K"
    return str(numel)


def _build_optimizer_param_groups(model: WorldTeacher, args, logger=None):
    base_lr = float(args.learning_rate)
    wan_scale = float(getattr(args, "wan_dit_lr_scale", 1.0))
    action_scale = float(getattr(args, "action_head_lr_scale", 1.0))

    wan_prefixes = ("wan_dit.", "wan_vace.")
    action_prefixes = (
        "action_decoder.",
        "action_to_latent.",
        "latent_to_action.",
        "front_proj.",
        "future_proj.",
        "proprio_proj.",
        "delta_t_proj.",
        "vision_action.",
        "vision_action_future_proj.",
    )

    grouped = {
        "base": {"params": [], "names": [], "lr": base_lr},
        "wan_dit": {"params": [], "names": [], "lr": base_lr * wan_scale},
        "action_head": {"params": [], "names": [], "lr": base_lr * action_scale},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(wan_prefixes):
            bucket = "wan_dit"
        elif name.startswith(action_prefixes):
            bucket = "action_head"
        else:
            bucket = "base"
        grouped[bucket]["params"].append(param)
        grouped[bucket]["names"].append(name)

    param_groups = []
    for bucket in ("base", "wan_dit", "action_head"):
        params = grouped[bucket]["params"]
        if not params:
            continue
        param_groups.append(
            {
                "params": params,
                "lr": grouped[bucket]["lr"],
                "weight_decay": float(args.weight_decay),
            }
        )

    if logger is not None:
        for bucket in ("base", "wan_dit", "action_head"):
            params = grouped[bucket]["params"]
            if not params:
                continue
            numel = sum(int(p.numel()) for p in params)
            sample_names = grouped[bucket]["names"][:6]
            logger.info(
                "Optimizer group %-11s | lr=%.3e | params=%s | sample=%s",
                bucket,
                grouped[bucket]["lr"],
                _format_numel(numel),
                sample_names,
            )
    return param_groups

def build_fsdp_plugin(args, logger=None):
    if not args.fsdp:
        return None
    try:
        try:
            from accelerate.utils import FSDPPlugin
        except Exception:
            from accelerate.utils import FullyShardedDataParallelPlugin as FSDPPlugin
        from torch.distributed.fsdp import ShardingStrategy, CPUOffload
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
    except Exception as exc:
        msg = f"FSDP not available: {exc}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None

    sharding_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    sharding = sharding_map.get(args.fsdp_sharding_strategy.lower(), ShardingStrategy.FULL_SHARD)

    auto_wrap_policy = None
    if args.fsdp_auto_wrap_policy == "transformer":
        try:
            from diffsynth.models.wan_video_dit import DiTBlock
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls={DiTBlock}
            )
        except Exception as exc:
            msg = f"FSDP auto-wrap policy fallback: {exc}"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            auto_wrap_policy = None
    elif args.fsdp_auto_wrap_policy == "size":
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=args.fsdp_min_num_params
        )

    cpu_offload = CPUOffload(offload_params=True) if args.fsdp_offload else None
    fsdp_kwargs = {
        "sharding_strategy": sharding,
        "auto_wrap_policy": auto_wrap_policy,
        "cpu_offload": cpu_offload,
        "use_orig_params": args.fsdp_use_orig_params,
    }
    sig = inspect.signature(FSDPPlugin)
    fsdp_kwargs = {k: v for k, v in fsdp_kwargs.items() if k in sig.parameters}
    return FSDPPlugin(**fsdp_kwargs)


def build_deepspeed_plugin(args, logger=None):
    if not args.deepspeed:
        return None
    try:
        from accelerate.utils import DeepSpeedPlugin
    except Exception as exc:
        msg = f"DeepSpeed not available: {exc}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None

    ds_kwargs = {
        "zero_stage": args.deepspeed_zero_stage,
    }
    if args.deepspeed_offload:
        ds_kwargs["offload_optimizer_device"] = "cpu"
        ds_kwargs["offload_param_device"] = "cpu"

    sig = inspect.signature(DeepSpeedPlugin)
    ds_kwargs = {k: v for k, v in ds_kwargs.items() if k in sig.parameters}
    plugin = DeepSpeedPlugin(**ds_kwargs)

    # accelerate==1.2.x keeps train_micro_batch_size_per_gpu in deepspeed_config (not ctor args).
    micro_bs = int(args.batch_size)
    plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = micro_bs
    if "gradient_accumulation_steps" in plugin.deepspeed_config and plugin.deepspeed_config["gradient_accumulation_steps"] == "auto":
        plugin.deepspeed_config["gradient_accumulation_steps"] = 1
    if hasattr(plugin, "hf_ds_config") and hasattr(plugin.hf_ds_config, "config"):
        plugin.hf_ds_config.config["train_micro_batch_size_per_gpu"] = micro_bs
        if (
            "gradient_accumulation_steps" in plugin.hf_ds_config.config
            and plugin.hf_ds_config.config["gradient_accumulation_steps"] == "auto"
        ):
            plugin.hf_ds_config.config["gradient_accumulation_steps"] = 1
    return plugin


def infer_total_train_steps(args, train_dataloader, use_epochs: bool, resume_step: int = 0) -> int:
    if not use_epochs:
        # args.iters means remaining steps when resuming.
        return max(1, int(args.iters + max(0, resume_step)))
    if args.steps_per_epoch and args.steps_per_epoch > 0:
        return max(1, int(args.steps_per_epoch * args.epochs + max(0, resume_step)))
    try:
        return max(1, int(len(train_dataloader) * args.epochs + max(0, resume_step)))
    except Exception:
        return 0


def build_lr_scheduler(optimizer, args, total_train_steps: int, start_step: int = 0):
    if args.lr_scheduler_type == "constant" or total_train_steps <= 0:
        return None
    warmup_steps = int(args.warmup_steps) if args.warmup_steps > 0 else int(total_train_steps * args.warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_train_steps))
    min_lr_ratio = float(max(0.0, min(1.0, args.min_lr_ratio)))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(min_lr_ratio, float(current_step) / float(max(1, warmup_steps)))
        progress = float(current_step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if args.lr_scheduler_type == "linear":
            decay = 1.0 - progress
        else:  # cosine
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, decay)

    if start_step > 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=start_step - 1)


def _scheduled_sampling_gt_prob(
    step: int,
    total_steps: int,
    start_prob: float,
    end_prob: float,
    strategy: str,
    warmup_steps: int,
    exp_k: float,
) -> float:
    start = float(max(0.0, min(1.0, start_prob)))
    end = float(max(0.0, min(1.0, end_prob)))
    warmup = max(0, int(warmup_steps))
    t_steps = max(1, int(total_steps))
    cur = max(0, int(step))
    if cur < warmup:
        return start

    denom = max(1, t_steps - warmup)
    progress = float(cur - warmup) / float(denom)
    progress = min(max(progress, 0.0), 1.0)

    if strategy == "cosine":
        mix = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end + (start - end) * mix
    if strategy == "exp":
        k = max(1e-8, float(exp_k))
        raw = math.exp(-k * progress)
        tail = math.exp(-k)
        norm = (raw - tail) / max(1e-8, (1.0 - tail))
        norm = min(max(norm, 0.0), 1.0)
        return end + (start - end) * norm
    # linear
    return start + (end - start) * progress

def sanitize_config(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if isinstance(v, (int, float, str, bool)) or torch.is_tensor(v):
            out[k] = v
        else:
            try:
                out[k] = json.dumps(v)
            except TypeError:
                out[k] = str(v)
    return out


def _param_group_name(param_name: str) -> str:
    if param_name.startswith("wan_dit."):
        return "wan_dit"
    if param_name.startswith("wan_text_encoder."):
        return "wan_text_encoder"
    if param_name.startswith("wan_vae."):
        return "wan_vae"
    if param_name.startswith("wan_image_encoder."):
        return "wan_image_encoder"
    if param_name.startswith("wan_vace."):
        return "wan_vace"
    return param_name.split(".", 1)[0]


def _format_param_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n / 1_000:.3f}K"
    return str(n)


def log_parameter_report(model: torch.nn.Module, logger, output_dir: Path, max_names: int = 40):
    trainable_names: List[str] = []
    frozen_names: List[str] = []
    total_params = 0
    trainable_params = 0
    group_stats: Dict[str, Dict[str, int]] = {}

    for name, p in model.named_parameters():
        n = int(p.numel())
        total_params += n
        grp = _param_group_name(name)
        if grp not in group_stats:
            group_stats[grp] = {"trainable": 0, "frozen": 0}
        if p.requires_grad:
            trainable_names.append(name)
            trainable_params += n
            group_stats[grp]["trainable"] += n
        else:
            frozen_names.append(name)
            group_stats[grp]["frozen"] += n

    frozen_params = total_params - trainable_params
    logger.info(
        "Param summary: total=%s | trainable=%s | frozen=%s",
        _format_param_count(total_params),
        _format_param_count(trainable_params),
        _format_param_count(frozen_params),
    )

    for grp in sorted(group_stats.keys()):
        t = group_stats[grp]["trainable"]
        f = group_stats[grp]["frozen"]
        logger.info(
            "Param group %-18s trainable=%8s | frozen=%8s",
            grp,
            _format_param_count(t),
            _format_param_count(f),
        )

    if trainable_names:
        logger.info("Trainable names (show %d/%d): %s", min(max_names, len(trainable_names)), len(trainable_names), trainable_names[:max_names])
    if frozen_names:
        logger.info("Frozen names (show %d/%d): %s", min(max_names, len(frozen_names)), len(frozen_names), frozen_names[:max_names])

    report = {
        "total_params": total_params,
        "total_params_human": _format_param_count(total_params),
        "trainable_params": trainable_params,
        "trainable_params_human": _format_param_count(trainable_params),
        "frozen_params": frozen_params,
        "frozen_params_human": _format_param_count(frozen_params),
        "group_stats": group_stats,
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "param_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning(f"Failed to write param_report.json: {exc}")


def _dataset_tag_from_name(name: str) -> str:
    n = str(name).rstrip("/")
    if not n:
        return "unknown"
    return os.path.basename(n) or n


def _target_dataset_tags_from_metas(metas_path: str) -> List[str]:
    tags: List[str] = []
    for part in str(metas_path).split(","):
        p = part.strip()
        if not p:
            continue
        p = p.rstrip("/")
        if p.endswith(".json"):
            # .../<dataset>/meta/info.json -> <dataset>
            parent = os.path.basename(os.path.dirname(p))
            if parent == "meta":
                tag = os.path.basename(os.path.dirname(os.path.dirname(p)))
            else:
                tag = os.path.basename(os.path.dirname(p))
        else:
            tag = os.path.basename(p) if os.path.basename(p) != "meta" else os.path.basename(os.path.dirname(p))
        if tag and tag not in tags:
            tags.append(tag)
        # Also accept root_path basename in LeRobot info.json, e.g.
        # train_metas_path tag "libero_overfit_ep0" while runtime dataset_tag is "libero".
        if p.endswith(".json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                root_tag = _dataset_tag_from_name(meta.get("root_path", ""))
                if root_tag and root_tag not in tags:
                    tags.append(root_tag)
            except Exception:
                pass
    return tags


def denorm_views(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    x = x * std + mean
    return x.clamp(0.0, 1.0)


def denorm_images(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x * std + mean
    return x.clamp(0.0, 1.0)


def maybe_resize(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if x.shape[-2:] == (height, width):
        return x
    return F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)


def _resolve_resume_dir(args) -> Path | None:
    if args.resume_from:
        p = Path(args.resume_from)
        return p if p.exists() and p.is_dir() else None
    # Compatibility: when --models points to ckpt-*, treat it as resume source.
    if args.models:
        p = Path(args.models)
        if p.exists() and p.is_dir():
            name = p.name
            if re.match(r"^ckpt-\d+$", name) or (p / "state.json").exists():
                return p
    return None


def _infer_resume_step(resume_dir: Path | None) -> int:
    if resume_dir is None:
        return 0
    m = re.search(r"ckpt-(\d+)$", resume_dir.name)
    if m:
        return int(m.group(1))
    state_path = resume_dir / "state.json"
    if state_path.exists():
        try:
            with state_path.open("r") as f:
                return int(json.load(f).get("global_step", 0))
        except Exception:
            return 0
    return 0


def _load_local_checkpoint_weights(model: WorldTeacher, ckpt_dir: Path, logger) -> bool:
    """
    Load local WorldTeacher checkpoint weights into a model that may use lazy Wan module init.
    Returns True if any local model weights were loaded.
    """
    if ckpt_dir is None:
        return False

    index_path = ckpt_dir / "model.safetensors.index.json"
    bin_index_path = ckpt_dir / "pytorch_model.bin.index.json"
    single_safe = ckpt_dir / "model.safetensors"
    single_bin = ckpt_dir / "pytorch_model.bin"
    has_weights = index_path.exists() or bin_index_path.exists() or single_safe.exists() or single_bin.exists()
    if not has_weights:
        return False

    # Ensure lazy Wan modules are instantiated before loading full local weights.
    try:
        model._ensure_wan_modules()
    except Exception as exc:
        logger.warning(f"Failed to initialize Wan modules before local checkpoint load: {exc}")
        return False

    def _load_state(state: Dict[str, torch.Tensor], log_incompatibility: bool = True) -> None:
        incompatible = model.load_state_dict(state, strict=False)
        if log_incompatibility:
            try:
                missing = getattr(incompatible, "missing_keys", [])
                unexpected = getattr(incompatible, "unexpected_keys", [])
                if unexpected:
                    logger.warning(f"Checkpoint load has {len(unexpected)} unexpected keys (show 8): {unexpected[:8]}")
                if missing:
                    logger.warning(f"Checkpoint load has {len(missing)} missing keys (show 8): {missing[:8]}")
            except Exception:
                pass

    loaded_tensors = 0
    if index_path.exists():
        try:
            from safetensors.torch import load_file as safe_load
        except Exception as exc:
            logger.warning(f"Cannot load sharded safetensors without safetensors package: {exc}")
            return False
        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        for shard in shard_files:
            shard_path = ckpt_dir / shard
            if not shard_path.exists():
                logger.warning(f"Missing shard file: {shard_path}")
                continue
            state = safe_load(str(shard_path))
            loaded_tensors += len(state)
            _load_state(state, log_incompatibility=False)
            del state
    elif bin_index_path.exists():
        with bin_index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        for shard in shard_files:
            shard_path = ckpt_dir / shard
            if not shard_path.exists():
                logger.warning(f"Missing shard file: {shard_path}")
                continue
            state = torch.load(str(shard_path), map_location="cpu")
            loaded_tensors += len(state) if isinstance(state, dict) else 0
            if isinstance(state, dict):
                _load_state(state, log_incompatibility=False)
            else:
                logger.warning(f"Unexpected checkpoint shard format in {shard_path}: {type(state)}")
            del state
    elif single_safe.exists():
        try:
            from safetensors.torch import load_file as safe_load
            state = safe_load(str(single_safe))
            loaded_tensors += len(state)
            _load_state(state)
        except Exception as exc:
            logger.warning(f"Failed to load safetensors checkpoint {single_safe}: {exc}")
            return False
    elif single_bin.exists():
        try:
            state = torch.load(str(single_bin), map_location="cpu")
            loaded_tensors += len(state) if isinstance(state, dict) else 0
            if isinstance(state, dict):
                _load_state(state)
            else:
                logger.warning(f"Unexpected checkpoint format in {single_bin}: {type(state)}")
                return False
        except Exception as exc:
            logger.warning(f"Failed to load torch checkpoint {single_bin}: {exc}")
            return False

    logger.info(f"Loaded local checkpoint weights from {ckpt_dir} (tensors loaded: {loaded_tensors}).")
    return loaded_tensors > 0


def main(args):
    output_dir = Path(args.output_dir)
    if args.fsdp and args.deepspeed:
        raise ValueError("FSDP and DeepSpeed are mutually exclusive. Please enable only one.")
    if args.deepspeed and int(args.deepspeed_zero_stage) == 3 and os.environ.get("WORLD_TEACHER_ALLOW_ZERO3", "0") != "1":
        print(
            "[warn] DeepSpeed ZeRO-3 is unstable for current WorldTeacher stack "
            "(may hit '_parameters._in_forward' hook error). Falling back to ZeRO-2. "
            "Set WORLD_TEACHER_ALLOW_ZERO3=1 to force stage-3."
        )
        args.deepspeed_zero_stage = 2
    log_with = ["tensorboard"]
    init_kwargs = {}
    tracker_project = "WorldTeacher-Training"
    if args.wandb:
        log_with.append("wandb")
        tracker_project = args.wandb_project
        wandb_kwargs = {}
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            wandb_kwargs["name"] = args.wandb_run_name
        init_kwargs = {"wandb": wandb_kwargs}
    fsdp_plugin = build_fsdp_plugin(args)
    deepspeed_plugin = build_deepspeed_plugin(args)
    kwargs_handlers = []
    if InitProcessGroupKwargs is not None:
        kwargs_handlers.append(InitProcessGroupKwargs(timeout=timedelta(hours=2)))
    if not args.fsdp and not args.deepspeed:
        try:
            kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))
        except Exception:
            pass
    accelerator = Accelerator(
        log_with=log_with,
        project_dir=output_dir,
        fsdp_plugin=fsdp_plugin,
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=kwargs_handlers if kwargs_handlers else None,
    )
    accelerator.init_trackers(
        tracker_project,
        config=sanitize_config(vars(args)),
        init_kwargs=init_kwargs,
    )

    # In single-process runs (especially with DeepSpeed plugin enabled),
    # torch.distributed may be unavailable/uninitialized.
    if accelerator.num_processes > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)

    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")
    resume_dir = _resolve_resume_dir(args)
    resume_step = _infer_resume_step(resume_dir)
    effective_resume_step = resume_step
    if resume_dir is not None:
        logger.info(
            "Resume source detected: %s | resume_step=%d | weights_only=%s",
            str(resume_dir),
            resume_step,
            str(args.resume_weights_only),
        )
    if args.resume_weights_only and resume_step > 0:
        logger.info(
            "resume_weights_only=True: loading model weights from %s but resetting optimizer/scheduler/global_step "
            "to 0 for a fresh training schedule.",
            str(resume_dir),
        )
        effective_resume_step = 0

    if args.init_from_wan_only:
        from models.configuration_world_teacher import WorldTeacherConfig
        cfg_kwargs = {"wan_model_id": args.wan_model_id}
        if args.action_head:
            cfg_kwargs["action_head_type"] = args.action_head
        if args.vision_action_model_id:
            cfg_kwargs["vision_action_model_id"] = args.vision_action_model_id
        if args.wan_backbone:
            cfg_kwargs["wan_backbone"] = args.wan_backbone
        if args.vace_scale is not None:
            cfg_kwargs["vace_scale"] = args.vace_scale
        if args.action_mode:
            cfg_kwargs["action_mode"] = args.action_mode
        if args.sparse_t2v:
            cfg_kwargs["sparse_t2v"] = True
        if args.joint_single_forward:
            cfg_kwargs["joint_single_forward"] = True
        if args.delta_t_scale is not None:
            cfg_kwargs["delta_t_scale"] = args.delta_t_scale
        if args.use_future_token is not None:
            cfg_kwargs["use_future_token"] = bool(args.use_future_token)
        if args.future_prompt_source is not None:
            cfg_kwargs["future_prompt_source"] = args.future_prompt_source
        cfg = WorldTeacherConfig(**cfg_kwargs)
        model = WorldTeacher(cfg)
    else:
        if not args.models:
            raise ValueError("--models is required unless --init_from_wan_only is set.")
        local_model_dir = Path(args.models)
        if local_model_dir.exists() and local_model_dir.is_dir() and (local_model_dir / "config.json").exists():
            # For local ckpt dirs, build from config and let local loader handle lazy Wan modules.
            from models.configuration_world_teacher import WorldTeacherConfig
            cfg = WorldTeacherConfig.from_pretrained(args.models)
            model = WorldTeacher(cfg)
        else:
            model = WorldTeacher.from_pretrained(args.models)
        if args.action_mode and args.action_mode != model.config.action_mode:
            if args.image_only_training:
                logger.warning(
                    "--action_mode %s overrides pretrained config %s in image-only mode. "
                    "This only affects dataloader/config behavior for world-teacher training; "
                    "the loaded action modules are kept as-is and should not be treated as an "
                    "action-compatible checkpoint.",
                    args.action_mode,
                    model.config.action_mode,
                )
                model.config.action_mode = args.action_mode
            else:
                logger.warning(
                    f"--action_mode {args.action_mode} ignored when loading pretrained model "
                    f"(current: {model.config.action_mode})."
                )
    if args.future_index is not None:
        model.config.future_index = args.future_index
    if args.sparse_t2v:
        model.config.sparse_t2v = True
    if args.joint_single_forward:
        model.config.joint_single_forward = True
    if args.delta_t_scale is not None:
        model.config.delta_t_scale = args.delta_t_scale
    if args.use_future_token is not None:
        model.config.use_future_token = bool(args.use_future_token)
    if args.future_prompt_source is not None:
        model.config.future_prompt_source = args.future_prompt_source
    if args.action_head:
        model.config.action_head_type = args.action_head
    if args.vision_action_model_id:
        model.config.vision_action_model_id = args.vision_action_model_id
    if args.wan_backbone:
        model.config.wan_backbone = args.wan_backbone
    if args.vace_scale is not None:
        model.config.vace_scale = args.vace_scale
    if args.num_actions is not None:
        model.config.num_actions = args.num_actions
    elif args.future_index is not None:
        model.config.num_actions = args.future_index
    if args.wan_height is not None:
        model.config.wan_height = args.wan_height
    if args.wan_width is not None:
        model.config.wan_width = args.wan_width
    if args.action_loss_weight is not None:
        model.config.action_loss_weight = args.action_loss_weight
    if args.action_supervised_weight is not None:
        model.config.action_supervised_weight = args.action_supervised_weight
    if args.image_loss_weight is not None:
        model.config.image_loss_weight = args.image_loss_weight
    if args.recon_loss_weight is not None:
        model.config.recon_loss_weight = args.recon_loss_weight
    model.config.image_only_training = bool(args.image_only_training)
    try:
        model._ensure_action_decoder()
    except Exception:
        pass
    if resume_dir is not None:
        loaded_local = _load_local_checkpoint_weights(model, resume_dir, logger)
        if not loaded_local:
            logger.info(f"No local full-model weights loaded from resume dir: {resume_dir}")
    if not args.head_only_optim:
        # Ensure lazy Wan modules are materialized before optimizer construction.
        logger.info("Initializing Wan backbone before optimizer construction.")
        model._ensure_wan_modules()
        if str(getattr(model.config, "action_head_type", "")).lower() == "vision_action":
            model._ensure_vision_action_modules()
        if args.freeze_wan_non_dit:
            logger.info("Applying freeze_wan_non_dit: freeze text/VAE/image/vace, train Wan DiT.")
            if getattr(model, "wan_text_encoder", None) is not None:
                model.wan_text_encoder.requires_grad_(False)
            if getattr(model, "wan_vae", None) is not None:
                model.wan_vae.requires_grad_(False)
            if getattr(model, "wan_image_encoder", None) is not None:
                model.wan_image_encoder.requires_grad_(False)
            if getattr(model, "wan_vace", None) is not None:
                model.wan_vace.requires_grad_(False)
            if getattr(model, "wan_dit", None) is not None:
                model.wan_dit.requires_grad_(True)
            model.config.freeze_text_encoder = True
            model.config.freeze_vae = True
            model.config.freeze_image_encoder = True
            model.config.freeze_dit = False
    else:
        logger.warning(
            "head_only_optim=True: optimizer may exclude Wan backbone params (debug mode)."
        )
    if accelerator.is_main_process:
        log_parameter_report(
            model,
            logger=logger,
            output_dir=output_dir,
            max_names=max(1, int(args.param_report_max_names)),
        )

    tokenizer = build_tokenizer(args.tokenizer_path)

    use_epochs = args.epochs is not None and args.epochs > 0
    if model.config.load_image_encoder:
        image_height, image_width = 224, 224
    else:
        image_height, image_width = model.config.wan_height, model.config.wan_width
    handler_kwargs = {}
    if args.vlabench_camera_order:
        handler_kwargs["vlabench_camera_order"] = args.vlabench_camera_order
    if args.vlabench_future_view:
        handler_kwargs["vlabench_future_view"] = args.vlabench_future_view
    if args.dobot_camera_order:
        handler_kwargs["dobot_camera_order"] = args.dobot_camera_order
    if args.dobot_future_view:
        handler_kwargs["dobot_future_view"] = args.dobot_future_view
    if args.dobot_action_offset is not None:
        handler_kwargs["dobot_action_offset"] = int(args.dobot_action_offset)
    if handler_kwargs:
        logger.info("Using dataset handler kwargs: %s", handler_kwargs)
    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.config.num_actions,
        action_mode=model.config.action_mode,
        data_proportions=args.data_proportions,
        training=True,
        future_index=model.config.future_index,
        return_future=args.return_future_image,
        infinite=not use_epochs,
        image_height=image_height,
        image_width=image_width,
        handler_kwargs=handler_kwargs or None,
    )

    param_groups = _build_optimizer_param_groups(model, args, logger=logger)
    optim = AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay, betas=tuple(args.betas))
    if resume_dir is not None and not args.resume_weights_only:
        opt_path = resume_dir / "optimizer.pt"
        if opt_path.exists():
            try:
                optim.load_state_dict(torch.load(opt_path, map_location="cpu"))
                logger.info(f"Loaded optimizer state from {opt_path}")
            except Exception as exc:
                logger.warning(f"Failed to load optimizer state from {opt_path}: {exc}")
        else:
            logger.warning(
                "No optimizer state found at %s. Falling back to fresh optimizer state.", str(opt_path)
            )

    model, optim = accelerator.prepare(model, optim)
    total_train_steps = infer_total_train_steps(
        args,
        train_dataloader,
        use_epochs=use_epochs,
        resume_step=effective_resume_step,
    )
    scheduler = build_lr_scheduler(optim, args, total_train_steps, start_step=effective_resume_step)
    if resume_dir is not None and not args.resume_weights_only and scheduler is not None:
        sch_path = resume_dir / "scheduler.pt"
        if sch_path.exists():
            try:
                scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
                logger.info(f"Loaded scheduler state from {sch_path}")
            except Exception as exc:
                logger.warning(f"Failed to load scheduler state from {sch_path}: {exc}")
        else:
            logger.warning(
                "No scheduler state found at %s. Falling back to scheduler inferred from resume_step=%d.",
                str(sch_path),
                effective_resume_step,
            )
    if scheduler is None and args.lr_scheduler_type != "constant":
        logger.warning(
            "Failed to infer total training steps for LR scheduler. Falling back to constant LR."
        )

    model.train()
    model_unwrapped = accelerator.unwrap_model(model)
    cfg = model_unwrapped.config
    base_future_prompt_source = str(getattr(cfg, "future_prompt_source", "pred")).lower()
    if base_future_prompt_source not in ("pred", "gt", "none"):
        logger.warning(
            "Unsupported future_prompt_source=%s, fallback to pred.",
            str(base_future_prompt_source),
        )
        base_future_prompt_source = "pred"
        cfg.future_prompt_source = "pred"
    if args.scheduled_sampling and (not bool(getattr(cfg, "use_future_token", True)) or base_future_prompt_source == "none"):
        logger.warning(
            "scheduled_sampling is enabled but use_future_token=%s and future_prompt_source=%s. "
            "Scheduled sampling will be disabled.",
            str(bool(getattr(cfg, "use_future_token", True))),
            str(base_future_prompt_source),
        )
    enable_scheduled_sampling = bool(
        args.scheduled_sampling
        and bool(getattr(cfg, "use_future_token", True))
        and base_future_prompt_source != "none"
    )

    action_flow_w = float(cfg.action_loss_weight)
    action_supervised_w = float(cfg.action_supervised_weight)
    image_w = float(cfg.image_loss_weight)
    recon_w = float(cfg.recon_loss_weight)
    logger.info(
        "Loss weights: action_flow=%.4f action_supervised=%.4f image=%.4f recon=%.4f",
        action_flow_w,
        action_supervised_w,
        image_w,
        recon_w,
    )
    logger.info(
        "Image-only training: %s",
        str(bool(getattr(cfg, "image_only_training", False))),
    )
    logger.info(
        "Image-only proprio conditioning: %s (action_mode=%s)",
        str(
            bool(getattr(cfg, "image_only_training", False))
            and str(getattr(cfg, "action_mode", "")).lower()
            in ("libero", "bridge_libero", "libero_abs", "bridge_libero_abs")
        ),
        str(getattr(cfg, "action_mode", "")),
    )
    logger.info(
        "Future prompt config: source=%s use_future_token=%s scheduled_sampling=%s strategy=%s start=%.4f end=%.4f warmup=%d exp_k=%.4f",
        base_future_prompt_source,
        str(bool(getattr(cfg, "use_future_token", True))),
        str(enable_scheduled_sampling),
        args.scheduled_sampling_strategy,
        float(args.scheduled_sampling_start),
        float(args.scheduled_sampling_end),
        int(args.scheduled_sampling_warmup_steps),
        float(args.scheduled_sampling_exp_k),
    )
    target_dataset_tags = _target_dataset_tags_from_metas(args.train_metas_path)
    sample_logged_tags = set()
    if accelerator.is_main_process:
        logger.info(f"W&B sample targets: {target_dataset_tags}")
    wandb_available = args.wandb and accelerator.is_main_process
    global_step, t0 = effective_resume_step, time.time()
    last_rolling_checkpoint = {"step": None, "path": None}

    def is_retained_checkpoint(step: int) -> bool:
        keep_interval = int(getattr(args, "keep_checkpoint_interval", 0) or 0)
        return keep_interval > 0 and step > 0 and step % keep_interval == 0

    def maybe_delete_previous_rolling_checkpoint():
        prev_step = last_rolling_checkpoint["step"]
        prev_path = last_rolling_checkpoint["path"]
        if prev_step is None or prev_path is None or is_retained_checkpoint(prev_step):
            return
        prev_path = Path(prev_path)
        expected_name = f"ckpt-{prev_step}"
        if prev_path.name != expected_name or not prev_path.exists():
            return
        try:
            shutil.rmtree(prev_path)
            logger.info(f"Deleted previous rolling checkpoint: {prev_path}")
        except Exception as exc:
            logger.warning(f"Failed to delete previous rolling checkpoint {prev_path}: {exc}")

    if resume_dir is not None and not args.resume_weights_only:
        try:
            resume_step_for_cleanup = _infer_resume_step(resume_dir)
            if (
                resume_step_for_cleanup > 0
                and not is_retained_checkpoint(resume_step_for_cleanup)
                and Path(resume_dir).resolve().parent == Path(output_dir).resolve()
            ):
                last_rolling_checkpoint["step"] = resume_step_for_cleanup
                last_rolling_checkpoint["path"] = str(resume_dir)
        except Exception as exc:
            if accelerator.is_main_process:
                logger.warning(f"Failed to initialize rolling checkpoint cleanup from resume dir: {exc}")

    def save_checkpoint(step: int):
        save_dir = os.path.join(output_dir, f"ckpt-{step}")
        accelerator.print(f"Saving model to {save_dir}")

        def legacy_torch_save(obj, path):
            torch.save(obj, path, _use_new_zipfile_serialization=False)

        def save_trainable_model_shards(unwrapped_model, target_dir: str, max_shard_size_bytes: int = 2 * 1024**3):
            os.makedirs(target_dir, exist_ok=True)
            if hasattr(unwrapped_model, "config"):
                unwrapped_model.config.save_pretrained(target_dir)

            trainable_names = {name for name, param in unwrapped_model.named_parameters() if param.requires_grad}
            state_dict = unwrapped_model.state_dict()
            items = [(name, tensor) for name, tensor in state_dict.items() if name in trainable_names]
            shards, current, current_size = [], [], 0
            total_size = 0
            for name, tensor in items:
                tensor_size = tensor.numel() * tensor.element_size()
                if current and current_size + tensor_size > max_shard_size_bytes:
                    shards.append(current)
                    current, current_size = [], 0
                current.append((name, tensor))
                current_size += tensor_size
                total_size += tensor_size
            if current:
                shards.append(current)

            weight_map = {}
            num_shards = max(1, len(shards))
            for idx, shard_items in enumerate(shards, start=1):
                shard_file = f"pytorch_model-{idx:05d}-of-{num_shards:05d}.bin"
                shard_state = {name: tensor for name, tensor in shard_items}
                legacy_torch_save(shard_state, os.path.join(target_dir, shard_file))
                for name, _ in shard_items:
                    weight_map[name] = shard_file

            with open(os.path.join(target_dir, "pytorch_model.bin.index.json"), "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "total_size": total_size,
                            "checkpoint_type": "trainable_only",
                            "num_tensors": len(items),
                        },
                        "weight_map": weight_map,
                    },
                    f,
                )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            if args.checkpoint_trainable_only:
                save_trainable_model_shards(unwrapped_model, save_dir)
            else:
                unwrapped_model.save_pretrained(
                    save_dir,
                    safe_serialization=False,
                    save_function=legacy_torch_save,
                    max_shard_size="2GB",
                )
        accelerator.wait_for_everyone()
        checkpoint_state_ok = True
        if accelerator.is_main_process:
            if args.skip_optimizer_state:
                logger.warning("Skipping optimizer state save at step %d; scheduler/global_step are still saved.", step)
            else:
                try:
                    legacy_torch_save(optim.state_dict(), os.path.join(save_dir, "optimizer.pt"))
                except Exception as exc:
                    checkpoint_state_ok = False
                    logger.warning(f"Failed to save optimizer state at step {step}: {exc}")
            if scheduler is not None:
                try:
                    legacy_torch_save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
                except Exception as exc:
                    checkpoint_state_ok = False
                    logger.warning(f"Failed to save scheduler state at step {step}: {exc}")
            try:
                with open(os.path.join(save_dir, "state.json"), "w") as f:
                    json.dump(
                        {
                            "global_step": step,
                            "checkpoint_trainable_only": bool(args.checkpoint_trainable_only),
                            "optimizer_state_saved": not bool(args.skip_optimizer_state),
                        },
                        f,
                    )
            except Exception as exc:
                checkpoint_state_ok = False
                logger.warning(f"Failed to save state.json at step {step}: {exc}")
            if checkpoint_state_ok:
                maybe_delete_previous_rolling_checkpoint()
                last_rolling_checkpoint["step"] = step
                last_rolling_checkpoint["path"] = save_dir
            else:
                logger.warning(
                    "Checkpoint state files were incomplete at step %d; previous rolling checkpoint was kept.",
                    step,
                )
        accelerator.wait_for_everyone()

    def run_batch(batch, global_step, step_in_epoch=None, progress=None, epoch_label=None, total_steps=None):
        nonlocal sample_logged_tags, t0
        lang = batch["language_instruction"]
        input_ids, attention_mask = tokenizer(lang, return_mask=True, add_special_tokens=True)
        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device)

        image_input = batch["image_input"].to(accelerator.device)
        image_input = denorm_views(image_input)
        image_mask = batch.get("image_mask")
        if image_mask is not None:
            image_mask = image_mask.to(accelerator.device).bool()
        wrist_valid_mask = None
        if image_mask is not None and image_mask.dim() >= 2 and image_mask.shape[1] > 1:
            wrist_valid_mask = image_mask[:, 1]
        front = image_input[:, 0]
        wrist = image_input[:, 1] if image_input.shape[1] > 1 else None

        front = maybe_resize(front, cfg.wan_height, cfg.wan_width)
        if wrist is not None:
            wrist = maybe_resize(wrist, cfg.wan_height, cfg.wan_width)
            if wrist_valid_mask is not None:
                if not bool(wrist_valid_mask.any()):
                    wrist = None
                else:
                    wrist = wrist * wrist_valid_mask.view(-1, 1, 1, 1).to(dtype=wrist.dtype)

        future = None
        has_future = "future_image" in batch
        if has_future:
            future = batch["future_image"].to(accelerator.device)
            future = denorm_images(future)
            future = maybe_resize(future, cfg.wan_height, cfg.wan_width)
        if args.return_future_image and not has_future and accelerator.is_main_process:
            logger.warning(f"Missing future_image in batch at step {global_step}.")

        proprio = batch["proprio"].to(accelerator.device)
        action = batch["action"].to(accelerator.device)
        image_only_training = bool(getattr(cfg, "image_only_training", False))
        keep_proprio_in_image_only = image_only_training and str(getattr(cfg, "action_mode", "")).lower() in (
            "libero",
            "bridge_libero",
            "libero_abs",
            "bridge_libero_abs",
        )
        model_proprio = proprio if (not image_only_training or keep_proprio_in_image_only) else None
        model_action = None if image_only_training else action
        if wandb_available and len(sample_logged_tags) < len(target_dataset_tags):
            try:
                import wandb
                batch_tags = batch.get("dataset_tag", None)
                if batch_tags is None:
                    batch_names = batch.get("dataset_name", None)
                    if batch_names is not None:
                        batch_tags = [_dataset_tag_from_name(x) for x in batch_names]
                if batch_tags is None:
                    batch_tags = ["unknown"] * int(front.shape[0])
                elif isinstance(batch_tags, tuple):
                    batch_tags = list(batch_tags)
                elif isinstance(batch_tags, str):
                    batch_tags = [batch_tags]

                for i in range(min(len(batch_tags), int(front.shape[0]))):
                    tag = _dataset_tag_from_name(batch_tags[i])
                    if tag not in target_dataset_tags or tag in sample_logged_tags:
                        continue
                    prefix = f"sample/{tag}"
                    sample = {
                        f"{prefix}/text": lang[i],
                        f"{prefix}/front": wandb.Image(
                            front[i].detach().float().cpu().permute(1, 2, 0).numpy(),
                            caption=lang[i],
                        ),
                    }
                    log_wrist = wrist is not None
                    if log_wrist and image_mask is not None and image_mask.dim() >= 2 and image_mask.shape[1] > 1:
                        log_wrist = bool(image_mask[i, 1].item())
                    if log_wrist:
                        sample[f"{prefix}/wrist"] = wandb.Image(
                            wrist[i].detach().float().cpu().permute(1, 2, 0).numpy()
                        )
                    if future is not None:
                        sample[f"{prefix}/future"] = wandb.Image(
                            future[i].detach().float().cpu().permute(1, 2, 0).numpy()
                        )
                    if action is not None:
                        act0 = action[i].detach().float().cpu()
                        act_cols = [f"d{j}" for j in range(act0.shape[-1])]
                        sample[f"{prefix}/action_table"] = wandb.Table(data=act0.tolist(), columns=act_cols)
                    if proprio is not None:
                        prop0 = proprio[i].detach().float().cpu()
                        prop_cols = [f"d{j}" for j in range(prop0.shape[-1])]
                        sample[f"{prefix}/proprio_table"] = wandb.Table(data=[prop0.tolist()], columns=prop_cols)
                    wandb.log(sample, step=global_step)
                    sample_logged_tags.add(tag)
                    if len(sample_logged_tags) >= len(target_dataset_tags):
                        break
            except Exception as exc:
                logger.warning(f"W&B sample logging failed: {exc}")

        ss_gt_prob = None
        future_prompt_source_now = base_future_prompt_source
        if enable_scheduled_sampling:
            ss_gt_prob = _scheduled_sampling_gt_prob(
                step=global_step,
                total_steps=total_train_steps,
                start_prob=args.scheduled_sampling_start,
                end_prob=args.scheduled_sampling_end,
                strategy=args.scheduled_sampling_strategy,
                warmup_steps=args.scheduled_sampling_warmup_steps,
                exp_k=args.scheduled_sampling_exp_k,
            )
            future_prompt_source_now = "gt" if random.random() < ss_gt_prob else "pred"

        prev_future_prompt_source = str(getattr(model_unwrapped.config, "future_prompt_source", base_future_prompt_source))
        if prev_future_prompt_source != future_prompt_source_now:
            model_unwrapped.config.future_prompt_source = future_prompt_source_now
        try:
            outputs: Dict[str, torch.Tensor] = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                front_pixel_values=front,
                wrist_pixel_values=wrist,
                wrist_valid_mask=wrist_valid_mask,
                future_pixel_values=future,
                proprio=model_proprio,
                action=model_action,
                language_instruction=lang,
            )
        finally:
            if prev_future_prompt_source != future_prompt_source_now:
                model_unwrapped.config.future_prompt_source = prev_future_prompt_source
        loss = outputs["loss"]
        if not torch.isfinite(loss):
            if accelerator.is_main_process:
                logger.warning(f"Non-finite loss at step {global_step}. Skipping batch.")
            optim.zero_grad()
            return False

        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
        if scheduler is not None:
            scheduler.step()
        optim.zero_grad()

        if global_step % args.log_interval == 0:
            raw_image_loss = float(outputs["image_loss"].detach().item())
            raw_action_flow_loss = float(outputs["action_flow_loss"].detach().item())
            raw_recon_loss = float(outputs["recon_loss"].detach().item())
            raw_action_supervised = float(outputs["action_supervised_total"].detach().item())
            logs = {
                "loss": float(loss.detach().item()),
                "image_loss": image_w * raw_image_loss,
                "action_flow_loss": action_flow_w * raw_action_flow_loss,
                "recon_loss": recon_w * raw_recon_loss,
                "action_supervised_total": action_supervised_w * raw_action_supervised,
                "raw_image_loss": raw_image_loss,
                "raw_action_flow_loss": raw_action_flow_loss,
                "raw_recon_loss": raw_recon_loss,
                "raw_action_supervised_total": raw_action_supervised,
                "lr": float(optim.param_groups[0]["lr"]),
                "future_prompt_source/is_gt": 1.0 if future_prompt_source_now == "gt" else 0.0,
            }
            if ss_gt_prob is not None:
                logs["scheduled_sampling/gt_prob"] = float(ss_gt_prob)
            raw_action_components = outputs.get("action_loss_raw", {})
            for k, v in raw_action_components.items():
                if torch.is_tensor(v):
                    logs[f"raw_action/{k}"] = float(v.detach().item())
                else:
                    try:
                        logs[f"raw_action/{k}"] = float(v)
                    except Exception:
                        pass
            accelerator.log(logs, step=global_step)
            if accelerator.is_main_process:
                if progress is not None:
                    postfix = {
                        "loss": f"{logs['loss']:.4f}",
                        "img": f"{logs['image_loss']:.4f}",
                        "act": f"{logs['action_flow_loss']:.4f}",
                        "lr": f"{logs['lr']:.2e}",
                        "fp": future_prompt_source_now,
                    }
                    if ss_gt_prob is not None:
                        postfix["p_gt"] = f"{ss_gt_prob:.2f}"
                    if step_in_epoch is not None:
                        postfix["step"] = step_in_epoch
                    progress.set_postfix(**postfix)
                dt = (time.time() - t0) / max(1, args.log_interval)
                t0 = time.time()
                cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
                gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
                if epoch_label is not None and step_in_epoch is not None:
                    prefix = f"[epoch {epoch_label} step {step_in_epoch}]"
                else:
                    prefix = f"[{global_step}/{total_steps}]"
                logger.info(
                    f"{prefix} loss={logs['loss']:.4f} img={logs['image_loss']:.4f} "
                    f"act={logs['action_flow_loss']:.4f} lr={logs['lr']:.2e} ({dt:.2f}s/it) "
                    f"CPU={cpu_mem:.1f}MB GPU={gpu_mem:.1f}MB"
                )
        return True

    if use_epochs:
        if effective_resume_step > 0:
            logger.info(
                "Resume step is %d; epoch-based resume will continue from a fresh data order.",
                effective_resume_step,
            )
        logger.info(
            f"Start training for {args.epochs} epochs | world_size={accelerator.num_processes} "
            f"| total_train_steps={total_train_steps} | resume_step={effective_resume_step}"
        )
        for epoch in range(args.epochs):
            dataset = getattr(train_dataloader, "dataset", None)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            epoch_total = args.steps_per_epoch if args.steps_per_epoch and args.steps_per_epoch > 0 else None
            if epoch_total is None:
                try:
                    epoch_total = len(train_dataloader)
                except Exception:
                    epoch_total = None
            progress = tqdm(
                total=epoch_total,
                disable=not accelerator.is_main_process,
                desc=f"epoch {epoch + 1}/{args.epochs}",
            )
            step_in_epoch = 0
            for batch in train_dataloader:
                did_step = run_batch(
                    batch,
                    global_step,
                    step_in_epoch + 1,
                    progress=progress,
                    epoch_label=f"{epoch + 1}/{args.epochs}",
                )
                if not did_step:
                    continue
                step_in_epoch += 1
                global_step += 1
                if accelerator.is_main_process:
                    progress.update(1)
                    if progress.total is not None and step_in_epoch > progress.total:
                        progress.total = step_in_epoch
                if global_step % args.save_interval == 0:
                    save_checkpoint(global_step)
            if accelerator.is_main_process:
                progress.close()
        if global_step % args.save_interval != 0:
            save_checkpoint(global_step)
    else:
        remaining_iters = max(0, total_train_steps - effective_resume_step)
        logger.info(
            f"Start training for remaining {remaining_iters} iterations | world_size={accelerator.num_processes} "
            f"| resume_step={effective_resume_step} total_iters={total_train_steps}"
        )
        progress = tqdm(total=remaining_iters, disable=not accelerator.is_main_process, desc="train")
        for batch in train_dataloader:
            did_step = run_batch(batch, global_step, progress=progress, total_steps=total_train_steps)
            if not did_step:
                continue
            global_step += 1
            if accelerator.is_main_process:
                progress.update(1)
            if global_step == total_train_steps or global_step % args.save_interval == 0:
                save_checkpoint(global_step)
            if global_step >= total_train_steps:
                break
        if accelerator.is_main_process:
            progress.close()

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WorldTeacher training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
