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

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel
from einops import rearrange

from .action_hub import build_action_space


def _force_move_tensor_attrs(module: nn.Module, device: torch.device, dtype: torch.dtype) -> None:
    """Best-effort move of stray tensor attributes that are not registered buffers/params."""
    visited: set[int] = set()

    def _move_obj(obj):
        obj_id = id(obj)
        if obj_id in visited:
            return obj
        visited.add(obj_id)
        if torch.is_tensor(obj):
            if obj.is_floating_point():
                return obj.to(device=device, dtype=dtype)
            return obj.to(device=device)
        if isinstance(obj, dict):
            changed = False
            new_dict = {}
            for k, v in obj.items():
                nv = _move_obj(v)
                new_dict[k] = nv
                if nv is not v:
                    changed = True
            return new_dict if changed else obj
        if isinstance(obj, (list, tuple)):
            new_list = []
            changed = False
            for v in obj:
                nv = _move_obj(v)
                new_list.append(nv)
                if nv is not v:
                    changed = True
            if not changed:
                return obj
            return type(obj)(new_list)
        if isinstance(obj, nn.Module):
            for name, val in list(obj.__dict__.items()):
                if name in ("_parameters", "_buffers", "_modules"):
                    continue
                nv = _move_obj(val)
                if nv is not val:
                    setattr(obj, name, nv)
            for child in obj.children():
                _move_obj(child)
        return obj

    _move_obj(module)
from .configuration_world_teacher import WorldTeacherConfig


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return torch.zeros((t.shape[0], dim), device=t.device, dtype=t.dtype)
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(1, half - 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ActionDiffusionDecoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        context_dim: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        time_embed_dim: int,
        proprio_dim: Optional[int] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.hidden_size = hidden_size
        self.time_embed_dim = time_embed_dim

        self.in_proj = nn.Linear(action_dim, hidden_size)
        self.context_proj = nn.Linear(context_dim, hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.proprio_proj = nn.Linear(proprio_dim, hidden_size) if proprio_dim is not None else None

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_size, action_dim)

    def forward(
        self,
        action_noisy: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.in_proj(action_noisy)
        t_emb = self.time_mlp(_sinusoidal_time_embedding(t, self.time_embed_dim))
        x = x + t_emb[:, None, :]
        if proprio is not None and self.proprio_proj is not None:
            x = x + self.proprio_proj(proprio)[:, None, :]
        memory = self.context_proj(context)
        x = self.decoder(tgt=x, memory=memory)
        return self.out_proj(x)


class WorldTeacher(PreTrainedModel):
    """
    WorldTeacher: Wan backbone (TI2V-5B or VACE-R2V) with multi-task outputs.

    Inputs:
      - language instruction (input_ids + attention_mask)
      - current observations (front + wrist images)

    Outputs:
      - future front frame (world model prediction, index = config.future_index)
      - future action sequence (diffusion-style)
    """

    config_class = WorldTeacherConfig
    base_model_prefix = "world_teacher"
    supports_gradient_checkpointing = True

    def __init__(self, config: WorldTeacherConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if config.future_index < 0:
            raise ValueError("future_index must be non-negative.")
        if not getattr(config, "sparse_t2v", False) and config.future_index >= config.wan_num_frames:
            raise ValueError("future_index must be within [0, wan_num_frames) when sparse_t2v is disabled.")
        if config.action_latent_hw % 2 != 0:
            raise ValueError("action_latent_hw must be divisible by 2 for Wan DiT patching.")

        # === Action space ===
        if config.action_mode.lower() == "auto":
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=config.real_action_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        # === Action diffusion (shared Wan DiT) ===
        self.action_latent_hw = config.action_latent_hw
        action_latent_dim = config.latent_dim * self.action_latent_hw * self.action_latent_hw
        self.action_to_latent = nn.Linear(dim_action, action_latent_dim)
        self.latent_to_action = nn.Linear(action_latent_dim, dim_action)
        self.proprio_proj = nn.Linear(dim_proprio, config.text_dim) if config.use_proprio else None
        self.action_head_type = getattr(config, "action_head_type", "dit")
        if self.action_head_type is None:
            self.action_head_type = "dit"
        self.action_head_type = str(self.action_head_type).lower()
        self.action_decoder = None
        self.vision_action = None
        self.vision_action_future_proj = None
        self.vision_action_tokenizer = None
        if self.action_head_type in ("decoder", "separate", "independent"):
            self.action_decoder = ActionDiffusionDecoder(
                action_dim=dim_action,
                context_dim=config.text_dim,
                hidden_size=config.action_hidden_size,
                num_layers=config.action_num_layers,
                num_heads=config.action_num_heads,
                ffn_dim=config.action_ffn_dim,
                dropout=config.action_dropout,
                time_embed_dim=config.action_time_dim,
                proprio_dim=None,
            )
        elif self.action_head_type == "vision_action":
            self._ensure_vision_action_modules()
        elif self.action_head_type not in ("dit", "shared", "shareddt", "shared_dit"):
            raise ValueError(f"Unknown action_head_type: {self.action_head_type}")

        # === Conditioning projections ===
        self.wrist_proj = nn.Linear(config.image_embed_dim, config.text_dim)
        self.front_proj = nn.Linear(config.latent_dim, config.text_dim)
        self.future_proj = nn.Linear(config.latent_dim, config.text_dim)
        self.delta_t_proj = nn.Linear(1, config.text_dim)

        # === Wan backbone (lazy init) ===
        self.wan_dit = None
        self.wan_vae = None
        self.wan_text_encoder = None
        self.wan_image_encoder = None
        self.wan_vace = None
        self._image_scheduler = None

    # ---------------------------------------------------------------------
    # Lazy imports / loading
    # ---------------------------------------------------------------------
    def _get_scheduler(self):
        if self._image_scheduler is None:
            try:
                from diffsynth.diffusion import FlowMatchScheduler
            except Exception as exc:
                raise RuntimeError(
                    "DiffSynth-Studio is required for WorldTeacher. Please ensure diffsynth is available."
                ) from exc
            self._image_scheduler = FlowMatchScheduler("Wan")
        return self._image_scheduler

    def _get_action_head_type(self) -> str:
        head = getattr(self.config, "action_head_type", None)
        if head is None:
            head = getattr(self, "action_head_type", "dit")
        head = str(head).lower()
        if head == "shared_dit":
            head = "dit"
        return head

    def _ensure_action_decoder(self) -> bool:
        head = self._get_action_head_type()
        if head in ("decoder", "separate", "independent"):
            if self.action_decoder is None:
                dim_action = self.action_space.dim_action
                self.action_decoder = ActionDiffusionDecoder(
                    action_dim=dim_action,
                    context_dim=self.config.text_dim,
                    hidden_size=self.config.action_hidden_size,
                    num_layers=self.config.action_num_layers,
                    num_heads=self.config.action_num_heads,
                    ffn_dim=self.config.action_ffn_dim,
                    dropout=self.config.action_dropout,
                    time_embed_dim=self.config.action_time_dim,
                    proprio_dim=None,
                )
            return True
        return False

    def _ensure_vision_action_modules(self) -> bool:
        head = self._get_action_head_type()
        if head != "vision_action":
            return False
        if self.vision_action is None:
            from .configuration_vision_action import VisionActionConfig
            from .modeling_vision_action import VisionAction

            vision_action_cfg = VisionActionConfig.from_pretrained(self.config.vision_action_model_id)
            vision_action_cfg.action_mode = self.config.action_mode
            vision_action_cfg.num_actions = self.config.num_actions
            vision_action_cfg.max_action_dim = self.config.max_action_dim
            vision_action_cfg.real_action_dim = self.config.real_action_dim
            self.vision_action = VisionAction.from_pretrained(
                self.config.vision_action_model_id,
                config=vision_action_cfg,
                ignore_mismatched_sizes=True,
            )
            projection_dim = getattr(self.vision_action.vlm.config, "projection_dim", None)
            if projection_dim is None:
                raise ValueError("VisionAction Florence config must provide projection_dim.")
            self.vision_action_future_proj = nn.Linear(self.config.latent_dim, projection_dim)
            self.vision_action_tokenizer = AutoTokenizer.from_pretrained(self.config.vision_action_model_id)
        return True

    def _encode_vision_action_text(self, language_instruction: List[str], device: torch.device) -> torch.LongTensor:
        self._ensure_vision_action_modules()
        if self.vision_action_tokenizer is None:
            raise RuntimeError("VisionAction tokenizer is not initialized.")
        toks = self.vision_action_tokenizer(
            language_instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=int(getattr(self.config, "vision_action_language_max_length", 50)),
            truncation=True,
        )
        return toks["input_ids"].to(device=device)

    def _build_vision_action_image_inputs(
        self,
        front_pixel_values: torch.Tensor,
        wrist_pixel_values: Optional[torch.Tensor],
        wrist_valid_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        front = front_pixel_values.to(device=device, dtype=dtype)
        if wrist_pixel_values is None:
            wrist = torch.zeros_like(front)
            wrist_valid = torch.zeros((front.shape[0],), device=device, dtype=torch.bool)
        else:
            wrist = wrist_pixel_values.to(device=device, dtype=dtype)
            wrist_valid = torch.ones((front.shape[0],), device=device, dtype=torch.bool)
            if wrist_valid_mask is not None:
                wrist_valid = wrist_valid_mask.to(device=device).view(-1).bool()
                wrist = wrist * wrist_valid.view(-1, 1, 1, 1).to(dtype=wrist.dtype)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
        front = (front - mean) / std
        wrist = (wrist - mean) / std

        image_input = torch.stack([front, wrist], dim=1)
        image_mask = torch.stack(
            [
                torch.ones((front.shape[0],), device=device, dtype=torch.bool),
                wrist_valid,
            ],
            dim=1,
        )
        return image_input, image_mask

    def _resolve_backbone(self) -> str:
        backbone = getattr(self.config, "wan_backbone", "auto")
        if backbone is None:
            backbone = "auto"
        backbone = backbone.lower()
        if backbone == "auto":
            model_id = (getattr(self.config, "wan_model_id", "") or "").lower()
            if "vace" in model_id:
                return "vace_r2v"
            return "ti2v"
        return backbone

    def _use_vace(self) -> bool:
        backbone = self._resolve_backbone()
        return backbone in ("vace", "vace_r2v", "vace-r2v", "r2v")

    def _ensure_wan_modules(self):
        if self.wan_dit is not None:
            device = next(self.parameters()).device
            dtype = self.config.wan_torch_dtype
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            for module in (self.wan_text_encoder, self.wan_dit, self.wan_vae, self.wan_image_encoder, self.wan_vace):
                if module is None:
                    continue
                try:
                    module.to(device=device, dtype=dtype)
                except Exception:
                    module.to(device=device)
                _force_move_tensor_attrs(module, device=device, dtype=dtype)
            return
        try:
            from diffsynth.pipelines.wan_video import WanVideoPipeline
            from diffsynth.core import ModelConfig
        except Exception as exc:
            raise RuntimeError(
                "DiffSynth-Studio is required for WorldTeacher. Please ensure diffsynth is available."
            ) from exc

        device = next(self.parameters()).device
        dtype = self.config.wan_torch_dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        use_vace = self._use_vace()
        vae_filename = "Wan2.1_VAE.pth" if use_vace else "Wan2.2_VAE.pth"
        model_configs = [
            ModelConfig(model_id=self.config.wan_model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=self.config.wan_model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=self.config.wan_model_id, origin_file_pattern=vae_filename),
        ]
        if self.config.load_image_encoder:
            model_configs.append(
                ModelConfig(
                    model_id=self.config.wan_model_id,
                    origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                )
            )

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=None,
            redirect_common_files=True,
        )

        self.wan_text_encoder = pipe.text_encoder
        self.wan_dit = pipe.dit
        self.wan_vae = pipe.vae
        self.wan_image_encoder = pipe.image_encoder
        self.wan_vace = pipe.vace
        for module in (self.wan_text_encoder, self.wan_dit, self.wan_vae, self.wan_image_encoder, self.wan_vace):
            if module is None:
                continue
            try:
                module.to(device=device, dtype=dtype)
            except Exception:
                module.to(device=device)
            _force_move_tensor_attrs(module, device=device, dtype=dtype)
        if self.wan_dit is not None:
            patch_h, patch_w = self.wan_dit.patch_size[1], self.wan_dit.patch_size[2]
            if self.action_latent_hw % patch_h != 0 or self.action_latent_hw % patch_w != 0:
                raise ValueError("action_latent_hw must be divisible by Wan DiT patch size.")
        if self.wan_vae is not None and hasattr(self.wan_vae, "z_dim"):
            if int(self.wan_vae.z_dim) != int(self.config.latent_dim):
                raise ValueError(
                    f"WorldTeacher latent_dim={self.config.latent_dim} does not match VAE z_dim={self.wan_vae.z_dim}. "
                    "Please set latent_dim to the correct value for the backbone."
                )

        if self.config.freeze_dit and self.wan_dit is not None:
            self.wan_dit.requires_grad_(False)
        if self.config.freeze_dit and self.wan_vace is not None:
            self.wan_vace.requires_grad_(False)
        if self.config.freeze_text_encoder and self.wan_text_encoder is not None:
            self.wan_text_encoder.requires_grad_(False)
        if self.config.freeze_vae and self.wan_vae is not None:
            self.wan_vae.requires_grad_(False)
        if self.config.freeze_image_encoder and self.wan_image_encoder is not None:
            self.wan_image_encoder.requires_grad_(False)

    # ---------------------------------------------------------------------
    # Encoding helpers
    # ---------------------------------------------------------------------
    def _encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        self._ensure_wan_modules()
        assert self.wan_text_encoder is not None
        device = next(self.parameters()).device
        try:
            device = next(self.wan_text_encoder.parameters()).device
        except StopIteration:
            pass
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        context = self.wan_text_encoder(input_ids, attention_mask)
        if attention_mask is not None:
            context = context * attention_mask.unsqueeze(-1)
        return context

    @staticmethod
    def _normalize_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            pixel_values = pixel_values.float()
        min_val = pixel_values.amin().item()
        max_val = pixel_values.amax().item()
        if min_val >= 0.0 and max_val <= 1.0:
            return pixel_values * 2 - 1
        return pixel_values

    def _encode_image_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self._ensure_wan_modules()
        assert self.wan_vae is not None
        pixel_values = self._normalize_pixel_values(pixel_values)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        target_device = next(self.wan_vae.parameters()).device
        target_dtype = next(self.wan_vae.parameters()).dtype
        if pixel_values.dim() == 4:
            videos = [pv.unsqueeze(1).to(device=target_device, dtype=target_dtype) for pv in pixel_values]
        elif pixel_values.dim() == 5:
            videos = [pv.to(device=target_device, dtype=target_dtype) for pv in pixel_values]
        else:
            raise ValueError(f"Unsupported pixel_values shape: {pixel_values.shape}")
        latents = self.wan_vae.encode(
            videos,
            device=target_device,
            tiled=self.config.wan_tiled,
            tile_size=self.config.wan_tile_size,
            tile_stride=self.config.wan_tile_stride,
        )
        return latents  # [B, C, T, H, W]

    def _resize_video_tensor(
        self, pixel_values: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        if pixel_values.shape[-2:] == (height, width):
            return pixel_values
        if pixel_values.dim() == 4:
            return F.interpolate(pixel_values, size=(height, width), mode="bilinear", align_corners=False)
        if pixel_values.dim() == 5:
            b, c, t, h, w = pixel_values.shape
            flat = pixel_values.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            flat = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
            return flat.reshape(b, t, c, height, width).permute(0, 2, 1, 3, 4)
        raise ValueError(f"Unsupported pixel_values shape for resize: {pixel_values.shape}")

    def _build_vace_context(
        self,
        reference_latents: Optional[torch.Tensor] = None,
        reference_pixel_values: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if not self._use_vace():
            return None
        self._ensure_wan_modules()
        if self.wan_vae is None or self.wan_vace is None:
            raise RuntimeError("VACE backbone requested but VACE modules are not available.")
        if reference_latents is None and reference_pixel_values is None:
            return None
        if num_frames is None:
            num_frames = 2 if self.config.sparse_t2v else self.config.wan_num_frames

        target_device = next(self.wan_vae.parameters()).device
        target_dtype = next(self.wan_vae.parameters()).dtype
        height = int(self.config.wan_height)
        width = int(self.config.wan_width)

        if reference_latents is None:
            ref = reference_pixel_values
            if ref is None:
                return None
            if ref.dim() == 3:
                ref = ref.unsqueeze(0)
            if ref.dim() == 4:
                ref = ref.unsqueeze(2)
            if ref.dim() != 5:
                raise ValueError(f"Unsupported reference_pixel_values shape: {ref.shape}")
            ref = self._normalize_pixel_values(ref)
            ref = self._resize_video_tensor(ref, height, width)
            ref = ref.to(device=target_device, dtype=target_dtype)
            ref_videos = [ref[i] for i in range(ref.shape[0])]
            reference_latents = self.wan_vae.encode(
                ref_videos,
                device=target_device,
                tiled=self.config.wan_tiled,
                tile_size=self.config.wan_tile_size,
                tile_stride=self.config.wan_tile_stride,
            )

        if reference_latents.dim() == 4:
            reference_latents = reference_latents.unsqueeze(2)
        if reference_latents.dim() != 5:
            raise ValueError(f"Unsupported reference_latents shape: {reference_latents.shape}")
        reference_latents = reference_latents.to(device=target_device, dtype=target_dtype)
        b = reference_latents.shape[0]

        vace_video = torch.zeros((b, 3, num_frames, height, width), device=target_device, dtype=target_dtype)
        vace_mask = torch.ones_like(vace_video)

        inactive = vace_video * (1 - vace_mask)
        reactive = vace_video * vace_mask
        inactive_latents = self.wan_vae.encode(
            [inactive[i] for i in range(b)],
            device=target_device,
            tiled=self.config.wan_tiled,
            tile_size=self.config.wan_tile_size,
            tile_stride=self.config.wan_tile_stride,
        ).to(device=target_device, dtype=target_dtype)
        reactive_latents = self.wan_vae.encode(
            [reactive[i] for i in range(b)],
            device=target_device,
            tiled=self.config.wan_tiled,
            tile_size=self.config.wan_tile_size,
            tile_stride=self.config.wan_tile_stride,
        ).to(device=target_device, dtype=target_dtype)
        vace_video_latents = torch.cat((inactive_latents, reactive_latents), dim=1)

        mask = vace_mask[:, 0]
        vace_mask_latents = rearrange(mask, "B T (H P) (W Q) -> B (P Q) T H W", P=8, Q=8)
        vace_mask_latents = F.interpolate(
            vace_mask_latents,
            size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]),
            mode="nearest-exact",
        ).to(device=target_device, dtype=target_dtype)

        reference_latents = torch.cat((reference_latents, torch.zeros_like(reference_latents)), dim=1)
        vace_video_latents = torch.cat((reference_latents, vace_video_latents), dim=2)
        zeros_mask = torch.zeros(
            (b, vace_mask_latents.shape[1], reference_latents.shape[2], vace_mask_latents.shape[3], vace_mask_latents.shape[4]),
            device=target_device,
            dtype=target_dtype,
        )
        vace_mask_latents = torch.cat((zeros_mask, vace_mask_latents), dim=2)
        vace_context = torch.cat((vace_video_latents, vace_mask_latents), dim=1)
        return vace_context

    def _encode_wrist_tokens(
        self,
        wrist_pixel_values: torch.Tensor,
        wrist_valid_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.config.use_wrist_token:
            return None
        self._ensure_wan_modules()
        if self.wan_image_encoder is None:
            return None
        if wrist_pixel_values.dim() == 3:
            wrist_pixel_values = wrist_pixel_values.unsqueeze(0)
        wrist_pixel_values = self._normalize_pixel_values(wrist_pixel_values)
        wrist_pixel_values = wrist_pixel_values.to(next(self.parameters()).device)
        valid_mask = None
        if wrist_valid_mask is not None:
            valid_mask = wrist_valid_mask.to(device=wrist_pixel_values.device).view(-1).bool()
            if valid_mask.numel() != wrist_pixel_values.shape[0]:
                raise ValueError(
                    f"wrist_valid_mask size mismatch: {valid_mask.numel()} vs batch {wrist_pixel_values.shape[0]}"
                )
            if not bool(valid_mask.any()):
                return None
        with torch.no_grad() if self.config.freeze_image_encoder else torch.enable_grad():
            if valid_mask is None or bool(valid_mask.all()):
                wrist_feat = self.wan_image_encoder.encode_image([wrist_pixel_values])
            else:
                wrist_feat_valid = self.wan_image_encoder.encode_image([wrist_pixel_values[valid_mask]])
                wrist_feat = wrist_feat_valid.new_zeros(
                    (wrist_pixel_values.shape[0], wrist_feat_valid.shape[1], wrist_feat_valid.shape[2])
                )
                wrist_feat[valid_mask] = wrist_feat_valid
        target_device = self.wrist_proj.weight.device
        target_dtype = self.wrist_proj.weight.dtype
        wrist_feat = wrist_feat.to(device=target_device, dtype=target_dtype)
        if self.config.wrist_token_mode == "cls":
            wrist_feat = wrist_feat[:, :1]
        elif self.config.wrist_token_mode == "mean":
            wrist_feat = wrist_feat.mean(dim=1, keepdim=True)
        wrist_tokens = self.wrist_proj(wrist_feat)
        if valid_mask is not None:
            wrist_tokens = wrist_tokens * valid_mask.to(device=wrist_tokens.device, dtype=wrist_tokens.dtype).view(-1, 1, 1)
        return wrist_tokens

    def _latent_token(self, latents: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        pooled = latents.mean(dim=(2, 3))
        pooled = pooled.to(device=proj.weight.device, dtype=proj.weight.dtype)
        return proj(pooled).unsqueeze(1)

    def _delta_t_token(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.delta_t_proj is None:
            return None
        dt = torch.full((batch_size, 1), float(self.config.future_index), device=device, dtype=dtype)
        scale = float(getattr(self.config, "delta_t_scale", 1.0))
        if scale and scale != 1.0:
            dt = dt / scale
        dt = dt.to(device=self.delta_t_proj.weight.device, dtype=self.delta_t_proj.weight.dtype)
        token = self.delta_t_proj(dt).unsqueeze(1)
        return token

    def _action_to_latents(self, action: torch.Tensor) -> torch.Tensor:
        b, t, _ = action.shape
        hw = self.action_latent_hw
        latents = self.action_to_latent(action).view(b, t, self.config.latent_dim, hw, hw)
        return latents.permute(0, 2, 1, 3, 4).contiguous()

    def _latents_to_action(self, latents: torch.Tensor) -> torch.Tensor:
        b, _, t, _, _ = latents.shape
        action_tokens = latents.permute(0, 2, 1, 3, 4).contiguous().view(b, t, -1)
        action_tokens = action_tokens.to(device=self.latent_to_action.weight.device, dtype=self.latent_to_action.weight.dtype)
        return self.latent_to_action(action_tokens)

    @staticmethod
    def _resize_temporal_latents_hw(latents: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        if latents.shape[-2:] == (out_h, out_w):
            return latents
        b, c, t, h, w = latents.shape
        flat = latents.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        flat = F.interpolate(flat, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return flat.reshape(b, t, c, out_h, out_w).permute(0, 2, 1, 3, 4).contiguous()

    def _build_context(
        self,
        text_context: torch.Tensor,
        wrist_tokens: Optional[torch.Tensor] = None,
        front_latents: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_dtype = text_context.dtype
        target_device = text_context.device
        tokens = [text_context]
        if wrist_tokens is not None:
            tokens.append(wrist_tokens.to(device=target_device, dtype=target_dtype))
        if self.config.use_front_latent_token and front_latents is not None:
            tokens.append(self._latent_token(front_latents, self.front_proj).to(device=target_device, dtype=target_dtype))
        if self.config.use_future_token and future_latents is not None:
            tokens.append(self._latent_token(future_latents, self.future_proj).to(device=target_device, dtype=target_dtype))
        if self.config.sparse_t2v:
            delta_token = self._delta_t_token(text_context.shape[0], target_device, target_dtype)
            if delta_token is not None:
                tokens.append(delta_token.to(device=target_device, dtype=target_dtype))
        if proprio is not None and self.proprio_proj is not None:
            proprio = proprio.to(device=self.proprio_proj.weight.device, dtype=self.proprio_proj.weight.dtype)
            tokens.append(self.proprio_proj(proprio).unsqueeze(1).to(device=target_device, dtype=target_dtype))
        return torch.cat(tokens, dim=1)

    # ---------------------------------------------------------------------
    # Training forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        front_pixel_values: Optional[torch.Tensor] = None,
        front_latents: Optional[torch.Tensor] = None,
        wrist_pixel_values: Optional[torch.Tensor] = None,
        wrist_valid_mask: Optional[torch.Tensor] = None,
        future_pixel_values: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        language_instruction: Optional[List[str]] = None,
        return_future_pixels: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        if front_latents is None and front_pixel_values is None:
            raise ValueError("Either front_latents or front_pixel_values must be provided.")

        if future_latents is None and future_pixel_values is not None:
            future_latents = self._encode_image_latents(future_pixel_values)
        front_latents_full = front_latents
        if front_latents_full is None and front_pixel_values is not None:
            front_latents_full = self._encode_image_latents(front_pixel_values)
        if front_latents_full is not None and front_latents_full.dim() == 4:
            front_latents_full = front_latents_full.unsqueeze(2)

        # Squeeze time dimension for single-frame latents
        front_latents = front_latents_full
        if front_latents is not None and front_latents.dim() == 5:
            front_latents = front_latents[:, :, 0]
        if future_latents is not None and future_latents.dim() == 5:
            future_latents = future_latents[:, :, 0]
        if front_latents is not None:
            front_latents = front_latents.to(device)
        if future_latents is not None:
            future_latents = future_latents.to(device)
        if action is not None:
            action = action.to(device)
        if proprio is not None:
            proprio = proprio.to(device)
        if front_latents is not None:
            front_latents = torch.nan_to_num(front_latents)
        if front_latents_full is not None:
            front_latents_full = torch.nan_to_num(front_latents_full)
        if future_latents is not None:
            future_latents = torch.nan_to_num(future_latents)
        if action is not None:
            action = torch.nan_to_num(action)

        text_context = self._encode_text(input_ids, attention_mask)
        wrist_tokens = None
        if wrist_pixel_values is not None:
            wrist_tokens = self._encode_wrist_tokens(wrist_pixel_values, wrist_valid_mask=wrist_valid_mask)

        proprio_token = proprio if self.config.use_proprio else None
        # Build image context (no future token)
        context_img = self._build_context(
            text_context=text_context,
            wrist_tokens=wrist_tokens,
            front_latents=front_latents,
            future_latents=None,
            proprio=None,
        )
        context_joint = self._build_context(
            text_context=text_context,
            wrist_tokens=wrist_tokens,
            front_latents=front_latents,
            future_latents=None,
            proprio=proprio_token,
        )

        image_loss = torch.zeros((), device=text_context.device)
        recon_loss = torch.zeros((), device=text_context.device)
        pred_future_latents = None
        action_flow_loss = torch.zeros((), device=text_context.device)
        action_supervised_total = torch.zeros((), device=text_context.device)
        action_loss_dict = {}
        action_loss_raw = {}
        pred_action = None

        model_fn_wan_video = None
        scheduler = None
        if future_latents is not None or action is not None:
            try:
                from diffsynth.pipelines.wan_video import model_fn_wan_video as _model_fn_wan_video
            except Exception as exc:
                raise RuntimeError("DiffSynth-Studio is required for WorldTeacher.") from exc
            model_fn_wan_video = _model_fn_wan_video
            scheduler = self._get_scheduler()
            scheduler.set_timesteps(self.config.image_num_train_timesteps, training=True)
        image_only_training = bool(getattr(self.config, "image_only_training", False))
        action_for_training = None if image_only_training else action
        action_head = self._get_action_head_type()
        use_decoder = self._ensure_action_decoder()
        joint_single_forward = bool(getattr(self.config, "joint_single_forward", False))
        use_joint_forward = (
            joint_single_forward
            and future_latents is not None
            and action_for_training is not None
            and model_fn_wan_video is not None
            and action_head in ("dit", "shared", "shareddt", "shared_dit")
            and not self._use_vace()
        )

        if use_joint_forward:
            # DreamZero-style single Wan DiT pass for future frame + action diffusion.
            timestep_id = torch.randint(0, len(scheduler.timesteps), (1,))

            timestep_img = scheduler.timesteps[timestep_id].to(device=future_latents.device, dtype=future_latents.dtype)
            sigma_img = scheduler.sigmas[timestep_id].to(device=future_latents.device, dtype=future_latents.dtype)
            noise_future = torch.randn_like(future_latents)
            noisy_future = scheduler.add_noise(future_latents, noise_future, timestep_img)
            target_future = scheduler.training_target(future_latents, noise_future, timestep_img)

            b, c, h, w = future_latents.shape
            if self.config.sparse_t2v:
                num_img_frames = 2
                target_index = 1
            else:
                num_img_frames = self.config.wan_num_frames
                target_index = self.config.future_index

            action_latents = self._action_to_latents(action_for_training).to(device=device, dtype=context_joint.dtype)
            timestep_act = scheduler.timesteps[timestep_id].to(device=action_latents.device, dtype=action_latents.dtype)
            sigma_act = scheduler.sigmas[timestep_id].to(device=action_latents.device, dtype=action_latents.dtype)
            noise_action = torch.randn_like(action_latents)
            action_noisy = scheduler.add_noise(action_latents, noise_action, timestep_act)
            target_action = scheduler.training_target(action_latents, noise_action, timestep_act)
            action_noisy_up = self._resize_temporal_latents_hw(action_noisy, h, w).to(
                device=future_latents.device, dtype=future_latents.dtype
            )

            joint_frames = num_img_frames + action_noisy_up.shape[2]
            latents = future_latents.new_zeros((b, c, joint_frames, h, w))
            latents[:, :, 0] = front_latents
            latents[:, :, target_index] = noisy_future
            latents[:, :, num_img_frames:] = action_noisy_up

            v_pred = model_fn_wan_video(
                dit=self.wan_dit,
                latents=latents,
                timestep=timestep_img,
                context=context_joint,
                fuse_vae_embedding_in_latents=True,
            )

            v_pred_future = v_pred[:, :, target_index]
            image_loss = F.mse_loss(v_pred_future.float(), target_future.float()) * scheduler.training_weight(timestep_img)
            pred_future_latents = noisy_future - sigma_img * v_pred_future
            if self.config.recon_loss_weight > 0:
                recon_loss = F.l1_loss(pred_future_latents.float(), future_latents.float())

            v_pred_action_up = v_pred[:, :, num_img_frames:]
            v_pred_action = self._resize_temporal_latents_hw(
                v_pred_action_up,
                action_noisy.shape[-2],
                action_noisy.shape[-1],
            ).to(device=action_noisy.device, dtype=action_noisy.dtype)
            action_flow_loss = F.mse_loss(v_pred_action.float(), target_action.float()) * scheduler.training_weight(timestep_act)
            pred_action_latents = action_noisy - sigma_act * v_pred_action
            pred_action = self._latents_to_action(pred_action_latents)
        else:
            if future_latents is not None:
                timestep_id = torch.randint(0, len(scheduler.timesteps), (1,))
                timestep = scheduler.timesteps[timestep_id].to(device=future_latents.device, dtype=future_latents.dtype)
                sigma = scheduler.sigmas[timestep_id].to(device=future_latents.device, dtype=future_latents.dtype)

                noise = torch.randn_like(future_latents)
                noisy_future = scheduler.add_noise(future_latents, noise, timestep)
                target = scheduler.training_target(future_latents, noise, timestep)

                # Prepare latents sequence with fused first frame
                b, c, h, w = future_latents.shape
                if self.config.sparse_t2v:
                    num_frames = 2
                    target_index = 1
                else:
                    num_frames = self.config.wan_num_frames
                    target_index = self.config.future_index
                latents = future_latents.new_zeros((b, c, num_frames, h, w))
                latents[:, :, 0] = front_latents
                latents[:, :, target_index] = noisy_future

                vace_context = None
                vace_scale = None
                if self._use_vace():
                    vace_context = self._build_vace_context(
                        reference_latents=front_latents_full,
                        num_frames=num_frames,
                    )
                    vace_scale = float(getattr(self.config, "vace_scale", 1.0))

                v_pred = model_fn_wan_video(
                    dit=self.wan_dit,
                    vace=self.wan_vace,
                    latents=latents,
                    timestep=timestep,
                    context=context_img,
                    fuse_vae_embedding_in_latents=True,
                    vace_context=vace_context,
                    vace_scale=vace_scale,
                )

                v_pred_future = v_pred[:, :, target_index]
                image_loss = F.mse_loss(v_pred_future.float(), target.float()) * scheduler.training_weight(timestep)
                pred_future_latents = noisy_future - sigma * v_pred_future

                if self.config.recon_loss_weight > 0:
                    recon_loss = F.l1_loss(pred_future_latents.float(), future_latents.float())

            # Build action context (optionally with future token)
            future_prompt = None
            if self.config.use_future_token and future_latents is not None and self.config.future_prompt_source != "none":
                if self.config.future_prompt_source == "gt":
                    future_prompt = future_latents
                elif self.config.future_prompt_source == "pred" and pred_future_latents is not None:
                    future_prompt = pred_future_latents
                if self.config.detach_future_prompt and future_prompt is not None:
                    future_prompt = future_prompt.detach()

            context_act = self._build_context(
                text_context=text_context,
                wrist_tokens=wrist_tokens,
                front_latents=front_latents,
                future_latents=future_prompt,
                proprio=proprio_token,
            )

            if action_for_training is not None:
                if action_head == "vision_action":
                    self._ensure_vision_action_modules()
                    vision_action_device = action_for_training.device
                    vision_action_dtype = action_for_training.dtype
                    if language_instruction is None:
                        raise ValueError("language_instruction is required when action_head_type='vision_action'.")
                    vision_action_input_ids = self._encode_vision_action_text(language_instruction, device=vision_action_device)
                    if front_pixel_values is None:
                        raise ValueError("front_pixel_values is required when action_head_type='vision_action'.")
                    vision_action_image_input, vision_action_image_mask = self._build_vision_action_image_inputs(
                        front_pixel_values=front_pixel_values,
                        wrist_pixel_values=wrist_pixel_values,
                        wrist_valid_mask=wrist_valid_mask,
                        device=vision_action_device,
                        dtype=vision_action_dtype,
                    )
                    vision_action_image_input = vision_action_image_input.to(
                        device=next(self.vision_action.parameters()).device,
                        dtype=next(self.vision_action.parameters()).dtype,
                    )
                    vision_action_image_mask = vision_action_image_mask.to(device=vision_action_image_input.device)
                    vision_action_input_ids = vision_action_input_ids.to(device=vision_action_image_input.device)
                    enc = self.vision_action.forward_vlm(
                        input_ids=vision_action_input_ids,
                        pixel_values=vision_action_image_input,
                        image_mask=vision_action_image_mask,
                    )
                    if (
                        getattr(self.config, "vision_action_use_future_latent", True)
                        and future_prompt is not None
                        and self.vision_action_future_proj is not None
                    ):
                        pooled_future = future_prompt.mean(dim=(2, 3))
                        pooled_future = pooled_future.to(
                            device=self.vision_action_future_proj.weight.device,
                            dtype=self.vision_action_future_proj.weight.dtype,
                        )
                        future_token = self.vision_action_future_proj(pooled_future).unsqueeze(1)
                        enc["aux_visual_inputs"] = torch.cat(
                            [enc["aux_visual_inputs"], future_token.to(enc["aux_visual_inputs"].dtype)],
                            dim=1,
                        )
                    bsz = action_for_training.shape[0]
                    t = (torch.rand(1, device=action_for_training.device, dtype=action_for_training.dtype)
                         + torch.arange(bsz, device=action_for_training.device, dtype=action_for_training.dtype) / max(1, bsz)) % (1 - 1e-5)
                    action_noisy = (
                        torch.randn_like(action_for_training) * t.view(-1, 1, 1)
                        + action_for_training * (1 - t).view(-1, 1, 1)
                    )
                    if proprio is None:
                        proprio_in = action_for_training.new_zeros(
                            (bsz, getattr(self.vision_action.action_space, "dim_proprio", action_for_training.shape[-1]))
                        )
                    else:
                        proprio_in = proprio
                    proprio_m, action_noisy_m = self.vision_action.action_space.preprocess(proprio_in, action_noisy)
                    domain_id = torch.full(
                        (bsz,),
                        int(getattr(self.config, "vision_action_domain_id", 0)),
                        dtype=torch.long,
                        device=action_noisy_m.device,
                    )
                    pred_action = self.vision_action.transformer(
                        domain_id=domain_id,
                        action_with_noise=action_noisy_m,
                        t=t.to(device=action_noisy_m.device, dtype=action_noisy_m.dtype),
                        proprio=proprio_m,
                        **enc,
                    )
                    action_flow_dict = self.vision_action.action_space.compute_loss(pred_action, action_for_training)
                    action_flow_loss = sum(action_flow_dict.values())
                else:
                    timestep_id = torch.randint(0, len(scheduler.timesteps), (1,))
                    timestep = scheduler.timesteps[timestep_id].to(
                        device=action_for_training.device, dtype=action_for_training.dtype
                    )
                    sigma = scheduler.sigmas[timestep_id].to(
                        device=action_for_training.device, dtype=action_for_training.dtype
                    )

                    if action_head in ("decoder", "separate", "independent") and use_decoder and self.action_decoder is not None:
                        action_noisy = action_for_training
                        noise = torch.randn_like(action_noisy)
                        action_noisy = scheduler.add_noise(action_noisy, noise, timestep)
                        target = scheduler.training_target(action_for_training, noise, timestep)

                        if timestep.dim() == 0:
                            timestep_act = timestep.unsqueeze(0).expand(action_noisy.shape[0])
                        else:
                            timestep_act = timestep

                        target_device = action_noisy.device
                        target_dtype = action_noisy.dtype
                        context_action = context_act.to(device=target_device, dtype=target_dtype)
                        v_pred_action = self.action_decoder(
                            action_noisy.to(device=target_device, dtype=target_dtype),
                            t=timestep_act.to(device=target_device, dtype=target_dtype),
                            context=context_action,
                            proprio=None,
                        )
                        action_flow_loss = F.mse_loss(v_pred_action.float(), target.float()) * scheduler.training_weight(timestep)
                        pred_action = action_noisy - sigma * v_pred_action
                    else:
                        action_latents = self._action_to_latents(action_for_training).to(device=device, dtype=context_act.dtype)
                        timestep = scheduler.timesteps[timestep_id].to(device=action_latents.device, dtype=action_latents.dtype)
                        sigma = scheduler.sigmas[timestep_id].to(device=action_latents.device, dtype=action_latents.dtype)
                        noise = torch.randn_like(action_latents)
                        action_noisy = scheduler.add_noise(action_latents, noise, timestep)
                        target = scheduler.training_target(action_latents, noise, timestep)

                        v_pred_action = model_fn_wan_video(
                            dit=self.wan_dit,
                            latents=action_noisy,
                            timestep=timestep,
                            context=context_act,
                            fuse_vae_embedding_in_latents=False,
                        )
                        action_flow_loss = F.mse_loss(v_pred_action.float(), target.float()) * scheduler.training_weight(timestep)
                        pred_action_latents = action_noisy - sigma * v_pred_action
                        pred_action = self._latents_to_action(pred_action_latents)

        if action_for_training is not None and pred_action is not None:
            if self.config.action_supervised_weight > 0:
                action_loss_dict = self.action_space.compute_loss(pred_action, action_for_training)
                action_supervised_total = sum(action_loss_dict.values())
            try:
                action_loss_raw = self.action_space.compute_loss_raw(pred_action, action_for_training)
            except Exception:
                action_loss_raw = {}

        total_loss = (
            self.config.action_loss_weight * action_flow_loss
            + self.config.action_supervised_weight * action_supervised_total
            + self.config.image_loss_weight * image_loss
            + self.config.recon_loss_weight * recon_loss
        )

        output = {
            "loss": total_loss,
            "action_loss": action_loss_dict,
            "action_flow_loss": action_flow_loss,
            "action_supervised_total": action_supervised_total,
            "action_loss_raw": action_loss_raw,
            "image_loss": image_loss,
            "recon_loss": recon_loss,
            "pred_action": pred_action,
            "pred_future_latents": pred_future_latents,
        }

        if return_future_pixels and pred_future_latents is not None:
            output["pred_future_pixels"] = self.decode_latents(pred_future_latents)
        return output

    # ---------------------------------------------------------------------
    # Inference helpers
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        self._ensure_wan_modules()
        assert self.wan_vae is not None
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)  # [B, C, 1, H, W]
        if latents.dim() == 5:
            if latents.shape[0] == 1:
                hidden_states = [latents[0]]  # [C, T, H, W]
            else:
                hidden_states = [latents[i] for i in range(latents.shape[0])]
        else:
            raise ValueError(f"Unsupported latents shape: {tuple(latents.shape)}")
        videos = self.wan_vae.decode(
            hidden_states,
            device=next(self.parameters()).device,
            tiled=self.config.wan_tiled,
            tile_size=self.config.wan_tile_size,
            tile_stride=self.config.wan_tile_stride,
        )
        video = videos[0]
        return (video.clamp(-1, 1) + 1) * 0.5

    @torch.no_grad()
    def sample_future_latents(
        self,
        front_latents: torch.Tensor,
        context: torch.Tensor,
        num_inference_steps: int = 50,
        reference_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scheduler = self._get_scheduler()
        scheduler.set_timesteps(num_inference_steps)

        # Ensure context is on the same device/dtype as latents/dit
        context = context.to(device=front_latents.device, dtype=front_latents.dtype)

        b, c, h, w = front_latents.shape
        if self.config.sparse_t2v:
            num_frames = 2
            target_index = 1
        else:
            num_frames = self.config.wan_num_frames
            target_index = self.config.future_index
        latents = torch.randn(b, c, num_frames, h, w, device=front_latents.device, dtype=front_latents.dtype)
        latents[:, :, 0] = front_latents

        try:
            from diffsynth.pipelines.wan_video import model_fn_wan_video
        except Exception as exc:
            raise RuntimeError("DiffSynth-Studio is required for WorldTeacher.") from exc

        vace_context = None
        vace_scale = None
        if self._use_vace():
            if reference_latents is None:
                reference_latents = front_latents.unsqueeze(2)
            vace_context = self._build_vace_context(
                reference_latents=reference_latents,
                num_frames=num_frames,
            )
            vace_scale = float(getattr(self.config, "vace_scale", 1.0))

        for progress_id, timestep in enumerate(scheduler.timesteps):
            timestep = timestep.to(device=front_latents.device, dtype=front_latents.dtype)
            if timestep.dim() == 0:
                # DiffSynth Wan video expects a scalar/length-1 timestep tensor, not a batch-expanded one.
                timestep = timestep.unsqueeze(0)
            v_pred = model_fn_wan_video(
                dit=self.wan_dit,
                vace=self.wan_vace,
                latents=latents,
                timestep=timestep,
                context=context,
                fuse_vae_embedding_in_latents=True,
                vace_context=vace_context,
                vace_scale=vace_scale,
            )
            latents = scheduler.step(v_pred, timestep, latents, to_final=progress_id + 1 == len(scheduler.timesteps))
            latents[:, :, 0] = front_latents

        return latents[:, :, target_index]

    @torch.no_grad()
    def generate_actions(
        self,
        context: Optional[torch.Tensor] = None,
        steps: int = 10,
        vision_action_input_ids: Optional[torch.LongTensor] = None,
        vision_action_image_input: Optional[torch.Tensor] = None,
        vision_action_image_mask: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scheduler = self._get_scheduler()
        scheduler.set_timesteps(steps)

        action_head = self._get_action_head_type()
        if action_head == "vision_action":
            self._ensure_vision_action_modules()
            if vision_action_input_ids is None or vision_action_image_input is None or vision_action_image_mask is None:
                raise ValueError("VisionAction action generation requires vision_action_input_ids/vision_action_image_input/vision_action_image_mask.")
            enc = self.vision_action.forward_vlm(
                input_ids=vision_action_input_ids,
                pixel_values=vision_action_image_input,
                image_mask=vision_action_image_mask,
            )
            if (
                getattr(self.config, "vision_action_use_future_latent", True)
                and future_latents is not None
                and self.vision_action_future_proj is not None
            ):
                pooled_future = future_latents.mean(dim=(2, 3))
                pooled_future = pooled_future.to(
                    device=self.vision_action_future_proj.weight.device,
                    dtype=self.vision_action_future_proj.weight.dtype,
                )
                future_token = self.vision_action_future_proj(pooled_future).unsqueeze(1)
                enc["aux_visual_inputs"] = torch.cat(
                    [enc["aux_visual_inputs"], future_token.to(enc["aux_visual_inputs"].dtype)],
                    dim=1,
                )
            b = vision_action_input_ids.shape[0]
            if proprio is None:
                proprio = vision_action_image_input.new_zeros((b, getattr(self.vision_action.action_space, "dim_proprio", self.action_space.dim_action)))
            domain_id = torch.full(
                (b,),
                int(getattr(self.config, "vision_action_domain_id", 0)),
                dtype=torch.long,
                device=vision_action_image_input.device,
            )
            d = self.vision_action.action_space.dim_action
            x1 = torch.randn(
                b, self.config.num_actions, d, device=vision_action_image_input.device, dtype=vision_action_image_input.dtype
            )
            action = torch.zeros_like(x1)
            total_steps = max(1, int(steps))
            for i in range(total_steps, 0, -1):
                t = torch.full((b,), i / total_steps, device=x1.device, dtype=x1.dtype)
                x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
                proprio_m, x_t_m = self.vision_action.action_space.preprocess(proprio, x_t)
                action = self.vision_action.transformer(
                    domain_id=domain_id,
                    action_with_noise=x_t_m,
                    proprio=proprio_m,
                    t=t,
                    **enc,
                )
            return self.vision_action.action_space.postprocess(action)

        if context is None:
            raise ValueError("context is required for non-VisionAction action heads.")
        b = context.shape[0]
        use_decoder = self._ensure_action_decoder()
        if action_head in ("decoder", "separate", "independent") and use_decoder and self.action_decoder is not None:
            target_device = next(self.action_decoder.parameters()).device
            target_dtype = next(self.action_decoder.parameters()).dtype
            context = context.to(device=target_device, dtype=target_dtype)
            latents = torch.randn(
                b, self.config.num_actions, self.action_space.dim_action, device=target_device, dtype=target_dtype
            )
            for progress_id, timestep in enumerate(scheduler.timesteps):
                timestep = timestep.to(device=target_device, dtype=target_dtype)
                if timestep.dim() == 0:
                    timestep = timestep.unsqueeze(0).expand(b)
                v_pred = self.action_decoder(
                    latents,
                    t=timestep,
                    context=context,
                    proprio=None,
                )
                latents = scheduler.step(v_pred, timestep, latents, to_final=progress_id + 1 == len(scheduler.timesteps))
            return self.action_space.postprocess(latents)

        if self.wan_dit is not None:
            target_device = next(self.wan_dit.parameters()).device
            target_dtype = next(self.wan_dit.parameters()).dtype
            context = context.to(device=target_device, dtype=target_dtype)
        c = self.config.latent_dim
        hw = self.action_latent_hw
        latents = torch.randn(
            b, c, self.config.num_actions, hw, hw, device=context.device, dtype=context.dtype
        )
        try:
            from diffsynth.pipelines.wan_video import model_fn_wan_video
        except Exception as exc:
            raise RuntimeError("DiffSynth-Studio is required for WorldTeacher.") from exc

        for progress_id, timestep in enumerate(scheduler.timesteps):
            timestep = timestep.to(device=context.device, dtype=context.dtype)
            if timestep.dim() == 0:
                # Keep Wan video timestep scalar-like; batch expansion breaks DiffSynth broadcasting.
                timestep = timestep.unsqueeze(0)
            v_pred = model_fn_wan_video(
                dit=self.wan_dit,
                latents=latents,
                timestep=timestep,
                context=context,
                fuse_vae_embedding_in_latents=False,
            )
            latents = scheduler.step(v_pred, timestep, latents, to_final=progress_id + 1 == len(scheduler.timesteps))

        actions = self._latents_to_action(latents)
        return self.action_space.postprocess(actions)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        front_pixel_values: torch.Tensor,
        wrist_pixel_values: Optional[torch.Tensor] = None,
        wrist_valid_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        language_instruction: Optional[List[str]] = None,
        num_inference_steps: int = 50,
        action_steps: int = 10,
    ) -> Dict[str, torch.Tensor]:
        front_latents_full = self._encode_image_latents(front_pixel_values)
        front_latents = front_latents_full[:, :, 0]
        text_context = self._encode_text(input_ids, attention_mask)
        wrist_tokens = None
        if wrist_pixel_values is not None:
            wrist_tokens = self._encode_wrist_tokens(wrist_pixel_values, wrist_valid_mask=wrist_valid_mask)

        context_img = self._build_context(
            text_context=text_context,
            wrist_tokens=wrist_tokens,
            front_latents=front_latents,
            future_latents=None,
        )
        future_latents = self.sample_future_latents(
            front_latents=front_latents,
            context=context_img,
            num_inference_steps=num_inference_steps,
            reference_latents=front_latents_full,
        )

        future_prompt = future_latents if self.config.future_prompt_source != "none" else None
        action_head = self._get_action_head_type()
        if action_head == "vision_action":
            if language_instruction is None:
                raise ValueError("language_instruction is required when action_head_type='vision_action'.")
            vision_action_input_ids = self._encode_vision_action_text(language_instruction, device=front_pixel_values.device)
            vision_action_image_input, vision_action_image_mask = self._build_vision_action_image_inputs(
                front_pixel_values=front_pixel_values,
                wrist_pixel_values=wrist_pixel_values,
                wrist_valid_mask=wrist_valid_mask,
                device=front_pixel_values.device,
                dtype=front_pixel_values.dtype,
            )
            vision_action_image_input = vision_action_image_input.to(
                device=next(self.vision_action.parameters()).device,
                dtype=next(self.vision_action.parameters()).dtype,
            )
            vision_action_image_mask = vision_action_image_mask.to(device=vision_action_image_input.device)
            vision_action_input_ids = vision_action_input_ids.to(device=vision_action_image_input.device)
            if proprio is not None:
                proprio = proprio.to(device=vision_action_image_input.device, dtype=vision_action_image_input.dtype)
            actions = self.generate_actions(
                steps=action_steps,
                vision_action_input_ids=vision_action_input_ids,
                vision_action_image_input=vision_action_image_input,
                vision_action_image_mask=vision_action_image_mask,
                future_latents=future_prompt if self.config.use_future_token else None,
                proprio=proprio if self.config.use_proprio else None,
            )
        else:
            context_act = self._build_context(
                text_context=text_context,
                wrist_tokens=wrist_tokens,
                front_latents=front_latents,
                future_latents=future_prompt if self.config.use_future_token else None,
                proprio=proprio if self.config.use_proprio else None,
            )
            actions = self.generate_actions(context=context_act, steps=action_steps)
        future_pixels = self.decode_latents(future_latents)
        return {
            "future_pixels": future_pixels,
            "future_latents": future_latents,
            "actions": actions,
        }
