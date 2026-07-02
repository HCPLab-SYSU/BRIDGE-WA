from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .action_hub import build_action_space
from .configuration_bridge_wa import BridgeWAConfig
from .configuration_vision_action import VisionActionConfig
from .lora_utils import count_lora_parameters, replace_linear_with_lora
from .modeling_florence2 import Florence2ForConditionalGeneration
from .modeling_world_dynamics import FutureTokenPredictor, FutureTokenProjector
from .modeling_change_dynamics import ChangeMapPredictor, ChangeMapProjector
from .modeling_motion_flow import FlowMapPredictor, FlowMapProjector
from .modeling_vision_action import VisionAction
from .transformer import Attention, DomainAwareLinear, Mlp, basic_init, timestep_embedding


class BridgeWAAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_future_gate: bool = True,
        use_change_gate: bool = True,
        use_flow_gate: bool = True,
        change_bias_scale: float = 1.0,
        flow_bias_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim should be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.change_bias_scale = float(change_bias_scale)
        self.flow_bias_scale = float(flow_bias_scale)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_self = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.k_future = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_future = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_change = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_change = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_flow = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_flow = nn.Linear(dim, dim, bias=qkv_bias)
        self.future_gate = self._build_gate(dim) if use_future_gate else None
        self.change_gate = self._build_gate(dim) if use_change_gate else None
        self.flow_gate = self._build_gate(dim) if use_flow_gate else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(basic_init)

    @staticmethod
    def _build_gate(dim: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tokens, _ = x.shape
        return x.view(bsz, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _apply_gate(tokens: torch.Tensor, gate: Optional[nn.Module]) -> torch.Tensor:
        if gate is None:
            return tokens
        token_gate = torch.sigmoid(gate(tokens.mean(dim=1))).unsqueeze(1)
        return tokens * token_gate

    def forward(
        self,
        x: torch.Tensor,
        future_tokens: Optional[torch.Tensor] = None,
        change_tokens: Optional[torch.Tensor] = None,
        flow_tokens: Optional[torch.Tensor] = None,
        change_token_bias: Optional[torch.Tensor] = None,
        flow_token_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._reshape(self.q(x))
        kv_self = self.kv_self(x)
        k_self, v_self = kv_self.chunk(2, dim=-1)

        k_chunks = [self._reshape(k_self)]
        v_chunks = [self._reshape(v_self)]
        bias_chunks = [x.new_zeros((x.shape[0], x.shape[1]))]

        if future_tokens is not None and future_tokens.numel() > 0:
            future = self._apply_gate(future_tokens, self.future_gate)
            k_chunks.append(self._reshape(self.k_future(future)))
            v_chunks.append(self._reshape(self.v_future(future)))
            bias_chunks.append(x.new_zeros((x.shape[0], future.shape[1])))

        if change_tokens is not None and change_tokens.numel() > 0:
            change = self._apply_gate(change_tokens, self.change_gate)
            k_chunks.append(self._reshape(self.k_change(change)))
            v_chunks.append(self._reshape(self.v_change(change)))
            if change_token_bias is None:
                bias_chunks.append(x.new_zeros((x.shape[0], change.shape[1])))
            else:
                bias_chunks.append(self.change_bias_scale * change_token_bias.float())

        if flow_tokens is not None and flow_tokens.numel() > 0:
            flow = self._apply_gate(flow_tokens, self.flow_gate)
            k_chunks.append(self._reshape(self.k_flow(flow)))
            v_chunks.append(self._reshape(self.v_flow(flow)))
            if flow_token_bias is None:
                bias_chunks.append(x.new_zeros((x.shape[0], flow.shape[1])))
            else:
                bias_chunks.append(self.flow_bias_scale * flow_token_bias.float())

        k = torch.cat(k_chunks, dim=2)
        v = torch.cat(v_chunks, dim=2)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if bias_chunks:
            attn_bias = torch.cat(bias_chunks, dim=1).to(dtype=attn.dtype, device=attn.device)
            attn = attn + attn_bias[:, None, None, :]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim)
        out = self.proj(out)
        return self.proj_drop(out)


class BridgeWAModulation(nn.Module):
    def __init__(self, dim: int, scale: float = 0.1) -> None:
        super().__init__()
        self.scale = float(scale)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2),
        )
        self.apply(basic_init)

    def forward(
        self,
        x: torch.Tensor,
        future_tokens: Optional[torch.Tensor],
        change_tokens: Optional[torch.Tensor],
        flow_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ref = x.mean(dim=1)

        def summarize(tokens: Optional[torch.Tensor]) -> torch.Tensor:
            if tokens is None or tokens.numel() == 0:
                return torch.zeros_like(ref)
            return tokens.mean(dim=1)

        summary = torch.cat(
            [summarize(future_tokens), summarize(change_tokens), summarize(flow_tokens)],
            dim=-1,
        )
        gamma, beta = self.mlp(summary).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1.0 + self.scale * gamma) + self.scale * beta


class BridgeWATransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        inject_bridge: bool = False,
        use_future_gate: bool = True,
        use_change_gate: bool = True,
        use_flow_gate: bool = True,
        change_bias_scale: float = 1.0,
        flow_bias_scale: float = 1.0,
        modulation_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.inject_bridge = inject_bridge
        if inject_bridge:
            self.attn = BridgeWAAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                attn_drop=0.1,
                proj_drop=0.1,
                use_future_gate=use_future_gate,
                use_change_gate=use_change_gate,
                use_flow_gate=use_flow_gate,
                change_bias_scale=change_bias_scale,
                flow_bias_scale=flow_bias_scale,
            )
            self.modulation = BridgeWAModulation(hidden_size, scale=modulation_scale)
        else:
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
            self.modulation = None
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    def forward(
        self,
        x: torch.Tensor,
        future_tokens: Optional[torch.Tensor] = None,
        change_tokens: Optional[torch.Tensor] = None,
        flow_tokens: Optional[torch.Tensor] = None,
        change_token_bias: Optional[torch.Tensor] = None,
        flow_token_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.inject_bridge:
            x = x + self.attn(
                self.norm1(x),
                future_tokens=future_tokens,
                change_tokens=change_tokens,
                flow_tokens=flow_tokens,
                change_token_bias=change_token_bias,
                flow_token_bias=flow_token_bias,
            )
            x = self.modulation(x, future_tokens, change_tokens, flow_tokens)
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BridgeWASoftPromptedTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        multi_modal_input_size: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 20,
        dim_action: int = 20,
        dim_propio: int = 20,
        dim_time: int = 32,
        len_soft_prompts: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        bridge_num_injected_layers: int = 4,
        bridge_injection_start_layer: int | None = None,
        bridge_inject_future: bool = True,
        bridge_inject_change: bool = True,
        bridge_inject_flow: bool = True,
        bridge_future_num_injected_layers: int | None = None,
        bridge_future_injection_start_layer: int | None = None,
        bridge_change_num_injected_layers: int | None = None,
        bridge_change_injection_start_layer: int | None = None,
        bridge_flow_num_injected_layers: int | None = None,
        bridge_flow_injection_start_layer: int | None = None,
        bridge_use_future_gate: bool = True,
        bridge_use_change_gate: bool = True,
        bridge_use_flow_gate: bool = True,
        bridge_change_bias_scale: float = 1.0,
        bridge_flow_bias_scale: float = 1.0,
        bridge_modulation_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.len_soft_prompts = len_soft_prompts
        self.use_hetero_proj = use_hetero_proj
        self.future_injected_layers = self._resolve_injected_layers(
            depth=depth,
            enabled=bridge_inject_future,
            default_num_layers=bridge_num_injected_layers,
            default_start_layer=bridge_injection_start_layer,
            num_layers=bridge_future_num_injected_layers,
            start_layer=bridge_future_injection_start_layer,
        )
        self.change_injected_layers = self._resolve_injected_layers(
            depth=depth,
            enabled=bridge_inject_change,
            default_num_layers=bridge_num_injected_layers,
            default_start_layer=bridge_injection_start_layer,
            num_layers=bridge_change_num_injected_layers,
            start_layer=bridge_change_injection_start_layer,
        )
        self.flow_injected_layers = self._resolve_injected_layers(
            depth=depth,
            enabled=bridge_inject_flow,
            default_num_layers=bridge_num_injected_layers,
            default_start_layer=bridge_injection_start_layer,
            num_layers=bridge_flow_num_injected_layers,
            start_layer=bridge_flow_injection_start_layer,
        )
        self.bridge_injected_layers = (
            self.future_injected_layers | self.change_injected_layers | self.flow_injected_layers
        )

        self.blocks = nn.ModuleList(
            [
                BridgeWATransformerBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    inject_bridge=(idx in self.bridge_injected_layers),
                    use_future_gate=bridge_use_future_gate,
                    use_change_gate=bridge_use_change_gate,
                    use_flow_gate=bridge_use_flow_gate,
                    change_bias_scale=bridge_change_bias_scale,
                    flow_bias_scale=bridge_flow_bias_scale,
                    modulation_scale=bridge_modulation_scale,
                )
                for idx in range(depth)
            ]
        )

        if use_hetero_proj:
            self.vlm_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
            self.aux_visual_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
        else:
            self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
            self.aux_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)
        self.action_encoder = DomainAwareLinear(dim_action + dim_time + dim_propio, hidden_size, num_domains=num_domains)
        self.action_decoder = DomainAwareLinear(hidden_size, dim_action, num_domains=num_domains)

        if len_soft_prompts > 0:
            self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
            nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)

        self.apply(basic_init)

    @staticmethod
    def _resolve_injected_layers(
        *,
        depth: int,
        enabled: bool,
        default_num_layers: int,
        default_start_layer: int | None,
        num_layers: int | None,
        start_layer: int | None,
    ) -> set[int]:
        if not enabled:
            return set()
        resolved_num_layers = default_num_layers if num_layers is None else int(num_layers)
        if resolved_num_layers <= 0:
            return set()
        if start_layer is None:
            if default_start_layer is None:
                resolved_start_layer = max(0, depth - resolved_num_layers)
            else:
                resolved_start_layer = int(default_start_layer)
        else:
            resolved_start_layer = int(start_layer)
        resolved_start_layer = max(0, min(depth, resolved_start_layer))
        end_layer = max(resolved_start_layer, min(depth, resolved_start_layer + resolved_num_layers))
        return set(range(resolved_start_layer, end_layer))

    def forward(
        self,
        domain_id: torch.LongTensor,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        future_tokens: Optional[torch.Tensor] = None,
        change_tokens: Optional[torch.Tensor] = None,
        flow_tokens: Optional[torch.Tensor] = None,
        change_token_bias: Optional[torch.Tensor] = None,
        flow_token_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, num_actions = action_with_noise.shape[:2]
        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(bsz, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(bsz, num_actions, proprio.shape[-1])
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens, domain_id)

        if self.use_hetero_proj:
            x = torch.cat(
                [x, self.vlm_proj(vlm_features, domain_id), self.aux_visual_proj(aux_visual_inputs, domain_id)],
                dim=1,
            )
        else:
            x = torch.cat([x, self.vlm_proj(vlm_features), self.aux_visual_proj(aux_visual_inputs)], dim=1)

        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}.")
        x = x + self.pos_emb[:, :seq_len, :]

        if self.len_soft_prompts > 0:
            soft_prompts = self.soft_prompt_hub(domain_id).view(bsz, self.len_soft_prompts, self.hidden_size)
            x = torch.cat([x, soft_prompts], dim=1)

        for idx, block in enumerate(self.blocks):
            if idx in self.bridge_injected_layers:
                x = block(
                    x,
                    future_tokens=future_tokens if idx in self.future_injected_layers else None,
                    change_tokens=change_tokens if idx in self.change_injected_layers else None,
                    flow_tokens=flow_tokens if idx in self.flow_injected_layers else None,
                    change_token_bias=change_token_bias if idx in self.change_injected_layers else None,
                    flow_token_bias=flow_token_bias if idx in self.flow_injected_layers else None,
                )
            else:
                x = block(x)

        return self.action_decoder(self.norm(x[:, :num_actions]), domain_id)


class BridgeWA(PreTrainedModel):
    config_class = BridgeWAConfig
    base_model_prefix = "bridge_wa"
    supports_gradient_checkpointing = True

    def __init__(self, config: BridgeWAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.num_actions = config.num_actions
        self.use_proprio = config.use_proprio
        self.action_mode = config.action_mode.lower()
        if self.action_mode == "auto":
            self.action_space = build_action_space(
                self.action_mode,
                real_dim=config.real_action_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(self.action_mode)
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        self.vlm = Florence2ForConditionalGeneration(config.florence_config).to(torch.float32)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        self.transformer = BridgeWASoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
            bridge_num_injected_layers=config.bridge_num_injected_layers,
            bridge_injection_start_layer=config.bridge_injection_start_layer,
            bridge_inject_future=config.bridge_inject_future,
            bridge_inject_change=config.bridge_inject_change,
            bridge_inject_flow=config.bridge_inject_flow,
            bridge_future_num_injected_layers=config.bridge_future_num_injected_layers,
            bridge_future_injection_start_layer=config.bridge_future_injection_start_layer,
            bridge_change_num_injected_layers=config.bridge_change_num_injected_layers,
            bridge_change_injection_start_layer=config.bridge_change_injection_start_layer,
            bridge_flow_num_injected_layers=config.bridge_flow_num_injected_layers,
            bridge_flow_injection_start_layer=config.bridge_flow_injection_start_layer,
            bridge_use_future_gate=config.bridge_use_future_gate,
            bridge_use_change_gate=config.bridge_use_change_gate,
            bridge_use_flow_gate=config.bridge_use_flow_gate,
            bridge_change_bias_scale=config.bridge_change_bias_scale,
            bridge_flow_bias_scale=config.bridge_flow_bias_scale,
            bridge_modulation_scale=config.bridge_modulation_scale,
        )

        self.future_token_projector = FutureTokenProjector(
            in_channels=48,
            out_dim=config.hidden_size,
            token_count=config.future_token_count,
            pool_hw=config.future_token_pool_hw,
            token_dim=config.future_token_dim,
        )
        self.future_token_predictor = FutureTokenPredictor(
            input_dim=projection_dim,
            hidden_size=config.hidden_size,
            token_count=config.future_token_count,
            num_heads=config.future_predictor_num_heads,
            mlp_ratio=config.future_predictor_mlp_ratio,
            token_dim=config.future_token_dim,
        )
        self.change_map_projector = ChangeMapProjector(
            out_dim=config.hidden_size,
            token_count=config.change_token_count,
            pool_hw=config.change_map_pool_hw,
            token_dim=config.change_token_dim,
        )
        self.change_map_predictor = ChangeMapPredictor(
            input_dim=projection_dim,
            hidden_size=config.hidden_size,
            token_count=config.change_token_count,
            pool_hw=config.change_map_pool_hw,
            num_heads=config.change_predictor_num_heads,
            mlp_ratio=config.change_predictor_mlp_ratio,
            token_dim=config.change_token_dim,
        )
        self.flow_map_projector = FlowMapProjector(
            out_dim=config.hidden_size,
            token_count=config.flow_token_count,
            pool_hw=config.flow_map_pool_hw,
            token_dim=config.flow_token_dim,
        )
        self.flow_map_predictor = FlowMapPredictor(
            input_dim=projection_dim,
            hidden_size=config.hidden_size,
            token_count=config.flow_token_count,
            pool_hw=config.flow_map_pool_hw,
            num_heads=config.flow_predictor_num_heads,
            mlp_ratio=config.flow_predictor_mlp_ratio,
            token_dim=config.flow_token_dim,
        )
        self._apply_bridge_lora_if_needed(config)
        self.app = None

    def _apply_bridge_lora_if_needed(self, config: BridgeWAConfig) -> None:
        rank = int(getattr(config, "bridge_lora_rank", 0) or 0)
        if rank <= 0:
            return
        total_blocks = len(self.transformer.blocks)
        last_n = int(getattr(config, "bridge_lora_last_n_blocks", 0) or 0)
        if last_n <= 0 or last_n > total_blocks:
            last_n = total_blocks
        start = max(0, total_blocks - last_n)
        only_bridge_layers = bool(getattr(config, "bridge_lora_only_bridge_layers", False))
        target_suffixes = tuple(getattr(config, "bridge_lora_target_modules", ()) or ())
        replaced = 0
        for idx, block in enumerate(self.transformer.blocks):
            if idx < start:
                continue
            if only_bridge_layers and idx not in getattr(self.transformer, "bridge_injected_layers", set()):
                continue
            replaced += replace_linear_with_lora(
                block,
                target_suffixes=target_suffixes,
                r=rank,
                lora_alpha=float(getattr(config, "bridge_lora_alpha", 16.0)),
                lora_dropout=float(getattr(config, "bridge_lora_dropout", 0.0)),
            )
        lora_total, _ = count_lora_parameters(self.transformer)
        print(
            f"[BridgeWA] Applied LoRA rank={rank} to {replaced} linear modules "
            f"across last {last_n} transformer blocks (lora_params={lora_total})."
        )

    @classmethod
    def from_vision_action_pretrained(cls, pretrained_model_name_or_path: str, **config_overrides) -> "BridgeWA":
        base_cfg = VisionActionConfig.from_pretrained(pretrained_model_name_or_path)
        cfg = BridgeWAConfig.from_vision_action_config(base_cfg, **config_overrides)
        model = cls(cfg)
        base_model = VisionAction.from_pretrained(pretrained_model_name_or_path)
        missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
        model._initialize_guided_attention_from_base(base_model)
        if missing:
            print(
                "[BridgeWA] Missing keys while loading VisionAction base:",
                missing[:12],
                "..." if len(missing) > 12 else "",
            )
        if unexpected:
            print(
                "[BridgeWA] Unexpected keys while loading VisionAction base:",
                unexpected[:12],
                "..." if len(unexpected) > 12 else "",
            )
        return model

    @torch.no_grad()
    def _initialize_guided_attention_from_base(self, base_model: VisionAction) -> None:
        for idx in getattr(self.transformer, "bridge_injected_layers", set()):
            if idx >= len(base_model.transformer.blocks) or idx >= len(self.transformer.blocks):
                continue
            base_attn = getattr(base_model.transformer.blocks[idx], "attn", None)
            bridge_attn = getattr(self.transformer.blocks[idx], "attn", None)
            if base_attn is None or bridge_attn is None or not hasattr(base_attn, "qkv"):
                continue
            q_weight, k_weight, v_weight = base_attn.qkv.weight.chunk(3, dim=0)
            bridge_attn.q.weight.copy_(q_weight)
            bridge_attn.kv_self.weight.copy_(torch.cat([k_weight, v_weight], dim=0))
            bridge_attn.k_future.weight.copy_(k_weight)
            bridge_attn.v_future.weight.copy_(v_weight)
            bridge_attn.k_change.weight.copy_(k_weight)
            bridge_attn.v_change.weight.copy_(v_weight)
            bridge_attn.k_flow.weight.copy_(k_weight)
            bridge_attn.v_flow.weight.copy_(v_weight)
            if base_attn.qkv.bias is not None:
                q_bias, k_bias, v_bias = base_attn.qkv.bias.chunk(3, dim=0)
                bridge_attn.q.bias.copy_(q_bias)
                bridge_attn.kv_self.bias.copy_(torch.cat([k_bias, v_bias], dim=0))
                bridge_attn.k_future.bias.copy_(k_bias)
                bridge_attn.v_future.bias.copy_(v_bias)
                bridge_attn.k_change.bias.copy_(k_bias)
                bridge_attn.v_change.bias.copy_(v_bias)
                bridge_attn.k_flow.bias.copy_(k_bias)
                bridge_attn.v_flow.bias.copy_(v_bias)
            if hasattr(base_attn, "proj"):
                bridge_attn.proj.weight.copy_(base_attn.proj.weight)
                if base_attn.proj.bias is not None:
                    bridge_attn.proj.bias.copy_(base_attn.proj.bias)

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz, views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")
        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm._encode_image(valid_images)
        n_tokens, dim = valid_feats.shape[1:]
        image_features = valid_feats.new_zeros((bsz * views, n_tokens, dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(bsz, views, n_tokens, dim)

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],
            inputs_embeds,
        )
        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]
        aux_visual_inputs = image_features[:, 1:].reshape(bsz, -1, dim)
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    def predict_future_tokens(self, vlm_features: torch.Tensor, aux_visual_inputs: torch.Tensor) -> torch.Tensor:
        return self.future_token_predictor(vlm_features, aux_visual_inputs)

    def predict_change_map(self, vlm_features: torch.Tensor, aux_visual_inputs: torch.Tensor) -> torch.Tensor:
        return self.change_map_predictor(vlm_features, aux_visual_inputs)

    def predict_flow_map(self, vlm_features: torch.Tensor, aux_visual_inputs: torch.Tensor) -> torch.Tensor:
        return self.flow_map_predictor(vlm_features, aux_visual_inputs)

    def project_teacher_future_latents(self, future_latents: torch.Tensor) -> torch.Tensor:
        return self.future_token_projector(future_latents)

    def project_change_map(self, change_map: torch.Tensor) -> torch.Tensor:
        return self.change_map_projector(change_map)

    def project_flow_map(self, flow_map: torch.Tensor) -> torch.Tensor:
        return self.flow_map_projector(flow_map)

    def build_change_token_bias(self, change_map: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(
            change_map.float(),
            (self.config.change_map_pool_hw, self.config.change_map_pool_hw),
        )
        return pooled.flatten(1)

    def build_flow_token_bias(self, flow_map: torch.Tensor) -> torch.Tensor:
        flow_mag = torch.linalg.vector_norm(flow_map.float(), dim=1, keepdim=True)
        pooled = F.adaptive_avg_pool2d(
            flow_mag,
            (self.config.flow_map_pool_hw, self.config.flow_map_pool_hw),
        )
        return pooled.flatten(1)

    @staticmethod
    def _select_guidance_tensor(
        predicted_tensor: torch.Tensor,
        teacher_tensor: Optional[torch.Tensor] = None,
        source: str = "predicted",
        blend_ratio: float = 0.5,
    ) -> torch.Tensor:
        source = str(source).lower()
        if teacher_tensor is None or source == "predicted":
            return predicted_tensor
        if source == "teacher":
            return teacher_tensor
        if source == "blend":
            ratio = float(max(0.0, min(1.0, blend_ratio)))
            return teacher_tensor * ratio + predicted_tensor * (1.0 - ratio)
        raise ValueError(f"Unsupported guidance source: {source}")

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        future_latents: Optional[torch.Tensor] = None,
        change_map: Optional[torch.Tensor] = None,
        flow_map: Optional[torch.Tensor] = None,
        guidance_source: str = "predicted",
        guidance_blend_ratio: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        enc = self.forward_vlm(input_ids, image_input, image_mask)
        predicted_future_tokens = self.predict_future_tokens(enc["vlm_features"], enc["aux_visual_inputs"])
        predicted_change_map = self.predict_change_map(enc["vlm_features"], enc["aux_visual_inputs"])
        predicted_change_tokens = self.project_change_map(predicted_change_map)
        predicted_flow_map = self.predict_flow_map(enc["vlm_features"], enc["aux_visual_inputs"])
        predicted_flow_tokens = self.project_flow_map(predicted_flow_map)
        predicted_change_bias = self.build_change_token_bias(predicted_change_map)
        predicted_flow_bias = self.build_flow_token_bias(predicted_flow_map)

        teacher_future_tokens = None
        teacher_change_lowres = None
        teacher_flow_lowres = None
        if future_latents is not None:
            teacher_future_tokens = self.project_teacher_future_latents(future_latents)
        if change_map is not None:
            teacher_change_lowres = F.adaptive_avg_pool2d(
                change_map.float(),
                (self.config.change_map_pool_hw, self.config.change_map_pool_hw),
            )
        if flow_map is not None:
            teacher_flow_lowres = F.adaptive_avg_pool2d(
                flow_map.float(),
                (self.config.flow_map_pool_hw, self.config.flow_map_pool_hw),
            )
        teacher_change_tokens = self.project_change_map(change_map) if change_map is not None else None
        teacher_flow_tokens = self.project_flow_map(flow_map) if flow_map is not None else None
        teacher_change_bias = self.build_change_token_bias(change_map) if change_map is not None else None
        teacher_flow_bias = self.build_flow_token_bias(flow_map) if flow_map is not None else None

        injected_future_tokens = self._select_guidance_tensor(
            predicted_tensor=predicted_future_tokens,
            teacher_tensor=teacher_future_tokens,
            source=guidance_source,
            blend_ratio=guidance_blend_ratio,
        )
        injected_change_tokens = self._select_guidance_tensor(
            predicted_tensor=predicted_change_tokens,
            teacher_tensor=teacher_change_tokens,
            source=guidance_source,
            blend_ratio=guidance_blend_ratio,
        )
        injected_flow_tokens = self._select_guidance_tensor(
            predicted_tensor=predicted_flow_tokens,
            teacher_tensor=teacher_flow_tokens,
            source=guidance_source,
            blend_ratio=guidance_blend_ratio,
        )
        injected_change_bias = self._select_guidance_tensor(
            predicted_tensor=predicted_change_bias,
            teacher_tensor=teacher_change_bias,
            source=guidance_source,
            blend_ratio=guidance_blend_ratio,
        )
        injected_flow_bias = self._select_guidance_tensor(
            predicted_tensor=predicted_flow_bias,
            teacher_tensor=teacher_flow_bias,
            source=guidance_source,
            blend_ratio=guidance_blend_ratio,
        )

        bsz = input_ids.shape[0]
        t = (torch.rand(1, device=input_ids.device) + torch.arange(bsz, device=input_ids.device) / bsz) % (1 - 1e-5)
        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            future_tokens=injected_future_tokens,
            change_tokens=injected_change_tokens,
            flow_tokens=injected_flow_tokens,
            change_token_bias=injected_change_bias,
            flow_token_bias=injected_flow_bias,
            **enc,
        )
        loss_dict = self.action_space.compute_loss(pred_action, action)
        loss_dict_raw = self.action_space.compute_loss_raw(pred_action, action)

        outputs: Dict[str, Any] = {
            **loss_dict,
            "pred_action": pred_action,
            "action_loss_raw": loss_dict_raw,
            "predicted_future_tokens": predicted_future_tokens,
            "predicted_change_map": predicted_change_map,
            "predicted_change_tokens": predicted_change_tokens,
            "predicted_flow_map": predicted_flow_map,
            "predicted_flow_tokens": predicted_flow_tokens,
        }
        if teacher_future_tokens is not None:
            predicted_future_tokens_f = predicted_future_tokens.float()
            teacher_future_tokens_f = teacher_future_tokens.float()
            future_mse = F.mse_loss(predicted_future_tokens_f, teacher_future_tokens_f)
            future_dir_loss = (
                1.0
                - F.cosine_similarity(
                    F.normalize(predicted_future_tokens_f, dim=-1),
                    F.normalize(teacher_future_tokens_f, dim=-1),
                    dim=-1,
                )
            ).mean()
            future_norm_loss = F.smooth_l1_loss(
                torch.log(predicted_future_tokens_f.norm(dim=-1) + 1e-6),
                torch.log(teacher_future_tokens_f.norm(dim=-1) + 1e-6),
            )
            future_cos = 1.0 - F.cosine_similarity(
                predicted_future_tokens_f.reshape(predicted_future_tokens.shape[0], -1),
                teacher_future_tokens_f.reshape(teacher_future_tokens.shape[0], -1),
                dim=1,
            ).mean()
            outputs["future_token_distill_loss"] = (
                float(self.config.future_distill_weight) * (future_dir_loss + future_norm_loss)
                + float(self.config.future_distill_cosine_weight) * future_cos
            )
            outputs["future_token_mse"] = future_mse.detach()
            outputs["future_token_dir_loss"] = future_dir_loss.detach()
            outputs["future_token_norm_loss"] = future_norm_loss.detach()
            outputs["future_token_cosine"] = (1.0 - future_cos).detach()
        if teacher_change_lowres is not None:
            change_mse = F.mse_loss(predicted_change_map.float(), teacher_change_lowres.float())
            change_cosine = F.cosine_similarity(
                predicted_change_map.float().reshape(predicted_change_map.shape[0], -1),
                teacher_change_lowres.float().reshape(teacher_change_lowres.shape[0], -1),
                dim=1,
            ).mean()
            outputs["change_map_distill_loss"] = (
                float(self.config.change_distill_weight) * change_mse
                + float(self.config.change_distill_cosine_weight) * (1.0 - change_cosine)
            )
            outputs["change_map_mse"] = change_mse.detach()
            outputs["change_map_cosine"] = change_cosine.detach()
        if teacher_flow_lowres is not None:
            flow_mse = F.mse_loss(predicted_flow_map.float(), teacher_flow_lowres.float())
            flow_cosine = F.cosine_similarity(
                predicted_flow_map.float().reshape(predicted_flow_map.shape[0], -1),
                teacher_flow_lowres.float().reshape(teacher_flow_lowres.shape[0], -1),
                dim=1,
            ).mean()
            outputs["flow_map_distill_loss"] = (
                float(self.config.flow_distill_weight) * flow_mse
                + float(self.config.flow_distill_cosine_weight) * (1.0 - flow_cosine)
            )
            outputs["flow_map_mse"] = flow_mse.detach()
            outputs["flow_map_cosine"] = flow_cosine.detach()
        return outputs

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        self.eval()
        enc = self.forward_vlm(input_ids, image_input, image_mask)
        predicted_future_tokens = self.predict_future_tokens(enc["vlm_features"], enc["aux_visual_inputs"])
        predicted_change_map = self.predict_change_map(enc["vlm_features"], enc["aux_visual_inputs"])
        predicted_change_tokens = self.project_change_map(predicted_change_map)
        predicted_flow_map = self.predict_flow_map(enc["vlm_features"], enc["aux_visual_inputs"])
        predicted_flow_tokens = self.project_flow_map(predicted_flow_map)
        predicted_change_bias = self.build_change_token_bias(predicted_change_map)
        predicted_flow_bias = self.build_flow_token_bias(predicted_flow_map)

        bsz = input_ids.shape[0]
        dim_action = self.action_space.dim_action
        x1 = torch.randn(bsz, self.num_actions, dim_action, device=proprio.device, dtype=proprio.dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((bsz,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                future_tokens=predicted_future_tokens,
                change_tokens=predicted_change_tokens,
                flow_tokens=predicted_flow_tokens,
                change_token_bias=predicted_change_bias,
                flow_token_bias=predicted_flow_bias,
                **enc,
            )
        return self.action_space.postprocess(action)
