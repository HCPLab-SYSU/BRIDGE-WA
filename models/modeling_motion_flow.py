from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .action_hub import build_action_space
from .configuration_motion_flow import MotionFlowConfig
from .configuration_vision_action import VisionActionConfig
from .modeling_florence2 import Florence2ForConditionalGeneration
from .modeling_vision_action import VisionAction
from .transformer import Attention, DomainAwareLinear, Mlp, basic_init, timestep_embedding


class FlowMapProjector(nn.Module):
    def __init__(
        self,
        out_dim: int,
        token_count: int = 16,
        pool_hw: int = 4,
        token_dim: int | None = None,
    ) -> None:
        super().__init__()
        if pool_hw * pool_hw != token_count:
            raise ValueError("flow_token_count must equal flow_map_pool_hw ** 2.")
        token_dim = int(token_dim or out_dim)
        self.token_count = token_count
        self.pool_hw = pool_hw
        self.value_proj = nn.Linear(2, token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.out_proj = nn.Identity() if token_dim == out_dim else nn.Linear(token_dim, out_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, token_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(basic_init)

    def forward(self, flow_map: torch.Tensor) -> torch.Tensor:
        x = flow_map.float()
        if x.dim() != 4 or x.shape[1] != 2:
            raise ValueError(f"Expected flow map [B,2,H,W], got {tuple(flow_map.shape)}")
        x = F.adaptive_avg_pool2d(x, (self.pool_hw, self.pool_hw))
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = x.to(device=self.value_proj.weight.device, dtype=self.value_proj.weight.dtype)
        x = self.value_proj(x)
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        x = self.token_norm(x)
        return self.out_proj(self.token_mlp(x))


class FlowMapPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        token_count: int,
        pool_hw: int,
        num_heads: int,
        mlp_ratio: float,
        token_dim: int | None = None,
    ) -> None:
        super().__init__()
        if pool_hw * pool_hw != token_count:
            raise ValueError("flow_token_count must equal flow_map_pool_hw ** 2.")
        token_dim = int(token_dim or hidden_size)
        if token_dim % num_heads != 0:
            raise ValueError("flow token_dim must be divisible by flow_predictor_num_heads.")
        self.token_count = token_count
        self.pool_hw = pool_hw
        self.context_proj = nn.Linear(input_dim, token_dim)
        self.query_tokens = nn.Parameter(torch.zeros(1, token_count, token_dim))
        self.query_norm = nn.LayerNorm(token_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = Mlp(
            in_features=token_dim,
            hidden_features=int(token_dim * mlp_ratio),
            out_features=token_dim,
            drop=0.0,
        )
        self.out_norm = nn.LayerNorm(token_dim)
        self.out_proj = nn.Linear(token_dim, 2)
        nn.init.normal_(self.query_tokens, std=0.02)
        self.apply(basic_init)

    def forward(self, vlm_features: torch.Tensor, aux_visual_inputs: torch.Tensor) -> torch.Tensor:
        context = torch.cat([vlm_features, aux_visual_inputs], dim=1)
        context = self.context_proj(context)
        context_n = self.context_norm(context)
        queries = self.query_tokens.expand(vlm_features.shape[0], -1, -1)
        q = self.query_norm(queries)
        attn_out, _ = self.cross_attn(q, context_n, context_n, need_weights=False)
        x = queries + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        x = self.out_proj(self.out_norm(x))
        return x.transpose(1, 2).reshape(vlm_features.shape[0], 2, self.pool_hw, self.pool_hw)


class FlowConditionedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim should be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_self = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.k_flow = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_flow = nn.Linear(dim, dim, bias=qkv_bias)
        self.gate = None
        if use_gate:
            self.gate = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(basic_init)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tokens, dim = x.shape
        return x.view(bsz, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def forward(self, x: torch.Tensor, flow_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self._reshape(self.q(x))
        kv_self = self.kv_self(x)
        k_self, v_self = kv_self.chunk(2, dim=-1)
        k = self._reshape(k_self)
        v = self._reshape(v_self)

        if flow_tokens is not None and flow_tokens.numel() > 0:
            cond = flow_tokens
            if self.gate is not None:
                gate = torch.sigmoid(self.gate(cond.mean(dim=1))).unsqueeze(1)
                cond = cond * gate
            k_flow = self._reshape(self.k_flow(cond))
            v_flow = self._reshape(self.v_flow(cond))
            k = torch.cat([k, k_flow], dim=2)
            v = torch.cat([v, v_flow], dim=2)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim)
        out = self.proj(out)
        return self.proj_drop(out)


class FlowTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        inject_flow: bool = False,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.inject_flow = inject_flow
        if inject_flow:
            self.attn = FlowConditionedAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                attn_drop=0.1,
                proj_drop=0.1,
                use_gate=use_gate,
            )
        else:
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    def forward(self, x: torch.Tensor, flow_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.inject_flow:
            x = x + self.attn(self.norm1(x), flow_tokens=flow_tokens)
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FlowSoftPromptedTransformer(nn.Module):
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
        flow_num_injected_layers: int = 4,
        flow_injection_start_layer: int | None = None,
        flow_use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.len_soft_prompts = len_soft_prompts
        self.use_hetero_proj = use_hetero_proj
        if flow_injection_start_layer is None:
            flow_injection_start_layer = max(0, depth - flow_num_injected_layers)
        end_layer = min(depth, flow_injection_start_layer + flow_num_injected_layers)
        self.flow_injected_layers = set(range(max(0, flow_injection_start_layer), end_layer))

        self.blocks = nn.ModuleList(
            [
                FlowTransformerBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    inject_flow=(idx in self.flow_injected_layers),
                    use_gate=flow_use_gate,
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

    def forward(
        self,
        domain_id: torch.LongTensor,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        flow_tokens: Optional[torch.Tensor] = None,
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
            block_flow = flow_tokens if idx in self.flow_injected_layers else None
            x = block(x, flow_tokens=block_flow)

        return self.action_decoder(self.norm(x[:, :num_actions]), domain_id)


class MotionFlow(PreTrainedModel):
    config_class = MotionFlowConfig
    base_model_prefix = "motion_flow"
    supports_gradient_checkpointing = True

    def __init__(self, config: MotionFlowConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.num_actions = config.num_actions
        self.use_proprio = config.use_proprio
        self.action_mode = config.action_mode.lower()
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

        self.transformer = FlowSoftPromptedTransformer(
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
            flow_num_injected_layers=config.flow_num_injected_layers,
            flow_injection_start_layer=config.flow_injection_start_layer,
            flow_use_gate=config.flow_use_gate,
        )

        self.flow_map_projector = FlowMapProjector(
            out_dim=config.hidden_size,
            token_count=config.flow_token_count,
            pool_hw=config.flow_map_pool_hw,
        )
        self.flow_map_predictor = FlowMapPredictor(
            input_dim=projection_dim,
            hidden_size=config.hidden_size,
            token_count=config.flow_token_count,
            pool_hw=config.flow_map_pool_hw,
            num_heads=config.flow_predictor_num_heads,
            mlp_ratio=config.flow_predictor_mlp_ratio,
        )
        self.app = None

    @classmethod
    def from_vision_action_pretrained(cls, pretrained_model_name_or_path: str, **config_overrides) -> "MotionFlow":
        base_cfg = VisionActionConfig.from_pretrained(pretrained_model_name_or_path)
        cfg = MotionFlowConfig.from_vision_action_config(base_cfg, **config_overrides)
        model = cls(cfg)
        base_model = VisionAction.from_pretrained(pretrained_model_name_or_path)
        missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
        if missing:
            print("[MotionFlow] Missing keys while loading VisionAction base:", missing[:12], "..." if len(missing) > 12 else "")
        if unexpected:
            print("[MotionFlow] Unexpected keys while loading VisionAction base:", unexpected[:12], "..." if len(unexpected) > 12 else "")
        return model

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

    def predict_flow_map(self, vlm_features: torch.Tensor, aux_visual_inputs: torch.Tensor) -> torch.Tensor:
        return self.flow_map_predictor(vlm_features, aux_visual_inputs)

    def project_flow_map(self, flow_map: torch.Tensor) -> torch.Tensor:
        return self.flow_map_projector(flow_map)

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        flow_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc = self.forward_vlm(input_ids, image_input, image_mask)
        predicted_flow_map = self.predict_flow_map(enc["vlm_features"], enc["aux_visual_inputs"])
        predicted_flow_tokens = self.project_flow_map(predicted_flow_map)

        teacher_flow_lowres = None
        if flow_map is not None:
            teacher_flow_lowres = F.adaptive_avg_pool2d(flow_map.float(), (self.config.flow_map_pool_hw, self.config.flow_map_pool_hw))

        bsz = input_ids.shape[0]
        t = (torch.rand(1, device=input_ids.device) + torch.arange(bsz, device=input_ids.device) / bsz) % (1 - 1e-5)
        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            flow_tokens=predicted_flow_tokens,
            **enc,
        )
        loss_dict = self.action_space.compute_loss(pred_action, action)
        loss_dict_raw = self.action_space.compute_loss_raw(pred_action, action)

        outputs: Dict[str, Any] = {
            **loss_dict,
            "predicted_flow_map": predicted_flow_map,
            "predicted_flow_tokens": predicted_flow_tokens,
            "pred_action": pred_action,
            "action_loss_raw": loss_dict_raw,
        }
        if teacher_flow_lowres is not None:
            mse = F.mse_loss(predicted_flow_map.float(), teacher_flow_lowres.float())
            cosine = F.cosine_similarity(
                predicted_flow_map.float().reshape(predicted_flow_map.shape[0], -1),
                teacher_flow_lowres.float().reshape(teacher_flow_lowres.shape[0], -1),
                dim=1,
            ).mean()
            cos_loss = 1.0 - cosine
            outputs["flow_map_distill_loss"] = (
                float(self.config.flow_distill_weight) * mse
                + float(self.config.flow_distill_cosine_weight) * cos_loss
            )
            outputs["teacher_flow_map_lowres"] = teacher_flow_lowres
            outputs["flow_map_mse"] = mse.detach()
            outputs["flow_map_cosine"] = cosine.detach()
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
        predicted_flow_map = self.predict_flow_map(enc["vlm_features"], enc["aux_visual_inputs"])
        predicted_flow_tokens = self.project_flow_map(predicted_flow_map)

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
                flow_tokens=predicted_flow_tokens,
                **enc,
            )
        return self.action_space.postprocess(action)
