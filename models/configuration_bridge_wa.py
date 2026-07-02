from __future__ import annotations

from .configuration_vision_action import VisionActionConfig


class BridgeWAConfig(VisionActionConfig):
    model_type = "bridge_wa"

    def __init__(
        self,
        future_token_count: int = 4,
        future_token_pool_hw: int = 2,
        future_token_dim: int | None = None,
        change_token_count: int = 16,
        change_map_pool_hw: int = 4,
        change_token_dim: int | None = None,
        flow_token_count: int = 16,
        flow_map_pool_hw: int = 4,
        flow_token_dim: int | None = None,
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
        future_predictor_num_heads: int = 8,
        future_predictor_mlp_ratio: float = 2.0,
        change_predictor_num_heads: int = 8,
        change_predictor_mlp_ratio: float = 2.0,
        flow_predictor_num_heads: int = 8,
        flow_predictor_mlp_ratio: float = 2.0,
        future_distill_weight: float = 0.1,
        future_distill_cosine_weight: float = 0.1,
        change_distill_weight: float = 0.1,
        change_distill_cosine_weight: float = 0.1,
        flow_distill_weight: float = 0.1,
        flow_distill_cosine_weight: float = 0.1,
        bridge_lora_rank: int = 0,
        bridge_lora_alpha: float = 16.0,
        bridge_lora_dropout: float = 0.0,
        bridge_lora_last_n_blocks: int = 0,
        bridge_lora_only_bridge_layers: bool = False,
        bridge_lora_target_modules: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.future_token_count = future_token_count
        self.future_token_pool_hw = future_token_pool_hw
        self.future_token_dim = None if future_token_dim is None else int(future_token_dim)
        self.change_token_count = change_token_count
        self.change_map_pool_hw = change_map_pool_hw
        self.change_token_dim = None if change_token_dim is None else int(change_token_dim)
        self.flow_token_count = flow_token_count
        self.flow_map_pool_hw = flow_map_pool_hw
        self.flow_token_dim = None if flow_token_dim is None else int(flow_token_dim)
        self.bridge_num_injected_layers = bridge_num_injected_layers
        self.bridge_injection_start_layer = bridge_injection_start_layer
        self.bridge_inject_future = bool(bridge_inject_future)
        self.bridge_inject_change = bool(bridge_inject_change)
        self.bridge_inject_flow = bool(bridge_inject_flow)
        self.bridge_future_num_injected_layers = (
            None if bridge_future_num_injected_layers is None else int(bridge_future_num_injected_layers)
        )
        self.bridge_future_injection_start_layer = (
            None if bridge_future_injection_start_layer is None else int(bridge_future_injection_start_layer)
        )
        self.bridge_change_num_injected_layers = (
            None if bridge_change_num_injected_layers is None else int(bridge_change_num_injected_layers)
        )
        self.bridge_change_injection_start_layer = (
            None if bridge_change_injection_start_layer is None else int(bridge_change_injection_start_layer)
        )
        self.bridge_flow_num_injected_layers = (
            None if bridge_flow_num_injected_layers is None else int(bridge_flow_num_injected_layers)
        )
        self.bridge_flow_injection_start_layer = (
            None if bridge_flow_injection_start_layer is None else int(bridge_flow_injection_start_layer)
        )
        self.bridge_use_future_gate = bridge_use_future_gate
        self.bridge_use_change_gate = bridge_use_change_gate
        self.bridge_use_flow_gate = bridge_use_flow_gate
        self.bridge_change_bias_scale = bridge_change_bias_scale
        self.bridge_flow_bias_scale = bridge_flow_bias_scale
        self.bridge_modulation_scale = bridge_modulation_scale
        self.future_predictor_num_heads = future_predictor_num_heads
        self.future_predictor_mlp_ratio = future_predictor_mlp_ratio
        self.change_predictor_num_heads = change_predictor_num_heads
        self.change_predictor_mlp_ratio = change_predictor_mlp_ratio
        self.flow_predictor_num_heads = flow_predictor_num_heads
        self.flow_predictor_mlp_ratio = flow_predictor_mlp_ratio
        self.future_distill_weight = future_distill_weight
        self.future_distill_cosine_weight = future_distill_cosine_weight
        self.change_distill_weight = change_distill_weight
        self.change_distill_cosine_weight = change_distill_cosine_weight
        self.flow_distill_weight = flow_distill_weight
        self.flow_distill_cosine_weight = flow_distill_cosine_weight
        self.bridge_lora_rank = int(bridge_lora_rank)
        self.bridge_lora_alpha = float(bridge_lora_alpha)
        self.bridge_lora_dropout = float(bridge_lora_dropout)
        self.bridge_lora_last_n_blocks = int(bridge_lora_last_n_blocks)
        self.bridge_lora_only_bridge_layers = bool(bridge_lora_only_bridge_layers)
        self.bridge_lora_target_modules = list(
            bridge_lora_target_modules
            if bridge_lora_target_modules is not None
            else [
                "attn.q",
                "attn.qkv",
                "attn.kv_self",
                "attn.k_future",
                "attn.v_future",
                "attn.k_change",
                "attn.v_change",
                "attn.k_flow",
                "attn.v_flow",
                "attn.proj",
                "mlp.fc1",
                "mlp.fc2",
                "modulation.mlp.1",
                "modulation.mlp.3",
            ]
        )

    @classmethod
    def from_vision_action_config(cls, config: VisionActionConfig, **overrides) -> "BridgeWAConfig":
        data = config.to_dict()
        data.update(overrides)
        data.pop("model_type", None)
        return cls(**data)
