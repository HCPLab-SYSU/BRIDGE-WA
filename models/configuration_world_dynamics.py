from __future__ import annotations

from .configuration_vision_action import VisionActionConfig


class WorldDynamicsConfig(VisionActionConfig):
    model_type = "world_dynamics"

    def __init__(
        self,
        future_token_count: int = 4,
        future_token_pool_hw: int = 2,
        future_num_injected_layers: int = 4,
        future_injection_start_layer: int | None = None,
        future_use_gate: bool = True,
        future_predictor_num_heads: int = 8,
        future_predictor_mlp_ratio: float = 2.0,
        future_distill_weight: float = 0.1,
        future_distill_cosine_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.future_token_count = future_token_count
        self.future_token_pool_hw = future_token_pool_hw
        self.future_num_injected_layers = future_num_injected_layers
        self.future_injection_start_layer = future_injection_start_layer
        self.future_use_gate = future_use_gate
        self.future_predictor_num_heads = future_predictor_num_heads
        self.future_predictor_mlp_ratio = future_predictor_mlp_ratio
        self.future_distill_weight = future_distill_weight
        self.future_distill_cosine_weight = future_distill_cosine_weight

    @classmethod
    def from_vision_action_config(cls, config: VisionActionConfig, **overrides) -> "WorldDynamicsConfig":
        data = config.to_dict()
        data.update(overrides)
        data.pop("model_type", None)
        return cls(**data)
