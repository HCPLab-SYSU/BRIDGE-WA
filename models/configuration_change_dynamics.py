from __future__ import annotations

from .configuration_vision_action import VisionActionConfig


class ChangeDynamicsConfig(VisionActionConfig):
    model_type = "change_dynamics"

    def __init__(
        self,
        change_token_count: int = 16,
        change_map_pool_hw: int = 4,
        change_num_injected_layers: int = 4,
        change_injection_start_layer: int | None = None,
        change_use_gate: bool = True,
        change_predictor_num_heads: int = 8,
        change_predictor_mlp_ratio: float = 2.0,
        change_distill_weight: float = 0.1,
        change_distill_cosine_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.change_token_count = change_token_count
        self.change_map_pool_hw = change_map_pool_hw
        self.change_num_injected_layers = change_num_injected_layers
        self.change_injection_start_layer = change_injection_start_layer
        self.change_use_gate = change_use_gate
        self.change_predictor_num_heads = change_predictor_num_heads
        self.change_predictor_mlp_ratio = change_predictor_mlp_ratio
        self.change_distill_weight = change_distill_weight
        self.change_distill_cosine_weight = change_distill_cosine_weight

    @classmethod
    def from_vision_action_config(cls, config: VisionActionConfig, **overrides) -> "ChangeDynamicsConfig":
        data = config.to_dict()
        data.update(overrides)
        data.pop("model_type", None)
        return cls(**data)
