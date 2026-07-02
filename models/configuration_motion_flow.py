from __future__ import annotations

from .configuration_vision_action import VisionActionConfig


class MotionFlowConfig(VisionActionConfig):
    model_type = "motion_flow"

    def __init__(
        self,
        flow_token_count: int = 16,
        flow_map_pool_hw: int = 4,
        flow_num_injected_layers: int = 4,
        flow_injection_start_layer: int | None = None,
        flow_use_gate: bool = True,
        flow_predictor_num_heads: int = 8,
        flow_predictor_mlp_ratio: float = 2.0,
        flow_distill_weight: float = 0.1,
        flow_distill_cosine_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.flow_token_count = flow_token_count
        self.flow_map_pool_hw = flow_map_pool_hw
        self.flow_num_injected_layers = flow_num_injected_layers
        self.flow_injection_start_layer = flow_injection_start_layer
        self.flow_use_gate = flow_use_gate
        self.flow_predictor_num_heads = flow_predictor_num_heads
        self.flow_predictor_mlp_ratio = flow_predictor_mlp_ratio
        self.flow_distill_weight = flow_distill_weight
        self.flow_distill_cosine_weight = flow_distill_cosine_weight

    @classmethod
    def from_vision_action_config(cls, config: VisionActionConfig, **overrides) -> "MotionFlowConfig":
        data = config.to_dict()
        data.update(overrides)
        data.pop("model_type", None)
        return cls(**data)
