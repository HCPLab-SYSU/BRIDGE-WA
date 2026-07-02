from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class ResidualLinear(nn.Module):
    """
    Frozen base linear + trainable residual delta.

    This keeps the bridge checkpoint readout as an explicit anchor and only
    learns a delta on top, which is the intended residual-adapt regime.
    """

    def __init__(self, base_linear: nn.Linear, zero_init: bool = True):
        super().__init__()
        self.base = copy.deepcopy(base_linear)
        self.base.requires_grad_(False)

        self.delta = nn.Linear(
            base_linear.in_features,
            base_linear.out_features,
            bias=base_linear.bias is not None,
        )
        if zero_init:
            nn.init.zeros_(self.delta.weight)
            if self.delta.bias is not None:
                nn.init.zeros_(self.delta.bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.delta.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.delta.bias

    @property
    def in_features(self) -> int:
        return self.delta.in_features

    @property
    def out_features(self) -> int:
        return self.delta.out_features

    def requires_grad_(self, requires_grad: bool = True):
        self.delta.requires_grad_(requires_grad)
        self.base.requires_grad_(False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.delta(x)


def install_residual_action_head(model, zero_init: bool = True, logger=None) -> bool:
    latent_to_action = getattr(model, "latent_to_action", None)
    if latent_to_action is None:
        raise AttributeError("Model does not expose latent_to_action.")
    if isinstance(latent_to_action, ResidualLinear):
        latent_to_action.base.requires_grad_(False)
        return False
    if not isinstance(latent_to_action, nn.Linear):
        raise TypeError(f"Expected latent_to_action to be nn.Linear, got {type(latent_to_action)!r}")

    residual = ResidualLinear(latent_to_action, zero_init=zero_init)
    residual = residual.to(
        device=latent_to_action.weight.device,
        dtype=latent_to_action.weight.dtype,
    )
    model.latent_to_action = residual
    setattr(model.config, "residual_action_head", True)
    if logger is not None:
        logger.info(
            "[residual] installed residual latent_to_action head "
            "(frozen bridge base + trainable delta, zero_init=%s)",
            bool(zero_init),
        )
    return True
