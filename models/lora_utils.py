from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0
        self.lora_dropout = nn.Dropout(float(lora_dropout))
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.empty(self.r, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, self.r))
            self.reset_lora_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def reset_lora_parameters(self) -> None:
        if self.r <= 0:
            return
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> "LoRALinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        module.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.data)
        return module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.linear(input, self.weight, self.bias)
        if self.r > 0:
            delta = F.linear(F.linear(self.lora_dropout(input), self.lora_A), self.lora_B)
            out = out + delta * self.scaling
        return out


def replace_linear_with_lora(
    root: nn.Module,
    *,
    target_suffixes: Sequence[str],
    r: int,
    lora_alpha: float,
    lora_dropout: float,
    prefix: str = "",
) -> int:
    replaced = 0
    suffixes = tuple(str(s) for s in target_suffixes)
    for child_name, child in list(root.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and any(full_name.endswith(sfx) for sfx in suffixes):
            setattr(
                root,
                child_name,
                LoRALinear.from_linear(
                    child,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                ),
            )
            replaced += 1
            continue
        replaced += replace_linear_with_lora(
            child,
            target_suffixes=suffixes,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            prefix=full_name,
        )
    return replaced


def mark_only_lora_trainable(module: nn.Module) -> int:
    count = 0
    for submodule in module.modules():
        if not isinstance(submodule, LoRALinear):
            continue
        submodule.weight.requires_grad_(False)
        if submodule.bias is not None:
            submodule.bias.requires_grad_(False)
        if submodule.lora_A is not None:
            submodule.lora_A.requires_grad_(True)
        if submodule.lora_B is not None:
            submodule.lora_B.requires_grad_(True)
        count += 1
    return count


def count_lora_parameters(module: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for name, param in module.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
    return total, trainable
