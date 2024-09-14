from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from torch import nn

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class SorsaConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`SorsaModel`].

    Args:
        r (`int`):
            Sorsa attention dimension (the "rank").

    """

    r: int = field(default=8, metadata={"help": "Sorsa attention dimension"})
    gamma: float = field(default=4e-4, metadata={"help": "Sorsa gamma"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "The target modules to apply Sorsa attention. If None, apply to all modules."},
    )
    sorsa_alpha: int = field(default=8, metadata={"help": "Sorsa alpha"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"}
    )

    def to_dict(self):
        rv = super().to_dict()
        return rv

    def __post_init__(self):
        self.peft_type = PeftType.SORSA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
