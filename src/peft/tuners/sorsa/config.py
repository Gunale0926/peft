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
        """
        Returns the configuration for your adapter model as a dictionary. Removes runtime configurations.
        """
        rv = super().to_dict()
        rv.pop("runtime_config")
        return rv
