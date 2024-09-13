from __future__ import annotations

import warnings

import torch
from torch import nn
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
)
from peft.utils.integrations import gather_params_ctx

from .config import SorsaConfig
from .layers import Linear, SorsaLayer


class SorsaModel(BaseTuner):
    prefix: str = "sorsa_"

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    def _create_and_replace(
        self,
        sorsa_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        if not isinstance(target, SorsaLayer):
            new_module = self._create_new_module(sorsa_config, adapter_name, target)
            self._replace_module(parent, target_name, new_module, target)
        target.update_layer(
            adapter_name,
            sorsa_config.r,
            sorsa_alpha=sorsa_config.alpha,
        )

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            if hasattr(new_module, "W_q"):  # HQQ
                new_module.W_q = child.W_q
            else:
                new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                weight = (
                    child.qweight
                    if hasattr(child, "qweight")
                    else (
                        child.W_q
                        if hasattr(child, "W_q")
                        else child.weight if hasattr(child, "weight") else next(child.parameters())
                    )
                )
                module.to(weight.device)

    @staticmethod
    def _create_new_module(sorsa_config, adapter_name, target):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if sorsa_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                sorsa_config.fan_in_fan_out = False
            new_module = Linear(
                target,
                adapter_name,
                sorsa_config.r,
                sorsa_alpha=sorsa_config.alpha,
                fan_in_fan_out=sorsa_config.fan_in_fan_out,
            )
            return new_module
        else:
            raise ValueError(f"Target module {target_base_layer} is not supported by SORSA.")

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)

        if (getattr(outputs, "loss", None) is not None) and isinstance(outputs.loss, torch.Tensor):
            # Calculate the orthogonal regularization
            gamma = self.peft_config[self.trainable_adapter_name].gamma

            if gamma <= 0:
                raise ValueError("gamma should be greater than 0. ")

            regu_loss = 0
            num_param = 0
            for n, p in self.model.named_parameters():
                if ("sorsa_A" in n or "sorsa_B" in n) and self.trainable_adapter_name in n:
                    if p.shape == torch.Size([0]):
                        with gather_params_ctx(p, fwd_module=self):
                            para_cov = p @ p.T if "sorsa_A" in n else p.T @ p
                    else:
                        para_cov = p @ p.T if "sorsa_A" in n else p.T @ p
                    I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))  # noqa: E741
                    I.requires_grad = False
                    num_param += 1
                    regu_loss += torch.norm(para_cov - I, p="fro")
            if num_param > 0:
                regu_loss = regu_loss / num_param
            else:
                regu_loss = 0
            outputs.loss += gamma * regu_loss
        return outputs
