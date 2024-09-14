from __future__ import annotations

import warnings
import re

import torch
from torch import nn
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_adapters_to_merge, check_target_module_exists
from peft.utils.integrations import gather_params_ctx
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import SorsaConfig
from .layer import Linear, SorsaLayer


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
        else:
            target.update_layer(
                adapter_name,
                sorsa_config.r,
                sorsa_alpha=sorsa_config.sorsa_alpha,
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
                sorsa_alpha=sorsa_config.sorsa_alpha,
                fan_in_fan_out=sorsa_config.fan_in_fan_out,
            )
            return new_module
        else:
            raise ValueError(f"Target module {target_base_layer} is not supported by SORSA.")

    @staticmethod
    def _check_target_module_exists(sorsa_config, key):
        return check_target_module_exists(sorsa_config, key)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "sorsa_only":
                for m in model.modules():
                    if isinstance(m, SorsaLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def enable_adapter_layers(self) -> None:
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:

        for module in self.model.modules():
            if isinstance(module, SorsaLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)

        if (getattr(outputs, "loss", None) is not None) and isinstance(outputs.loss, torch.Tensor):
            # Calculate the orthogonal regularization
            gamma = self.peft_config[self.trainable_adapter_name].gamma

            if gamma < 0:
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
