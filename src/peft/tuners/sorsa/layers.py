#  ------------------------------------------------------------------------------------------
#  Copyright 2024 Yang Cao.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .config import SorsaConfig


class SorsaLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("sorsa_A", "sorsa_S", "sorsa_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "sorsa_alpha", "scaling")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.sorsa_alpha = {}
        self.scaling = {}
        self.sorsa_A = nn.ParameterDict({})
        self.sorsa_S = nn.ParameterDict({})
        self.sorsa_B = nn.ParameterDict({})

        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, sorsa_alpha):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.sorsa_alpha[adapter_name] = sorsa_alpha
        # Actual trainable parameters
        self.sorsa_A[adapter_name] = nn.Parameter(torch.empty(r, self.in_features))
        self.sorsa_S[adapter_name] = nn.Parameter(torch.empty(r))
        self.sorsa_B[adapter_name] = nn.Parameter(torch.empty(self.out_features, r))
        self.sorsa_init(adapter_name)

    def sorsa_init(self, adapter_name):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize SORSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        v, s, ut = torch.linalg.svd(self.weight, full_matrices=False)

        sorsa_A = ut[:, : self.r[adapter_name]]
        sorsa_S = s[: self.r[adapter_name]]
        sorsa_B = v[: self.r[adapter_name], :]

        self.sorsa_A[adapter_name].data = sorsa_A
        self.sorsa_S[adapter_name].data = sorsa_S
        self.sorsa_B[adapter_name].data = sorsa_B

        weight = weight.data - self.scaling[adapter_name] * sorsa_B @ torch.diag(sorsa_S) @ sorsa_A
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.sorsa_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.sorsa_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.sorsa_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.sorsa_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.sorsa_A.keys():
                continue

            sorsa_A = self.sorsa_A[active_adapter]
            sorsa_S = self.sorsa_S[active_adapter]
            sorsa_B = self.sorsa_B[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(sorsa_A.dtype)
            sorsa_output = F.linear(sub_batch, sorsa_B @ torch.diag(sorsa_S) @ sorsa_A) * scaling
            result[sub_batch_indices_list[i]] += sorsa_output.to(torch_result_dtype)

        return result


class Linear(nn.Linear, SorsaLayer):
    # SORSA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        sorsa_alpha: int = 1,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ) -> None:
        super().__init__()
        SorsaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            sorsa_alpha=sorsa_alpha,
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.sorsa_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.sorsa_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.sorsa_B[adapter].device
        dtype = self.sorsa_B[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.sorsa_A[adapter]
        weight_S = self.sorsa_S[adapter]
        weight_B = self.sorsa_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_S = weight_S.float()
            weight_B = weight_B.float()

        output_tensor = (
            transpose(weight_B @ torch.diag(weight_S) @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        )
        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.sorsa_A[adapter].data = weight_A.to(dtype)
            self.sorsa_S[adapter].data = weight_S.to(dtype)
            self.sorsa_B[adapter].data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.sorsa_A.keys():
                    continue
                sorsa_A = self.sorsa_A[active_adapter]
                sorsa_S = self.sorsa_S[active_adapter]
                sorsa_B = self.sorsa_B[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(sorsa_A.dtype)

                result = (
                    result
                    + F.linear(
                        x,
                        sorsa_B @ torch.diag(sorsa_S) @ sorsa_A,
                    )
                    * scaling
                )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "sorsa." + rep

    def calc_ortho(self):
        ortho_loss = 0.0
        for active_adapter in adapter_names:
            if active_adapter in self.sorsa_A.keys():
                a = model.get_submodule(name).sorsa_A
                ia = torch.eye(a.shape[0], device=a.device)
                ia.requires_grad = False
                a = a @ a.T - ia
                ortho_loss += torch.norm(a, p="fro")
                b = model.get_submodule(name).sorsa_B
                ib = torch.eye(b.shape[1], device=a.device)
                ib.requires_grad = False
                b = b.T @ b - ib
                ortho_loss += torch.norm(b, p="fro")
            return ortho_loss / len(model.replaced) / 2


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    sorsa_config: SorsaConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = sorsa_config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, **kwargs)

    return new_module
