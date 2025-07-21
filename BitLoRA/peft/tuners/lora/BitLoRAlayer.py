from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from ....peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from ....peft.utils.other import transpose

from .config import BitLoraConfig

from ....BitNet.bitnet import BitLinear


class BitLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: BitLinear, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.lora_bias: dict[str, bool] = {}
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if not isinstance(base_layer, BitLinear):
            raise TypeError(f"[location: BitLoraLayer] Only BitLinear is supported, but got: {type(base_layer)}")
        
        in_features, out_features = base_layer.in_features, base_layer.out_features
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        lora_bias: bool = False,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        
        # 하이퍼 파라미터 저장
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # base_model과 adapter의 dtype이 다른 에러를 해결하기 위한 코드
        dtype = self.base_layer.weight.dtype
        # Actual trainable parameters
        self.lora_A[adapter_name] = BitLinear(self.in_features, r, bias=False).to(dtype)
        self.lora_B[adapter_name] = BitLinear(r, self.out_features, bias=lora_bias).to(dtype)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            raise ValueError(f"only support 'eva'")
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            raise ValueError(f"only support 'eva'")
        elif init_lora_weights == "loftq":
            raise ValueError(f"only support 'eva'")
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return
        if adapter_name in self.lora_B:
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            if self.lora_bias.get(adapter_name, False):
                nn.init.zeros_(self.lora_B[adapter_name].bias)
        if adapter_name in self.lora_embedding_B:
            nn.init.zeros_(self.lora_embedding_B[adapter_name])
            if self.lora_bias.get(adapter_name, False):
                nn.init.zeros_(self.lora_embedding_B[adapter_name].bias)

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value
    
    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
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
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result
    

class BitLoraLinear(nn.Module, BitLoraLayer):
    # Lora implemented in a BitLinear layer
    def __init__(
        self,
        base_layer,
        adapter_name,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        BitLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            lora_bias=lora_bias
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return
        
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
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

                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias
            
            else:
                delta_weight = self.get_delta_weight(active_adapter)
                base_layer.weight.data += delta_weight

                if self.lora_bias[active_adapter]:
                    base_layer.bias.data += self.lora_B[active_adapter].bias

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
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight
            
            if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter].bias

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        self.lora_A[adapter]와 self.lora_B[adapter]는 BitLinear.
        단순히 .weight에 접근해서 weight_B @ weight_A 형태의 행렬곱을 하면 정상 동작 X.
        BitLoRA에서는 forward 연산을 직접 호출해서 delta weight를 구해야함.

        BitLoRA 방식으로 delta weight 계산
        ΔW = transpose(B(A(I))) * scaling
        """
        A_layer = self.lora_A[adapter]
        B_layer = self.lora_B[adapter]
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        in_features = A_layer.in_features
        identity_input = torch.eye(in_features, device=device, dtype=dtype)

        with torch.no_grad():  # gradient 추적 불필요
            A_out = A_layer(identity_input)       # shape: (in_features, r)
            B_out = B_layer(A_out)                # shape: (in_features, out_features)
            delta = transpose(B_out, self.fan_in_fan_out) * self.scaling[adapter]  # shape: match base layer

        return delta

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        # adapter 완전히 비활성화된 경우
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        
        # adapter names가 명시된 경우 (mixed-batch inference 등)
        if adapter_names is not None:
            return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        
        # merge된 상태에서는 LoRA 경로 무시하고 base_layer 그대로 사용
        if self.merged:
            return self.base_layer(x, *args, **kwargs)
        
        # BitLoRA forward
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue

            lora_A = self.lora_A[active_adapter] # BitLinear
            lora_B = self.lora_B[active_adapter] # BitLinear
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            x_bit = x
            if isinstance(dropout, nn.Dropout): # dropout 적용은 FloatTensor 기준
                x_bit = dropout(x)
            
            # BitLinear forward 그대로 사용 (양자화 적용)
            delta = lora_B(lora_A(x_bit)) * scaling
            result = result + delta

        return result
    
    def __repr__(self) -> str:
        rep = super().__repr__()
        return "bitlora." + rep


# 주어진 torch.nn.Module을 LoRA 적용이 가능한 튜너 모듈로 적절히 변환 해주는 역할
def dispatch_default(
    target: nn.Module,
    adapter_name: str,
    lora_config,
    **kwargs,
) -> Optional[nn.Module]:
    new_module = None

    # base layer 추출
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # ---- BitLoRA 전용 분기 ----
    if isinstance(target_base_layer, BitLinear):
        new_module = BitLoraLinear(target, adapter_name, **kwargs)

    elif isinstance(target_base_layer, nn.Linear):
        # fan_in_fan_out 안전 처리
        if kwargs.get("fan_in_fan_out", False):
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False

        kwargs.update(lora_config.loftq_config)
        new_module = BitLoraLinear(target, adapter_name, **kwargs)

    return new_module