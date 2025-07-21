from __future__ import annotations

import math
import operator
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from typing import Literal, Optional

import torch
from torch import nn
from tqdm import tqdm

from ....peft.import_utils import is_bnb_4bit_available, is_bnb_available
from ....peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from ....peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
    get_quantization_config,
)
from ....peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from ....peft.utils.other import get_pattern_key

from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import BitLoraConfig
from .eetq import dispatch_eetq
from .gptq import dispatch_gptq
from .hqq import dispatch_hqq
from .layer import Conv2d, LoraLayer, dispatch_default
from .torchao import dispatch_torchao
from .tp_layer import dispatch_megatron


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
    return args, kwargs


class BitLoraModel(BaseTuner):
    prefix: str = "bitlora_"

    def __init__(self, model, config: BitLoraConfig, adapter_name: str, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _check_new_adapter_config(self, config: BitLoraConfig) -> None:
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. "
                "When using multiple adapters, set bias to 'none' for all adapters."
            )
        
    @staticmethod
    def _check_target_module_exists(bitlora_config, key):
        return check_target_module_exists(bitlora_config, key)
    
    def _prepare_model(self, peft_confg: BitLoraConfig, model: nn.Module):
        if peft_confg.layer_replication:
            replicate_layers(model, peft_confg.layer_replication)
        
        # BitLinear 여부 확인
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                raise TypeError(f"BitLoraModel은 BitLinear 기반 모델에만 사용할 수 있습니다. nn.Linear 발견: {name}")
            
    # LoRA를 실제로 레이어에 삽입하거나 교체할 때 쓰는 메서드.
    def _create_and_replace(
        self,
        bitlora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't not be 'None'")
        
        r_key = get_pattern_key(bitlora_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(bitlora_config.alpha_pattern.keys(), current_key)
        r = bitlora_config.rank_pattern.get(r_key, bitlora_config.r)
        alpha = bitlora_config.alpha_pattern.get(alpha_key, bitlora_config.lora_alpha)

        kwargs = {
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": bitlora_config.lora_dropout,
        "fan_in_fan_out": bitlora_config.fan_in_fan_out,
        "init_lora_weights": bitlora_config.init_lora_weights,
        "use_rslora": bitlora_config.use_rslora,
        "lora_bias": bitlora_config.lora_bias,
        }
        
        from .BitLoRAlayer import BitLoraLayer

        if isinstance(target, BitLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=bitlora_config.lora_dropout,
                init_lora_weights=bitlora_config.init_lora_weights,
                use_rslora=bitlora_config.use_rslora,
                lora_bias=bitlora_config.lora_bias,
            )
        else:
            new_module = self._create_new_module(bitlora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)

        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        # 디바이스 일치 (meta 디바이스 대응)
        meta = torch.device("meta")
        for _, module in new_module.named_modules():
            if any(p.device == meta for p in module.parameters()):
                weight = getattr(child, "weight", next(child.parameters()))
                module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:  # 일반적으로 prefix는 "bitlora_"
                p.requires_grad = False

        from .BitLoRAlayer import BitLoraLayer

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            elif bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.modules():
                    if isinstance(m, BitLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    # def _create_new_module() -> BitLoRAlayer.py의 dispatch_default()안에 isinstance(target_base_layer, BitLinear) 분기로 대체
    @staticmethod
    def _create_new_module(bitlora_config, adapter_name, target, **kwargs):
        """
        BitLoRA에 맞게 LoRA 레이어를 생성하는 함수.

        BitLinear를 기반으로 하는 레이어만 지원하며,
        nn.Linear 등의 다른 레이어는 처리하지 않는다.
        """

        from .BitLoRAlayer import BitLoraLayer
        from .BitLoRAlayer import BitLoraLinear
        from ....BitNet.bitnet import BitLinear

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        # BitLinear가 아닌 경우 처리하지 않음
        if isinstance(target_base_layer, BitLinear):
            return BitLoraLinear(
                target=target,
                adapter_name=adapter_name,
                r=kwargs.get("r"),
                lora_alpha=kwargs.get("lora_alpha"),
                lora_dropout=kwargs.get("lora_dropout"),
                fan_in_fan_out=kwargs.get("fan_in_fan_out"),
                init_lora_weights=kwargs.get("init_lora_weights", True),
                use_rslora=kwargs.get("use_rslora", False),
                lora_bias=kwargs.get("lora_bias", "none"),
                # 필요 시 추가적으로 확장
            )

        # 지원하지 않는 레이어일 경우 에러 발생
        raise ValueError(
            f"[BitLoRAModel] Unsupported module type for BitLoRA: {type(target_base_layer)}. "
            f"Expected BitLinear, got {type(target_base_layer)}."
        )


    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config
    
    def _set_adapter_layers(self, enabled: bool = True) -> None:
        from .BitLoRAlayer import BitLoraLayer
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)
                if isinstance(module, BitLoraLayer):
                    print(f"[BitLoRA] Adapter {'ENABLED' if enabled else 'DISABLED'} for: {module}")

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
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
        from .BitLoRAlayer import BitLoraLayer
        for module in self.model.modules():
            if isinstance(module, BitLoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)

        if isinstance(adapter_name, str):
            self.active_adapters = [adapter_name]
        else:
            self.active_adapters = adapter_name

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        from .BitLoRAlayer import BitLoraLayer
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            yield
            return

        if self.training:
            raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        expected_adapters = set()
        for layer in self.modules():
            if isinstance(layer, BitLoraLayer):
                expected_adapters |= layer.lora_A.keys()
                expected_adapters |= layer.lora_embedding_A.keys()

        unique_adapters = {name for name in adapter_names if name != "__base__"}
        unexpected_adapters = unique_adapters - expected_adapters
        if unexpected_adapters:
            raise ValueError(f"Trying to infer with non-existing adapter(s): {', '.join(sorted(unexpected_adapters))}")

        hook_handles = []
        for module in self.modules():
            if isinstance(module, BitLoraLayer) or isinstance(module, ModulesToSaveWrapper):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

    def _check_merge_allowed(self):
        super()._check_merge_allowed()
        if getattr(self.model, "quantization_method", None) in {"gptq", "awq", "aqlm"}:
            raise ValueError("Cannot merge BitLoRA layers when the model uses unsupported quantization")
        if self.peft_config.get("layer_replication"):
            raise ValueError("Cannot merge BitLoRA layers when base model layers are replicated")

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            self._check_merge_allowed()  # BitLoRA용 병합 조건 확인

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "BitLoRA model"

        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)

                elif isinstance(target, ModulesToSaveWrapper):
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)

        return self.model

    def _check_add_weighted_adapter(
        self, adapters: list[str], combination_type: str, svd_rank: int | None
    ) -> tuple[str, int, str]:
        """
        Helper function to check if the arguments to add_weighted_adapter are valid and compatible with the underlying
        model.
        """
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # If more than one of the adapters targets the same module with modules_to_save, raise an error, as these
        # modules cannot be merged. First, find the ModulesToSaveWrapper instances in the model, then check if they
        # have modules for the adapters to be merged.
        modules_to_save_wrappers = [module for module in self.modules() if isinstance(module, ModulesToSaveWrapper)]
        problematic_wrappers = [
            wrapper
            for wrapper in modules_to_save_wrappers
            if sum(adapter in wrapper.modules_to_save for adapter in adapters) > 1
        ]
        if problematic_wrappers:
            raise ValueError(
                "Cannot add weighted adapters if they target the same module with modules_to_save, but found "
                f"{len(problematic_wrappers)} such instance(s)."
            )

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks = [self.peft_config[adapter].r for adapter in adapters]
        if combination_type in ("linear", "ties", "dare_ties", "dare_linear", "magnitude_prune"):
            # all adapters ranks should be same, new rank is just this value
            if len(set(adapters_ranks)) != 1:
                raise ValueError(
                    "All adapters must have the same r value when using combination_type linear, ties, dare_ties or "
                    "dare_linear."
                )
            new_rank = adapters_ranks[0]
        elif combination_type == "cat":
            # adapters ranks may be different, new rank is sum of all ranks
            # be careful, because output adapter rank may be really big if mixing a lot of adapters
            new_rank = sum(adapters_ranks)
        elif combination_type.endswith("svd"):
            # new rank is the max of all ranks of the adapters if not provided
            new_rank = svd_rank or max(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        target_module_types = [type(self.peft_config[adapter].target_modules) for adapter in adapters]
        if not target_module_types:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        if len(set(target_module_types)) > 1:
            raise ValueError(
                "all adapter configs should follow the same target modules type. "
                "Combining adapters with `target_modules` type being a mix of list/set and string is not supported."
            )

        if target_module_types[0] is str:
            new_target_modules = "|".join(f"({self.peft_config[adapter].target_modules})" for adapter in adapters)
        elif target_module_types[0] is set:
            new_target_modules = reduce(
                operator.or_, (self.peft_config[adapter].target_modules for adapter in adapters)
            )
        else:
            raise TypeError(f"Invalid type {target_module_types[0]} found in target_modules")

        return combination_type, new_rank, new_target_modules
    
    def add_weighted_adapter(
        self,
        adapters: list[str],
        weights: list[float],
        adapter_name: str,
        combination_type: str = "svd",
        svd_rank: int | None = None,
        svd_clamp: int | None = None,
        svd_full_matrices: bool = True,
        svd_driver: str | None = None,
        density: float | None = None,
        majority_sign_method: Literal["total", "frequency"] = "total",
    ) -> None:
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                The merging type can be one of [`svd`, `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`,
                `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`]. When using the `cat`
                combination_type, the rank of the resulting adapter is equal to the sum of all adapters ranks (the
                mixed adapter may be too big and result in OOM errors).
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
                clamping. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            svd_driver (`str`, *optional*):
                Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
                one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
                documentation. Defaults to None.
            density (`float`, *optional*):
                Value between 0 and 1. 0 means all values are pruned and 1 means no values are pruned. Should be used
                with [`ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`,
                `magnintude_prune`, `magnitude_prune_svd`]
            majority_sign_method (`str`):
                The method, should be one of ["total", "frequency"], to use to get the magnitude of the sign values.
                Should be used with [`ties`, `ties_svd`, `dare_ties`, `dare_ties_svd`]
        """

        if adapter_name in list(self.peft_config.keys()):
            return

        combination_type, new_rank, new_target_modules = self._check_add_weighted_adapter(
            adapters=adapters,
            combination_type=combination_type,
            svd_rank=svd_rank,
        )

        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            r=new_rank,
            lora_alpha=new_rank,
            target_modules=new_target_modules,
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target_lora_A = target.lora_A[adapter_name].weight
                    target_lora_B = target.lora_B[adapter_name].weight
                elif adapter_name in target.lora_embedding_A:
                    target_lora_A = target.lora_embedding_A[adapter_name]
                    target_lora_B = target.lora_embedding_B[adapter_name]
                else:
                    continue

                target_lora_A.data = target_lora_A.data * 0.0
                target_lora_B.data = target_lora_B.data * 0.0
                if combination_type == "cat":
                    loras_A, loras_B = [], []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        else:
                            continue
                        loras_A.append(current_adapter_lora_A.data * weight * target.scaling[adapter])
                        loras_B.append(current_adapter_lora_B.data)

                    if len(loras_A) == 0:
                        raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
                    loras_A = torch.cat(loras_A, dim=0)
                    loras_B = torch.cat(loras_B, dim=1)
                    target_lora_A.data[: loras_A.shape[0], :] = loras_A
                    target_lora_B.data[:, : loras_B.shape[1]] = loras_B
                elif combination_type in [
                    "svd",
                    "ties_svd",
                    "dare_linear_svd",
                    "dare_ties_svd",
                    "magnitude_prune_svd",
                ]:
                    target_lora_A.data, target_lora_B.data = self._svd_generalized_task_arithmetic_weighted_adapter(
                        combination_type,
                        adapters,
                        weights,
                        new_rank,
                        target,
                        target_lora_A,
                        target_lora_B,
                        density,
                        majority_sign_method,
                        svd_clamp,
                        full_matrices=svd_full_matrices,
                        driver=svd_driver,
                    )
                elif combination_type in ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"]:
                    target_lora_A.data, target_lora_B.data = self._generalized_task_arithmetic_weighted_adapter(
                        combination_type, adapters, weights, target, density, majority_sign_method
                    )

    def _svd_generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        new_rank,
        target,
        target_lora_A,
        target_lora_B,
        density,
        majority_sign_method,
        clamp=None,
        full_matrices=True,
        driver=None,
    ):
        valid_adapters = []
        valid_weights = []
        is_embedding = any(adapter in target.lora_embedding_A for adapter in adapters)
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A or adapter in target.lora_embedding_A:
                valid_adapters.append(adapter)
                valid_weights.append(weight * target.scaling[adapter])

        # if no valid adapter, nothing to do
        if len(valid_adapters) == 0:
            raise ValueError("No matching LoRAs found. Please raise an issue on Github.")
        delta_weight = [target.get_delta_weight(adapter) for adapter in valid_adapters]
        valid_weights = torch.tensor(valid_weights).to(delta_weight[0].device)
        if combination_type == "svd":
            delta_weight = task_arithmetic(delta_weight, valid_weights)
        elif combination_type == "ties_svd":
            delta_weight = ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "dare_linear_svd":
            delta_weight = dare_linear(delta_weight, valid_weights, density)
        elif combination_type == "dare_ties_svd":
            delta_weight = dare_ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "magnitude_prune_svd":
            delta_weight = magnitude_prune(delta_weight, valid_weights, density)
        else:
            raise ValueError(f"Invalid value passed to combination type: {combination_type}")

        conv2d = isinstance(target, Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if (hasattr(target, "fan_in_fan_out") and target.fan_in_fan_out) or is_embedding:
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
        U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=full_matrices, driver=driver)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_lora_B.data.shape)
            Vh = Vh.reshape(target_lora_A.data.shape)
        return Vh, U

    def _generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        target,
        density,
        majority_sign_method,
    ):
        # account weights for LoRA A and B layers.
        valid_weights = []
        lora_A_deltas = []
        lora_B_deltas = []
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A:
                current_adapter_lora_A = target.lora_A[adapter].weight
                current_adapter_lora_B = target.lora_B[adapter].weight
            elif adapter in target.lora_embedding_A:
                current_adapter_lora_A = target.lora_embedding_A[adapter]
                current_adapter_lora_B = target.lora_embedding_B[adapter]
            else:
                continue
            valid_weights.append(math.sqrt(weight * target.scaling[adapter]))
            lora_A_deltas.append(current_adapter_lora_A.data)
            lora_B_deltas.append(current_adapter_lora_B.data)
        valid_weights = torch.tensor(valid_weights).to(lora_A_deltas[0].device)
        lora_deltas = [lora_A_deltas, lora_B_deltas]
        dtype = lora_A_deltas[0].dtype
        for i, task_tensors in enumerate(lora_deltas):
            if combination_type == "linear":
                lora_deltas[i] = task_arithmetic(task_tensors, valid_weights)
            elif combination_type == "ties":
                lora_deltas[i] = ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "dare_linear":
                lora_deltas[i] = dare_linear(task_tensors, valid_weights, density)
            elif combination_type == "dare_ties":
                lora_deltas[i] = dare_ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "magnitude_prune":
                lora_deltas[i] = magnitude_prune(task_tensors, valid_weights, density)
            else:
                raise ValueError("Invalid combination type")
        lora_deltas = [delta.to(dtype) for delta in lora_deltas]
        return lora_deltas
    
    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def subtract_mutated_init(self, output_state_dict: dict[str, torch.Tensor], adapter_name: str, kwargs=None):
        """
        This function can calculate the updates of the [PiSSA | OLoRA] by comparing the parameters of the [PiSSA |
        OLoRA] adapter in `output_state_dict` with the initial values of [PiSSA | OLoRA] in `adapter_name`, thus
        converting [PiSSA | OLoRA] to LoRA.
        """
        for name, param in self.model.named_parameters():
            if (
                param.data.dtype != torch.float32
                and param.data.dtype != torch.float16
                and param.data.dtype != torch.bfloat16
            ) and adapter_name.startswith("pissa"):
                warnings.warn(
                    r"Note that Quant(W_res) + AB != Quant(W) + \Delta(AB); "
                    "the converted LoRA, when combined with W or Quant(W), may introduce a certain gap in the fine-tuned model. "
                    "Therefore, we recommend directly using the Quant(W_res) in conjunction with the PiSSA adapter. "
                )
        mutated_init_state_dict = get_peft_model_state_dict(
            self,
            state_dict=kwargs.get("state_dict", None),
            adapter_name=adapter_name,
        )
        tensors_lora = {}
        for name in output_state_dict.keys():
            ## W = W^{res} + A_0 \times B_0,
            ## W + \Delta W = W^{res} + A \times B,
            ## \Delta W = A \times B - A_0 \times B_0 = [A | A_0] \times [B | -B_0]^T = A'B'.
            if "lora_A" in name:
                tensors_lora[name] = torch.cat(
                    [output_state_dict[name], mutated_init_state_dict[".".join(name.split(".")[1:])]], dim=0
                )
            elif "lora_B" in name:
                tensors_lora[name] = torch.cat(
                    [output_state_dict[name], -mutated_init_state_dict[".".join(name.split(".")[1:])]], dim=1
                )

        return tensors_lora