from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from peft import LoraConfig, TaskType, get_peft_model

from src.config import LoraConfigSpec


LORA_TARGET_PRESETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"],
    "attn_connector": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "modality_projection.proj",
    ],
}


def resolve_target_modules(config: LoraConfigSpec) -> list[str]:
    if config.target_modules:
        return list(config.target_modules)
    try:
        return list(LORA_TARGET_PRESETS[config.target_preset])
    except KeyError as exc:
        raise ValueError(f"Unknown LoRA target preset: {config.target_preset}") from exc


def count_trainable_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def freeze_all_parameters(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _matches_target(module_name: str, target_name: str) -> bool:
    return module_name == target_name or module_name.endswith(f".{target_name}")


def find_target_modules(model: Any, target_modules: list[str]) -> tuple[list[str], list[str]]:
    module_names = [name for name, _ in model.named_modules()]
    matched_modules = sorted(
        {
            module_name
            for module_name in module_names
            for target_name in target_modules
            if _matches_target(module_name, target_name)
        }
    )
    unmatched_targets = sorted(
        target_name
        for target_name in target_modules
        if not any(_matches_target(module_name, target_name) for module_name in module_names)
    )
    return matched_modules, unmatched_targets


@dataclass
class LoraApplicationResult:
    model: Any
    trainable_parameters: int
    matched_modules: list[str]
    unmatched_targets: list[str]
    target_modules: list[str]
    adapter_enabled: bool


def apply_lora(model: Any, config: LoraConfigSpec, will_train: bool) -> LoraApplicationResult:
    target_modules = resolve_target_modules(config) if config.enabled else []
    matched_modules, unmatched_targets = find_target_modules(model, target_modules) if target_modules else ([], [])

    if not config.enabled:
        if not will_train:
            freeze_all_parameters(model)
        trainable_parameters = 0 if not will_train else count_trainable_parameters(model)
        print("LoRA disabled for this run.")
        print(f"Matched trainable modules: {matched_modules or 'none'}")
        print(f"Trainable parameter count: {trainable_parameters}")
        return LoraApplicationResult(
            model=model,
            trainable_parameters=trainable_parameters,
            matched_modules=matched_modules,
            unmatched_targets=unmatched_targets,
            target_modules=target_modules,
            adapter_enabled=False,
        )

    if not matched_modules:
        raise ValueError(f"No modules matched the requested LoRA targets: {target_modules}")

    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias="none",
        target_modules=resolve_target_modules(config),
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, peft_config)
    trainable_parameters = count_trainable_parameters(peft_model)
    if trainable_parameters > config.trainable_parameter_cap:
        raise ValueError(
            "LoRA setup exceeds trainable parameter cap: "
            f"{trainable_parameters} > {config.trainable_parameter_cap}"
        )
    print(f"LoRA target patterns: {target_modules}")
    print("Matched LoRA modules:")
    for module_name in matched_modules:
        print(f"  - {module_name}")
    if unmatched_targets:
        print(f"Unmatched LoRA target patterns: {unmatched_targets}")
    print(f"Trainable parameter count: {trainable_parameters}")
    return LoraApplicationResult(
        model=peft_model,
        trainable_parameters=trainable_parameters,
        matched_modules=matched_modules,
        unmatched_targets=unmatched_targets,
        target_modules=target_modules,
        adapter_enabled=True,
    )
