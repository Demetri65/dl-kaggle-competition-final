from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.config import ExperimentConfig
from src.modeling.lora import freeze_all_parameters


@dataclass
class ModelBundle:
    model: Any
    processor: Any
    device: torch.device
    dtype: torch.dtype


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(config: ExperimentConfig) -> torch.dtype:
    if torch.cuda.is_available():
        if config.training.bf16:
            return torch.bfloat16
        if config.training.fp16:
            return torch.float16
    return torch.float32


def _load_processor(processor_source: str | Path, config: ExperimentConfig) -> Any:
    processor = AutoProcessor.from_pretrained(
        processor_source,
        local_files_only=config.model.local_files_only,
    )
    _configure_processor(processor, config)
    return processor


def _load_base_model(model_source: str | Path, config: ExperimentConfig, dtype: torch.dtype) -> Any:
    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        local_files_only=config.model.local_files_only,
    )
    if not torch.cuda.is_available():
        model.to(resolve_device())
    return model


def _finalize_model(model: Any, config: ExperimentConfig) -> Any:
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model


def _configure_processor(processor: Any, config: ExperimentConfig) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_model_and_processor(config: ExperimentConfig) -> ModelBundle:
    device = resolve_device()
    dtype = resolve_dtype(config)
    processor = _load_processor(config.model.model_id, config)
    model = _finalize_model(_load_base_model(config.model.model_id, config, dtype), config)
    return ModelBundle(model=model, processor=processor, device=device, dtype=dtype)


def load_model_and_processor_from_artifacts(config: ExperimentConfig, eval_artifact_dir: Path) -> ModelBundle:
    device = resolve_device()
    dtype = resolve_dtype(config)
    model_dir = eval_artifact_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing saved model artifacts: {model_dir}")

    processor_dir = eval_artifact_dir / "processor"
    processor_source: str | Path = processor_dir if processor_dir.exists() else config.model.model_id
    processor = _load_processor(processor_source, config)

    adapter_config_path = model_dir / "adapter_config.json"
    if adapter_config_path.exists():
        base_model = _load_base_model(config.model.model_id, config, dtype)
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        model = _load_base_model(model_dir, config, dtype)

    freeze_all_parameters(model)
    model = _finalize_model(model, config)
    return ModelBundle(model=model, processor=processor, device=device, dtype=dtype)
