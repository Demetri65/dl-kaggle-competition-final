from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin, get_type_hints

import yaml


FORMULATION_MODES = {
    "restricted_index_scoring",
    "restricted_letter_scoring",
    "candidate_yes_no",
}
SAMPLING_MODES = {
    "uniform",
    "weighted_loss_answer_index",
    "balanced_answer_index",
    "stratified_num_choices_task",
}
PROMPT_TEMPLATE_SUPPORT: dict[str, set[str]] = {
    "strict_index_output": {"restricted_index_scoring"},
    "strict_letter_output": {"restricted_letter_scoring"},
    "candidate_yes_no": {"candidate_yes_no"},
    "compact_prompt": {"restricted_index_scoring", "restricted_letter_scoring"},
    "strong_output_constraint": {"restricted_index_scoring", "restricted_letter_scoring"},
    "silent_reasoning_cue": {"restricted_index_scoring", "restricted_letter_scoring"},
}
EVAL_RUNTIME_OVERRIDE_KEYS = (
    "output_root",
    "seed",
    "predict_test",
    "num_workers",
    "save_predictions",
    "max_val_examples",
    "max_test_examples",
)
EVAL_TRAINING_OVERRIDE_KEYS = (
    "eval_batch_size",
    "bf16",
    "fp16",
)
EVAL_SCORING_OVERRIDE_KEYS = (
    "max_completion_batch_size",
)


@dataclass
class DataConfig:
    data_dir: str = "data"


@dataclass
class ModelConfig:
    model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    local_files_only: bool = False


@dataclass
class FormulationConfig:
    mode: str = "restricted_index_scoring"


@dataclass
class FieldConfig:
    image: bool = True
    image_caption: bool = False
    question: bool = True
    choices: bool = True
    hint: bool = True
    lecture: bool = True
    task: bool = False
    grade: bool = False
    subject: bool = False
    topic: bool = False
    category: bool = False
    skill: bool = False
    solution: bool = False


@dataclass
class ImageAugmentationConfig:
    enabled: bool = False
    random_resized_crop_scale: list[float] = field(default_factory=lambda: [0.9, 1.0])
    rotation_degrees: float = 3.0
    brightness_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class ImageConfig:
    resize_mode: str = "stretch"
    target_long_edge: int = 384
    augmentation: ImageAugmentationConfig = field(default_factory=ImageAugmentationConfig)


@dataclass
class PromptingConfig:
    template: str = "strict_index_output"
    answer_prefix: str = "Answer:"


@dataclass
class LoraConfigSpec:
    enabled: bool = True
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    use_dora: bool = False
    target_preset: str = "attn"
    target_modules: list[str] = field(default_factory=list)
    trainable_parameter_cap: int = 5_000_000


@dataclass
class CaptioningConfig:
    enabled: bool = False
    cache_dir: str = "data/cache/image_captions"
    prompt: str = "Describe the image in one concise sentence, focusing on text, labels, axes, objects, and relationships relevant to answering a science question."
    max_new_tokens: int = 48
    batch_size: int = 1
    force_refresh: bool = False


@dataclass
class TrainingCheckpointConfig:
    enabled: bool = False
    save_steps: int = 0
    save_epochs: bool = True
    max_to_keep: int = 3


@dataclass
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation: int = 4
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    checkpointing: TrainingCheckpointConfig = field(default_factory=TrainingCheckpointConfig)
    max_new_tokens: int = 8


@dataclass
class ScoringConfig:
    max_completion_batch_size: int = 8


@dataclass
class HardExampleConfig:
    enabled: bool = False
    epochs: int = 1
    max_examples: int | None = None


@dataclass
class SamplingConfig:
    mode: str = "uniform"
    hard_example_follow_up: HardExampleConfig = field(default_factory=HardExampleConfig)


@dataclass
class RuntimeConfig:
    seed: int = 42
    output_root: str = "outputs"
    predict_test: bool = False
    final_retrain: bool = False
    eval_artifact_dir: str | None = None
    resume_from_checkpoint: str | None = None
    num_workers: int = 0
    logging_steps: int = 10
    save_predictions: bool = True
    max_train_examples: int | None = None
    max_val_examples: int | None = None
    max_test_examples: int | None = None


@dataclass
class ExperimentConfig:
    experiment_id: str = "base"
    parent_experiment_id: str | None = None
    merge_experiment_ids: list[str] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    formulation: FormulationConfig = field(default_factory=FormulationConfig)
    fields: FieldConfig = field(default_factory=FieldConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    prompting: PromptingConfig = field(default_factory=PromptingConfig)
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    captioning: CaptioningConfig = field(default_factory=CaptioningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def run_name(self) -> str:
        return f"{self.experiment_id}_seed{self.runtime.seed}"

    @property
    def output_dir(self) -> Path:
        return Path(self.runtime.output_root) / self.run_name

    @property
    def is_eval_only(self) -> bool:
        return self.training.epochs == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def experiments_config_dir(repo_root: Path) -> Path:
    return repo_root / "configs" / "experiments"


def _experiment_relative_ref(repo_root: Path, path: Path) -> str:
    return str(path.relative_to(experiments_config_dir(repo_root)).with_suffix(""))


def _candidate_experiment_paths(repo_root: Path, experiment_ref: str) -> list[Path]:
    configs_dir = experiments_config_dir(repo_root)
    reference_path = Path(experiment_ref)
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        if path.exists() and path not in candidates:
            candidates.append(path)

    if reference_path.is_absolute():
        add_candidate(reference_path)
        return candidates

    if experiment_ref.endswith(".yaml"):
        add_candidate(configs_dir / experiment_ref)
    if "/" in experiment_ref or "\\" in experiment_ref:
        add_candidate(configs_dir / f"{experiment_ref}.yaml")

    flat_path = configs_dir / f"{experiment_ref}.yaml"
    add_candidate(flat_path)

    for path in sorted(configs_dir.rglob("*.yaml")):
        if path in candidates:
            continue
        if path.stem == experiment_ref or _experiment_relative_ref(repo_root, path) == experiment_ref:
            candidates.append(path)
            continue
        if str((_read_yaml(path).get("experiment_id") or "")) == experiment_ref:
            candidates.append(path)
    return candidates


def experiment_config_path(repo_root: Path, experiment_ref: str) -> Path:
    candidates = _candidate_experiment_paths(repo_root, experiment_ref)
    if not candidates:
        raise FileNotFoundError(
            f"Config file not found for experiment reference {experiment_ref!r} under {experiments_config_dir(repo_root)}"
        )
    if len(candidates) > 1:
        candidate_refs = ", ".join(_experiment_relative_ref(repo_root, path) for path in candidates)
        raise ValueError(
            f"Experiment reference {experiment_ref!r} is ambiguous. Use one of: {candidate_refs}"
        )
    return candidates[0]


def experiment_config_write_path(repo_root: Path, experiment_ref: str) -> Path:
    configs_dir = experiments_config_dir(repo_root)
    reference_path = Path(experiment_ref)
    if reference_path.is_absolute():
        return reference_path
    if experiment_ref.endswith(".yaml"):
        return configs_dir / experiment_ref
    if "/" in experiment_ref or "\\" in experiment_ref:
        return configs_dir / f"{experiment_ref}.yaml"
    return configs_dir / f"{experiment_ref}.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_nested_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = data
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if next_value is None or not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[keys[-1]] = value


def apply_overrides(config_data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    merged = dict(config_data)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must use key=value syntax: {override}")
        key, raw_value = override.split("=", 1)
        parsed_value = yaml.safe_load(raw_value)
        _set_nested_value(merged, key, parsed_value)
    return merged


def overrides_to_dict(overrides: list[str] | None) -> dict[str, Any]:
    return apply_overrides({}, overrides)


def _coerce_scalar(value: Any, target_type: type[Any]) -> Any:
    if target_type is float and isinstance(value, str):
        return float(value)
    if target_type is int and isinstance(value, str):
        return int(value)
    if target_type is bool and isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
    if target_type is str and not isinstance(value, str):
        return str(value)
    return value


def _coerce_value(field_type: Any, value: Any) -> Any:
    origin = get_origin(field_type)
    if origin is None:
        if isinstance(field_type, type):
            return _coerce_scalar(value, field_type)
        return value

    if origin is list:
        element_types = get_args(field_type)
        if isinstance(value, list) and len(element_types) == 1:
            return [_coerce_value(element_types[0], item) for item in value]
        return value

    union_types = [union_type for union_type in get_args(field_type) if union_type is not type(None)]
    if union_types and len(union_types) == 1:
        return _coerce_value(union_types[0], value)
    return value


def _build_dataclass(dataclass_type: type[Any], data: Mapping[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(dataclass_type)
    for field_info in dataclass_type.__dataclass_fields__.values():
        value = data.get(field_info.name)
        if value is None:
            continue
        field_type = type_hints.get(field_info.name, field_info.type)
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, Mapping):
            kwargs[field_info.name] = _build_dataclass(field_type, value)
        else:
            kwargs[field_info.name] = _coerce_value(field_type, value)
    return dataclass_type(**kwargs)


def load_experiment_config(
    repo_root: Path,
    experiment_id: str,
    cli_overrides: list[str] | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
) -> ExperimentConfig:
    base_path = repo_root / "configs" / "base.yaml"
    merged = deep_merge_dicts(_read_yaml(base_path), _resolve_experiment_override(repo_root, experiment_id, seen=[]))
    merged = apply_overrides(merged, cli_overrides)
    if seed is not None:
        _set_nested_value(merged, "runtime.seed", seed)
    if output_dir is not None:
        _set_nested_value(merged, "runtime.output_root", output_dir)
    config = _build_dataclass(ExperimentConfig, merged)
    validate_config(config)
    return config


def load_saved_run_config(run_dir: Path) -> ExperimentConfig:
    resolved_config_path = run_dir / "resolved_config.yaml"
    config = _build_dataclass(ExperimentConfig, _read_yaml(resolved_config_path))
    validate_config(config)
    return config


def resolve_eval_artifact_dir(repo_root: Path, eval_artifact_dir: str) -> Path:
    artifact_dir = Path(eval_artifact_dir)
    if artifact_dir.is_absolute():
        return artifact_dir
    return repo_root / artifact_dir


def resolve_effective_config(repo_root: Path, requested_config: ExperimentConfig) -> ExperimentConfig:
    eval_artifact_dir = requested_config.runtime.eval_artifact_dir
    if not eval_artifact_dir:
        return requested_config

    source_config = load_saved_run_config(resolve_eval_artifact_dir(repo_root, eval_artifact_dir))
    merged = source_config.to_dict()
    merged["experiment_id"] = requested_config.experiment_id
    merged["parent_experiment_id"] = requested_config.parent_experiment_id

    training_data = dict(merged["training"])
    training_data["epochs"] = 0
    requested_training = requested_config.to_dict()["training"]
    for key in EVAL_TRAINING_OVERRIDE_KEYS:
        training_data[key] = requested_training[key]
    merged["training"] = training_data

    fields_data = dict(requested_config.to_dict()["fields"])
    fields_data["solution"] = False
    merged["fields"] = fields_data

    scoring_data = dict(merged.get("scoring") or {})
    requested_scoring = requested_config.to_dict()["scoring"]
    for key in EVAL_SCORING_OVERRIDE_KEYS:
        scoring_data[key] = requested_scoring[key]
    merged["scoring"] = scoring_data

    runtime_data = dict(merged["runtime"])
    requested_runtime = requested_config.to_dict()["runtime"]
    for key in EVAL_RUNTIME_OVERRIDE_KEYS:
        runtime_data[key] = requested_runtime[key]
    runtime_data["final_retrain"] = False
    runtime_data["eval_artifact_dir"] = eval_artifact_dir
    merged["runtime"] = runtime_data

    config = _build_dataclass(ExperimentConfig, merged)
    validate_config(config)
    return config


def _parent_experiment_refs(override_data: Mapping[str, Any]) -> list[str]:
    merge_experiment_ids = override_data.get("merge_experiment_ids") or []
    if not isinstance(merge_experiment_ids, list):
        raise ValueError("merge_experiment_ids must be a list of experiment references.")

    parent_refs: list[str] = []
    parent_experiment_id = override_data.get("parent_experiment_id")
    if parent_experiment_id:
        parent_refs.append(str(parent_experiment_id))
    parent_refs.extend(str(experiment_id) for experiment_id in merge_experiment_ids)
    return parent_refs


def _resolve_experiment_override(repo_root: Path, experiment_id: str, seen: list[str]) -> dict[str, Any]:
    if experiment_id in seen:
        chain = " -> ".join(seen + [experiment_id])
        raise ValueError(f"Detected a config inheritance cycle: {chain}")

    experiment_path = experiment_config_path(repo_root, experiment_id)
    override_data = _read_yaml(experiment_path)
    parent_refs = _parent_experiment_refs(override_data)
    if not parent_refs:
        return override_data

    merged_parent_data: dict[str, Any] = {}
    for parent_ref in parent_refs:
        parent_data = _resolve_experiment_override(repo_root, parent_ref, seen + [experiment_id])
        merged_parent_data = deep_merge_dicts(merged_parent_data, parent_data)

    resolved_data = deep_merge_dicts(merged_parent_data, override_data)
    if "parent_experiment_id" not in override_data:
        resolved_data.pop("parent_experiment_id", None)
    if "merge_experiment_ids" not in override_data:
        resolved_data.pop("merge_experiment_ids", None)
    return resolved_data


def write_experiment_override(repo_root: Path, experiment_id: str, override_data: Mapping[str, Any]) -> Path:
    path = experiment_config_write_path(repo_root, experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(dict(override_data), handle, sort_keys=False)
    return path


def validate_config(config: ExperimentConfig) -> None:
    if config.formulation.mode not in FORMULATION_MODES:
        raise ValueError(f"Unsupported formulation mode: {config.formulation.mode}")
    if config.prompting.template not in PROMPT_TEMPLATE_SUPPORT:
        raise ValueError(f"Unsupported prompt template: {config.prompting.template}")
    supported = PROMPT_TEMPLATE_SUPPORT[config.prompting.template]
    if config.formulation.mode not in supported:
        raise ValueError(
            "Prompt template and formulation are incompatible: "
            f"{config.prompting.template} vs {config.formulation.mode}"
        )
    if config.sampling.mode not in SAMPLING_MODES:
        raise ValueError(f"Unsupported sampling mode: {config.sampling.mode}")
    if config.image.resize_mode not in {"stretch", "aspect_pad"}:
        raise ValueError(f"Unsupported image resize mode: {config.image.resize_mode}")
    if len(config.image.augmentation.random_resized_crop_scale) != 2:
        raise ValueError("image.augmentation.random_resized_crop_scale must contain two values.")
    crop_min, crop_max = config.image.augmentation.random_resized_crop_scale
    if not 0 < crop_min <= crop_max <= 1:
        raise ValueError("image.augmentation.random_resized_crop_scale must satisfy 0 < min <= max <= 1.")
    if len(config.image.augmentation.brightness_range) != 2:
        raise ValueError("image.augmentation.brightness_range must contain two values.")
    brightness_min, brightness_max = config.image.augmentation.brightness_range
    if not 0 < brightness_min <= brightness_max:
        raise ValueError("image.augmentation.brightness_range must satisfy 0 < min <= max.")
    if config.captioning.batch_size <= 0:
        raise ValueError("captioning.batch_size must be positive.")
    if config.captioning.max_new_tokens <= 0:
        raise ValueError("captioning.max_new_tokens must be positive.")
    if config.training.bf16 and config.training.fp16:
        raise ValueError("Config cannot enable both bf16 and fp16.")
    if config.lora.enabled and config.lora.rank <= 0:
        raise ValueError("LoRA rank must be positive when LoRA is enabled.")
    if config.training.epochs < 0:
        raise ValueError("Training epochs cannot be negative.")
    if config.training.batch_size <= 0 or config.training.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if config.scoring.max_completion_batch_size <= 0:
        raise ValueError("Scoring completion batch size must be positive.")
    if config.training.gradient_accumulation <= 0:
        raise ValueError("Gradient accumulation must be positive.")
    if config.training.checkpointing.save_steps < 0:
        raise ValueError("training.checkpointing.save_steps cannot be negative.")
    if config.training.checkpointing.max_to_keep <= 0:
        raise ValueError("training.checkpointing.max_to_keep must be positive.")
    if config.runtime.eval_artifact_dir and config.training.epochs != 0:
        raise ValueError("runtime.eval_artifact_dir requires training.epochs=0.")
    if config.runtime.eval_artifact_dir and config.runtime.final_retrain:
        raise ValueError("runtime.eval_artifact_dir cannot be combined with runtime.final_retrain.")
    if config.runtime.resume_from_checkpoint and config.runtime.eval_artifact_dir:
        raise ValueError("runtime.resume_from_checkpoint cannot be combined with runtime.eval_artifact_dir.")
    if config.runtime.resume_from_checkpoint and config.training.epochs == 0:
        raise ValueError("runtime.resume_from_checkpoint requires training. Set training.epochs > 0.")
    if not config.lora.enabled and config.training.epochs > 0:
        raise ValueError(
            "LoRA-disabled training is not supported in this framework. "
            "Use training.epochs=0 for eval-only runs or enable LoRA."
        )
    if config.training.epochs == 0 and config.sampling.hard_example_follow_up.enabled:
        raise ValueError("Hard-example follow-up requires training. Set training.epochs > 0.")
    if config.runtime.final_retrain and config.fields.solution:
        # Keep the rule explicit because final retrain often mixes train and val data.
        pass


def selected_fields(config: ExperimentConfig) -> list[str]:
    return [name for name, enabled in asdict(config.fields).items() if enabled]
