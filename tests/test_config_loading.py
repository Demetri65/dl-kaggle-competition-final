from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import load_experiment_config, resolve_effective_config, selected_fields


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_override_loads_from_base() -> None:
    config = load_experiment_config(REPO_ROOT, "f03_multimodal_letter")
    assert config.experiment_id == "f03_multimodal_letter"
    assert config.formulation.mode == "restricted_letter_scoring"
    assert config.prompting.template == "strict_letter_output"
    assert config.fields.image is True


def test_stage0_formulations_resolve_to_intended_modes_templates_and_fields() -> None:
    expectations = {
        "f03_multimodal_letter": {
            "mode": "restricted_letter_scoring",
            "template": "strict_letter_output",
            "fields": ["image", "question", "choices", "hint", "lecture"],
        },
        "f04_candidate_yes_no": {
            "mode": "candidate_yes_no",
            "template": "candidate_yes_no",
            "fields": ["image", "question"],
        },
        "x11_f04_candidate_choices": {
            "mode": "candidate_yes_no",
            "template": "candidate_yes_no",
            "fields": ["image", "question", "choices"],
        },
        "x12_f04_candidate_hint": {
            "mode": "candidate_yes_no",
            "template": "candidate_yes_no",
            "fields": ["image", "question", "hint"],
        },
        "c03_balanced_answer": {
            "mode": "candidate_yes_no",
            "template": "candidate_yes_no",
            "fields": ["image", "question"],
        },
        "c04_task_choice_stratified": {
            "mode": "candidate_yes_no",
            "template": "candidate_yes_no",
            "fields": ["image", "question"],
        },
    }

    for experiment_id, expected in expectations.items():
        config = load_experiment_config(REPO_ROOT, experiment_id)
        assert config.formulation.mode == expected["mode"]
        assert config.prompting.template == expected["template"]
        assert selected_fields(config) == expected["fields"]


def test_stage3_sampling_configs_use_selected_f04_parent() -> None:
    expectations = {
        "c03_balanced_answer": "balanced_answer_index",
        "c04_task_choice_stratified": "stratified_num_choices_task",
    }

    for experiment_id, sampling_mode in expectations.items():
        config = load_experiment_config(REPO_ROOT, experiment_id)
        assert config.parent_experiment_id == "f04_candidate_yes_no"
        assert config.formulation.mode == "candidate_yes_no"
        assert config.prompting.template == "candidate_yes_no"
        assert config.lora.rank == 16
        assert config.lora.alpha == 32
        assert config.sampling.mode == sampling_mode


def test_cli_overrides_take_precedence() -> None:
    config = load_experiment_config(
        REPO_ROOT,
        "f03_multimodal_letter",
        cli_overrides=[
            "training.learning_rate=0.0001",
            "fields.solution=true",
            "scoring.max_completion_batch_size=64",
        ],
        seed=7,
        output_dir="tmp_outputs",
    )
    assert config.training.learning_rate == pytest.approx(0.0001)
    assert config.fields.solution is True
    assert config.scoring.max_completion_batch_size == 64
    assert config.runtime.seed == 7
    assert config.runtime.output_root == "tmp_outputs"


def test_cli_overrides_coerce_scientific_notation_strings_for_float_fields() -> None:
    config = load_experiment_config(
        REPO_ROOT,
        "f04_candidate_yes_no",
        cli_overrides=["training.learning_rate=5e-5", "lora.dropout=1e-1"],
    )
    assert config.training.learning_rate == pytest.approx(5e-5)
    assert config.lora.dropout == pytest.approx(1e-1)


def test_invalid_template_formulation_combo_fails() -> None:
    with pytest.raises(ValueError):
        load_experiment_config(
            REPO_ROOT,
            "f03_multimodal_letter",
            cli_overrides=["prompting.template=candidate_yes_no"],
        )


def test_eval_only_config_requires_zero_epochs_when_lora_disabled() -> None:
    with pytest.raises(ValueError):
        load_experiment_config(
            REPO_ROOT,
            "f03_multimodal_letter",
            cli_overrides=["lora.enabled=false", "training.epochs=1"],
        )


def test_eval_artifact_dir_requires_eval_only_mode() -> None:
    with pytest.raises(ValueError):
        load_experiment_config(
            REPO_ROOT,
            "f03_multimodal_letter",
            cli_overrides=["runtime.eval_artifact_dir=outputs/source_run"],
        )


def test_eval_artifact_dir_rejects_final_retrain() -> None:
    with pytest.raises(ValueError):
        load_experiment_config(
            REPO_ROOT,
            "f03_multimodal_letter",
            cli_overrides=[
                "training.epochs=0",
                "runtime.eval_artifact_dir=outputs/source_run",
                "runtime.final_retrain=true",
            ],
        )


def test_experiment_can_inherit_from_parent_override(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs" / "experiments"
    configs_dir.mkdir(parents=True)
    (tmp_path / "configs" / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "base",
                "formulation": {"mode": "restricted_index_scoring"},
                "prompting": {"template": "strict_index_output"},
                "lora": {"enabled": True},
                "training": {"epochs": 1},
            },
            sort_keys=False,
        )
    )
    (configs_dir / "parent.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "parent",
                "training": {"learning_rate": 0.0002},
                "fields": {"hint": False},
            },
            sort_keys=False,
        )
    )
    (configs_dir / "child.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "child",
                "parent_experiment_id": "parent",
                "fields": {"lecture": False},
            },
            sort_keys=False,
        )
    )

    config = load_experiment_config(tmp_path, "child")
    assert config.parent_experiment_id == "parent"
    assert config.training.learning_rate == pytest.approx(0.0002)
    assert config.fields.hint is False
    assert config.fields.lecture is False


def test_experiment_can_load_from_nested_stage_folder_by_id(tmp_path: Path) -> None:
    stage_dir = tmp_path / "configs" / "experiments" / "stage0"
    stage_dir.mkdir(parents=True)
    (tmp_path / "configs" / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "base",
                "formulation": {"mode": "restricted_index_scoring"},
                "prompting": {"template": "strict_index_output"},
                "lora": {"enabled": True},
                "training": {"epochs": 1},
            },
            sort_keys=False,
        )
    )
    (stage_dir / "nested.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "nested_config",
                "training": {"learning_rate": 0.0003},
            },
            sort_keys=False,
        )
    )

    config = load_experiment_config(tmp_path, "nested_config")
    assert config.experiment_id == "nested_config"
    assert config.training.learning_rate == pytest.approx(0.0003)


def test_experiment_can_merge_multiple_parent_configs(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs" / "experiments"
    configs_dir.mkdir(parents=True)
    (tmp_path / "configs" / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "base",
                "formulation": {"mode": "restricted_index_scoring"},
                "prompting": {"template": "strict_index_output"},
                "lora": {"enabled": True},
                "training": {"epochs": 1},
            },
            sort_keys=False,
        )
    )
    (configs_dir / "parent_a.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "parent_a",
                "fields": {"hint": False},
                "training": {"learning_rate": 0.0002},
            },
            sort_keys=False,
        )
    )
    (configs_dir / "parent_b.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "parent_b",
                "image": {"resize_mode": "aspect_pad"},
                "training": {"learning_rate": 0.0004},
            },
            sort_keys=False,
        )
    )
    (configs_dir / "child.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "child",
                "parent_experiment_id": "parent_a",
                "merge_experiment_ids": ["parent_b"],
                "fields": {"lecture": False},
            },
            sort_keys=False,
        )
    )

    config = load_experiment_config(tmp_path, "child")
    assert config.parent_experiment_id == "parent_a"
    assert config.merge_experiment_ids == ["parent_b"]
    assert config.fields.hint is False
    assert config.fields.lecture is False
    assert config.image.resize_mode == "aspect_pad"
    assert config.training.learning_rate == pytest.approx(0.0004)


def test_eval_artifact_config_uses_saved_resolved_config_with_runtime_field_and_precision_overrides(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "outputs" / "source_run"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "source_run",
                "model": {"model_id": "saved-model"},
                "formulation": {"mode": "candidate_yes_no"},
                "prompting": {"template": "candidate_yes_no"},
                "fields": {"image": True, "question": True, "choices": False, "hint": False, "solution": True},
                "image": {"resize_mode": "aspect_pad", "target_long_edge": 512},
                "lora": {"enabled": True, "rank": 16, "alpha": 32, "target_preset": "attn"},
                "training": {
                    "epochs": 2,
                    "learning_rate": 0.0002,
                    "batch_size": 2,
                    "eval_batch_size": 3,
                    "gradient_accumulation": 1,
                    "bf16": True,
                    "fp16": False,
                    "gradient_checkpointing": False,
                    "max_new_tokens": 12,
                },
                "scoring": {"max_completion_batch_size": 4},
                "sampling": {"mode": "balanced_answer_index"},
                "runtime": {
                    "seed": 11,
                    "output_root": "outputs",
                    "predict_test": False,
                    "final_retrain": False,
                    "num_workers": 0,
                    "logging_steps": 10,
                    "save_predictions": True,
                    "max_train_examples": None,
                    "max_val_examples": None,
                    "max_test_examples": None,
                },
            },
            sort_keys=False,
        )
    )

    requested = load_experiment_config(
        REPO_ROOT,
        "x11_f04_candidate_choices",
        cli_overrides=[
            "training.epochs=0",
            "training.eval_batch_size=24",
            "training.bf16=false",
            "training.fp16=true",
            "scoring.max_completion_batch_size=48",
            f"runtime.eval_artifact_dir={artifact_dir.relative_to(tmp_path)}",
            "runtime.predict_test=true",
            "runtime.num_workers=3",
            "runtime.save_predictions=false",
            "runtime.max_val_examples=5",
            "runtime.max_test_examples=7",
        ],
        seed=99,
        output_dir="custom_outputs",
    )

    effective = resolve_effective_config(tmp_path, requested)
    assert effective.experiment_id == requested.experiment_id
    assert effective.formulation.mode == "candidate_yes_no"
    assert effective.prompting.template == "candidate_yes_no"
    assert effective.image.resize_mode == "aspect_pad"
    assert effective.training.learning_rate == pytest.approx(0.0002)
    assert effective.training.epochs == 0
    assert effective.training.eval_batch_size == 24
    assert effective.training.bf16 is False
    assert effective.training.fp16 is True
    assert effective.scoring.max_completion_batch_size == 48
    assert effective.fields.choices is True
    assert effective.fields.hint is False
    assert effective.fields.solution is False
    assert effective.runtime.output_root == "custom_outputs"
    assert effective.runtime.seed == 99
    assert effective.runtime.predict_test is True
    assert effective.runtime.num_workers == 3
    assert effective.runtime.save_predictions is False
    assert effective.runtime.max_val_examples == 5
    assert effective.runtime.max_test_examples == 7
    assert effective.runtime.eval_artifact_dir == str(artifact_dir.relative_to(tmp_path))


def test_archived_stage0_configs_are_not_active() -> None:
    for experiment_id in (
        "f02_multimodal_index",
        "f05_candidate_yes_no_full_context",
        "f06_multimodal_letter_meta_full",
    ):
        with pytest.raises(FileNotFoundError):
            load_experiment_config(REPO_ROOT, experiment_id)
