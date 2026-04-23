from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import ExperimentConfig, selected_fields


RESULTS_PATH = Path("results") / "experiments.csv"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def get_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_results_row(
    repo_root: Path,
    config: ExperimentConfig,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    val_metrics = summary.get("val_metrics") or {}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config.experiment_id,
        "parent_experiment_id": config.parent_experiment_id,
        "run_name": config.run_name,
        "seed": config.runtime.seed,
        "git_commit": get_git_commit(repo_root),
        "run_mode": "eval_only" if config.is_eval_only else "train_eval",
        "formulation_mode": config.formulation.mode,
        "prompt_template": config.prompting.template,
        "selected_fields": json_dumps(selected_fields(config)),
        "image_settings": json_dumps(
            {
                "enabled": config.fields.image,
                "resize_mode": config.image.resize_mode,
                "target_long_edge": config.image.target_long_edge,
            }
        ),
        "lora_settings": json_dumps(
            {
                "enabled": config.lora.enabled,
                "rank": config.lora.rank,
                "alpha": config.lora.alpha,
                "dropout": config.lora.dropout,
                "target_preset": config.lora.target_preset,
                "target_modules": config.lora.target_modules,
            }
        ),
        "trainable_parameter_count": summary.get("trainable_parameters"),
        "lora_matched_modules": json_dumps(summary.get("matched_lora_modules")),
        "lora_unmatched_targets": json_dumps(summary.get("unmatched_lora_targets")),
        "sampling_mode": config.sampling.mode,
        "training_hyperparameters": json_dumps(
            {
                "epochs": config.training.epochs,
                "learning_rate": config.training.learning_rate,
                "warmup_ratio": config.training.warmup_ratio,
                "weight_decay": config.training.weight_decay,
                "batch_size": config.training.batch_size,
                "eval_batch_size": config.training.eval_batch_size,
                "gradient_accumulation": config.training.gradient_accumulation,
                "bf16": config.training.bf16,
                "fp16": config.training.fp16,
                "gradient_checkpointing": config.training.gradient_checkpointing,
            }
        ),
        "val_accuracy": val_metrics.get("overall_accuracy"),
        "two_choice_accuracy": val_metrics.get("two_choice_accuracy"),
        "two_choice_support": val_metrics.get("two_choice_support"),
        "yes_no_true_false_accuracy": val_metrics.get("yes_no_true_false_accuracy"),
        "yes_no_true_false_support": val_metrics.get("yes_no_true_false_support"),
        "accuracy_by_num_choices": json_dumps(val_metrics.get("accuracy_by_num_choices")),
        "accuracy_by_task": json_dumps(val_metrics.get("accuracy_by_task")),
        "accuracy_by_subject": json_dumps(val_metrics.get("accuracy_by_subject")),
        "prediction_distribution": json_dumps(val_metrics.get("prediction_distribution")),
        "val_predictions_path": summary.get("val_predictions_path"),
        "test_predictions_path": summary.get("test_predictions_path"),
        "submission_path": summary.get("submission_path"),
        "resolved_config_path": summary.get("resolved_config_path"),
        "run_summary_path": summary.get("run_summary_path"),
        "eval_artifact_dir": summary.get("eval_artifact_dir"),
        "output_dir": str(config.output_dir),
    }


def append_results_row(row: Mapping[str, Any], results_path: Path = RESULTS_PATH) -> None:
    ensure_directory(results_path.parent)
    if not results_path.exists():
        with results_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with results_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        existing_rows = list(reader)
        existing_headers = list(reader.fieldnames or [])

    merged_headers = list(existing_headers)
    for key in row.keys():
        if key not in merged_headers:
            merged_headers.append(key)

    existing_rows.append({key: row.get(key) for key in merged_headers})
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_headers)
        writer.writeheader()
        for existing_row in existing_rows:
            normalized_row = {key: existing_row.get(key) for key in merged_headers}
            writer.writerow(normalized_row)
