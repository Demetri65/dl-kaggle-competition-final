from __future__ import annotations

import csv
from pathlib import Path

from src.config import load_experiment_config
from src.results import append_results_row, build_results_row


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_append_results_row_expands_existing_schema(tmp_path: Path) -> None:
    results_path = tmp_path / "experiments.csv"
    append_results_row({"experiment_id": "old", "val_accuracy": 0.5}, results_path=results_path)
    append_results_row(
        {
            "experiment_id": "new",
            "val_accuracy": 0.6,
            "trainable_parameter_count": 1234,
        },
        results_path=results_path,
    )

    with results_path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["experiment_id"] == "old"
    assert rows[0]["trainable_parameter_count"] == ""
    assert rows[1]["experiment_id"] == "new"
    assert rows[1]["trainable_parameter_count"] == "1234"


def test_build_results_row_includes_metadata_paths() -> None:
    config = load_experiment_config(
        REPO_ROOT,
        "f03_multimodal_letter",
        cli_overrides=["training.epochs=0", "runtime.eval_artifact_dir=outputs/source_run"],
        output_dir="custom_outputs",
    )
    row = build_results_row(
        REPO_ROOT,
        config,
        {
            "trainable_parameters": 0,
            "matched_lora_modules": [],
            "unmatched_lora_targets": [],
            "val_metrics": {"overall_accuracy": 0.8},
            "resolved_config_path": "custom_outputs/f03_multimodal_letter_seed42/resolved_config.yaml",
            "run_summary_path": "custom_outputs/f03_multimodal_letter_seed42/run_summary.yaml",
            "eval_artifact_dir": "outputs/source_run",
        },
    )

    assert row["resolved_config_path"].endswith("resolved_config.yaml")
    assert row["run_summary_path"].endswith("run_summary.yaml")
    assert row["eval_artifact_dir"] == "outputs/source_run"
