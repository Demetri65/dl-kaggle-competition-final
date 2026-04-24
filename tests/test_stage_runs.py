from __future__ import annotations

import sys
from pathlib import Path

from scripts.run_stage import build_stage_row, main as run_stage_main
from src.config import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_stage_row_includes_metadata_paths() -> None:
    config = load_experiment_config(REPO_ROOT, "f03_multimodal_letter")
    row = build_stage_row(
        config,
        {
            "run_mode": "eval_only",
            "trainable_parameters": 0,
            "val_metrics": {"overall_accuracy": 0.7},
            "resolved_config_path": "outputs/run/resolved_config.yaml",
            "run_summary_path": "outputs/run/run_summary.yaml",
            "eval_artifact_dir": "outputs/source_run",
        },
    )

    assert row["resolved_config_path"].endswith("resolved_config.yaml")
    assert row["run_summary_path"].endswith("run_summary.yaml")
    assert row["eval_artifact_dir"] == "outputs/source_run"


def test_run_stage_auto_writes_summary_when_stage_name_is_omitted(monkeypatch, tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    config = load_experiment_config(REPO_ROOT, "f03_multimodal_letter", output_dir=str(output_root))

    monkeypatch.setattr(
        "scripts.run_stage.load_experiment_config",
        lambda **_: config,
    )
    monkeypatch.setattr(
        "scripts.run_stage.run_experiment",
        lambda *_: {
            "run_mode": "train_eval",
            "trainable_parameters": 123,
            "val_metrics": {"overall_accuracy": 0.6},
            "resolved_config_path": str(config.output_dir / "resolved_config.yaml"),
            "run_summary_path": str(config.output_dir / "run_summary.yaml"),
            "eval_artifact_dir": None,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_stage.py", "f03_multimodal_letter", "--output-dir", str(output_root)],
    )

    run_stage_main()

    summary_paths = list((output_root / "stages").glob("*/summary.csv"))
    assert len(summary_paths) == 1
