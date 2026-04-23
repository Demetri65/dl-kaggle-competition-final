from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import deep_merge_dicts, load_experiment_config, overrides_to_dict, write_experiment_override
from src.submission_utils import ensemble_prediction_files
from src.training.train import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sweep of experiment ids.")
    parser.add_argument("experiments", nargs="+", help="Experiment ids or nested config references to run sequentially.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Shared config override in key=value form.")
    parser.add_argument("--seed", type=int, help="Optional global seed override.")
    parser.add_argument("--output-dir", help="Optional output root override.")
    parser.add_argument("--predict-test", action="store_true", help="Run test predictions for every experiment.")
    parser.add_argument("--final-retrain", action="store_true", help="Train on train+val for every experiment.")
    parser.add_argument("--ensemble-name", help="If set, average normalized test scores into one submission.")
    parser.add_argument("--stage-name", help="Optional stage name for writing summary artifacts.")
    parser.add_argument("--select-best-by", default="val_accuracy", help="Metric column used to select the best parent config.")
    parser.add_argument("--ascending", action="store_true", help="Select the smallest value instead of the largest.")
    parser.add_argument("--promote-best-to", help="Write a child experiment config derived from the best stage parent.")
    parser.add_argument("--child-set", dest="child_overrides", action="append", default=[], help="Override to apply when promoting the best parent to a child config.")
    parser.add_argument("--overwrite-child", action="store_true", help="Allow overwriting an existing promoted child config.")
    return parser.parse_args()


def build_stage_row(config: Any, summary: dict[str, Any]) -> dict[str, Any]:
    val_metrics = summary.get("val_metrics") or {}
    return {
        "experiment_id": config.experiment_id,
        "parent_experiment_id": config.parent_experiment_id,
        "run_name": config.run_name,
        "run_mode": summary.get("run_mode"),
        "seed": config.runtime.seed,
        "trainable_parameter_count": summary.get("trainable_parameters"),
        "val_accuracy": val_metrics.get("overall_accuracy"),
        "two_choice_accuracy": val_metrics.get("two_choice_accuracy"),
        "two_choice_support": val_metrics.get("two_choice_support"),
        "yes_no_true_false_accuracy": val_metrics.get("yes_no_true_false_accuracy"),
        "yes_no_true_false_support": val_metrics.get("yes_no_true_false_support"),
        "output_dir": str(config.output_dir),
        "val_predictions_path": summary.get("val_predictions_path"),
        "test_predictions_path": summary.get("test_predictions_path"),
        "submission_path": summary.get("submission_path"),
        "resolved_config_path": summary.get("resolved_config_path"),
        "run_summary_path": summary.get("run_summary_path"),
        "eval_artifact_dir": summary.get("eval_artifact_dir"),
    }


def write_stage_artifacts(stage_dir: Path, stage_rows: list[dict[str, Any]], best_row: dict[str, Any] | None) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(stage_rows)
    dataframe.to_csv(stage_dir / "summary.csv", index=False)
    with (stage_dir / "summary.json").open("w") as handle:
        json.dump(stage_rows, handle, indent=2, sort_keys=True)
    if best_row is not None:
        with (stage_dir / "best_parent.json").open("w") as handle:
            json.dump(best_row, handle, indent=2, sort_keys=True)


def _normalize_stage_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def select_best_row(stage_rows: list[dict[str, Any]], metric_name: str, ascending: bool) -> dict[str, Any] | None:
    if not stage_rows:
        return None
    dataframe = pd.DataFrame(stage_rows)
    if metric_name not in dataframe.columns:
        raise ValueError(f"Stage metric not found: {metric_name}")
    ranked = dataframe.dropna(subset=[metric_name]).sort_values(metric_name, ascending=ascending)
    if ranked.empty:
        return None
    return {
        key: _normalize_stage_value(value)
        for key, value in ranked.iloc[0].to_dict().items()
    }


def promote_best_parent(
    repo_root: Path,
    child_experiment_id: str,
    parent_experiment_id: str,
    stage_name: str | None,
    select_best_by: str,
    shared_overrides: list[str],
    child_overrides: list[str],
    overwrite: bool,
) -> Path:
    child_path = repo_root / "configs" / "experiments" / f"{child_experiment_id}.yaml"
    if child_path.exists() and not overwrite:
        raise FileExistsError(f"Child config already exists: {child_path}")

    shared_override_data = overrides_to_dict(shared_overrides)
    child_override_data = overrides_to_dict(child_overrides)
    override_data: dict[str, Any] = {
        "experiment_id": child_experiment_id,
        "parent_experiment_id": parent_experiment_id,
        "promotion": {
            "stage_name": stage_name,
            "selected_metric": select_best_by,
            "parent_experiment_id": parent_experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    for payload in (shared_override_data, child_override_data):
        override_data = deep_merge_dicts(override_data, payload)

    write_experiment_override(repo_root, child_experiment_id, override_data)
    return child_path


def main() -> None:
    args = parse_args()
    shared_overrides = list(args.overrides)
    overrides = list(shared_overrides)
    if args.predict_test:
        overrides.append("runtime.predict_test=true")
    if args.final_retrain:
        overrides.append("runtime.final_retrain=true")

    prediction_paths: list[Path] = []
    stage_rows: list[dict[str, Any]] = []
    for experiment_id in args.experiments:
        config = load_experiment_config(
            repo_root=REPO_ROOT,
            experiment_id=experiment_id,
            cli_overrides=overrides,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        summary = run_experiment(REPO_ROOT, config)
        stage_row = build_stage_row(config, summary)
        stage_rows.append(stage_row)
        print(
            json.dumps(
                stage_row,
                sort_keys=True,
            )
        )
        if summary.get("test_predictions_path"):
            prediction_paths.append(Path(summary["test_predictions_path"]))

    if args.ensemble_name:
        if not prediction_paths:
            raise ValueError("Ensembling requested, but no test prediction files were produced.")
        output_root = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs"
        ensemble_path = output_root / "ensembles" / f"{args.ensemble_name}.csv"
        ensemble_prediction_files(prediction_paths, ensemble_path)
        print(json.dumps({"ensemble_submission": str(ensemble_path)}, sort_keys=True))

    best_row = select_best_row(stage_rows, args.select_best_by, ascending=args.ascending)
    output_root = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs"
    stage_name = args.stage_name or f"stage_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    stage_dir = output_root / "stages" / stage_name
    write_stage_artifacts(stage_dir, stage_rows, best_row)
    print(json.dumps({"stage_summary_dir": str(stage_dir)}, sort_keys=True))

    if best_row is not None:
        print(json.dumps({"best_parent": best_row}, sort_keys=True))

    if args.promote_best_to:
        if best_row is None:
            raise ValueError("Could not promote a best parent because the selected metric is missing for every run.")
        child_path = promote_best_parent(
            repo_root=REPO_ROOT,
            child_experiment_id=args.promote_best_to,
            parent_experiment_id=str(best_row["experiment_id"]),
            stage_name=stage_name,
            select_best_by=args.select_best_by,
            shared_overrides=shared_overrides,
            child_overrides=args.child_overrides,
            overwrite=args.overwrite_child,
        )
        print(json.dumps({"promoted_child_config": str(child_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
