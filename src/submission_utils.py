from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def resolve_test_prediction_path(run_dir: Path) -> Path:
    prediction_path = run_dir / "artifacts" / "test_predictions.jsonl"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing test prediction file: {prediction_path}")
    return prediction_path


def submission_rows_from_predictions(predictions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": row["id"], "answer": row["predicted_index"]}
        for row in predictions
    ]


def save_submission_rows(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_submission_from_predictions(predictions: list[dict[str, Any]], path: Path) -> Path:
    return save_submission_rows(submission_rows_from_predictions(predictions), path)


def save_submission_from_prediction_file(prediction_path: Path, output_path: Path) -> Path:
    return save_submission_rows(
        submission_rows_from_predictions(load_prediction_rows(prediction_path)),
        output_path,
    )


def ensemble_prediction_files(prediction_paths: list[Path], output_path: Path) -> Path:
    if not prediction_paths:
        raise ValueError("No prediction files were provided for ensembling.")

    aggregated: dict[str, dict[str, object]] = {}
    for path in prediction_paths:
        for row in load_prediction_rows(path):
            choice_scores = row["choice_scores"]
            normalized_scores = [0.0] * len(choice_scores)
            for score in choice_scores:
                normalized_scores[int(score["choice_index"])] = float(score["normalized_score"])
            entry = aggregated.setdefault(
                row["id"],
                {
                    "count": 0,
                    "scores": [0.0] * len(normalized_scores),
                },
            )
            if len(entry["scores"]) != len(normalized_scores):
                raise ValueError(f"Mismatched choice counts for id={row['id']}")
            entry["count"] = int(entry["count"]) + 1
            entry["scores"] = [
                existing + new
                for existing, new in zip(entry["scores"], normalized_scores)
            ]

    output_rows = []
    for example_id, entry in sorted(aggregated.items(), key=lambda item: item[0]):
        averaged_scores = [score / int(entry["count"]) for score in entry["scores"]]
        answer_index = max(range(len(averaged_scores)), key=lambda index: averaged_scores[index])
        output_rows.append({"id": example_id, "answer": answer_index})

    return save_submission_rows(output_rows, output_path)
