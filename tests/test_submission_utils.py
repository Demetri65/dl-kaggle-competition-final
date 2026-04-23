from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.submission_utils import (
    ensemble_prediction_files,
    resolve_test_prediction_path,
    save_submission_from_prediction_file,
)


def _write_prediction_file(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_save_submission_from_prediction_file(tmp_path: Path) -> None:
    prediction_path = _write_prediction_file(
        tmp_path / "run" / "artifacts" / "test_predictions.jsonl",
        [
            {
                "id": "a",
                "predicted_index": 1,
                "choice_scores": [],
            },
            {
                "id": "b",
                "predicted_index": 0,
                "choice_scores": [],
            },
        ],
    )

    output_path = tmp_path / "submissions" / "single.csv"
    save_submission_from_prediction_file(prediction_path, output_path)

    dataframe = pd.read_csv(output_path)
    assert dataframe.to_dict(orient="records") == [
        {"id": "a", "answer": 1},
        {"id": "b", "answer": 0},
    ]


def test_ensemble_prediction_files_averages_normalized_scores(tmp_path: Path) -> None:
    first_path = _write_prediction_file(
        tmp_path / "run1" / "artifacts" / "test_predictions.jsonl",
        [
            {
                "id": "a",
                "predicted_index": 0,
                "choice_scores": [
                    {"choice_index": 0, "normalized_score": 0.8},
                    {"choice_index": 1, "normalized_score": 0.2},
                ],
            }
        ],
    )
    second_path = _write_prediction_file(
        tmp_path / "run2" / "artifacts" / "test_predictions.jsonl",
        [
            {
                "id": "a",
                "predicted_index": 1,
                "choice_scores": [
                    {"choice_index": 0, "normalized_score": 0.3},
                    {"choice_index": 1, "normalized_score": 0.7},
                ],
            }
        ],
    )

    output_path = tmp_path / "ensembles" / "ensemble.csv"
    ensemble_prediction_files([first_path, second_path], output_path)

    dataframe = pd.read_csv(output_path)
    assert dataframe.to_dict(orient="records") == [{"id": "a", "answer": 0}]


def test_resolve_test_prediction_path_points_to_artifact_jsonl(tmp_path: Path) -> None:
    prediction_path = tmp_path / "run" / "artifacts" / "test_predictions.jsonl"
    prediction_path.parent.mkdir(parents=True)
    prediction_path.write_text("")

    assert resolve_test_prediction_path(tmp_path / "run") == prediction_path
