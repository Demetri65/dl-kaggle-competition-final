from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.submission_utils import save_submission_from_predictions


def _group_accuracy(predictions: list[dict[str, Any]], field_name: str) -> dict[str, float] | None:
    labeled = [row for row in predictions if row.get("answer") is not None]
    if not labeled:
        return None
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in labeled:
        key = row.get(field_name) or "unknown"
        grouped[str(key)].append(row["predicted_index"] == row["answer"])
    return {
        key: sum(values) / len(values)
        for key, values in sorted(grouped.items(), key=lambda item: item[0])
    }


def _slice_accuracy(
    predictions: list[dict[str, Any]],
    predicate: Any,
) -> tuple[float | None, int]:
    labeled = [
        row
        for row in predictions
        if row.get("answer") is not None and predicate(row)
    ]
    if not labeled:
        return None, 0
    accuracy = sum(row["predicted_index"] == row["answer"] for row in labeled) / len(labeled)
    return accuracy, len(labeled)


def _normalize_choice(value: Any) -> str:
    return str(value).strip().lower()


def _is_yes_no_true_false_slice(row: dict[str, Any]) -> bool:
    choices = row.get("choices") or []
    if len(choices) != 2:
        return False
    normalized_choices = {_normalize_choice(choice) for choice in choices}
    return normalized_choices in ({"yes", "no"}, {"true", "false"})


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in predictions if row.get("answer") is not None]
    overall_accuracy = None
    if labeled:
        overall_accuracy = sum(row["predicted_index"] == row["answer"] for row in labeled) / len(labeled)
    distribution_counts = Counter(row["predicted_label"] for row in predictions)
    total_predictions = sum(distribution_counts.values()) or 1
    two_choice_accuracy, two_choice_support = _slice_accuracy(predictions, lambda row: row.get("num_choices") == 2)
    yes_no_true_false_accuracy, yes_no_true_false_support = _slice_accuracy(
        predictions,
        _is_yes_no_true_false_slice,
    )
    return {
        "overall_accuracy": overall_accuracy,
        "two_choice_accuracy": two_choice_accuracy,
        "two_choice_support": two_choice_support,
        "yes_no_true_false_accuracy": yes_no_true_false_accuracy,
        "yes_no_true_false_support": yes_no_true_false_support,
        "accuracy_by_num_choices": _group_accuracy(predictions, "num_choices"),
        "accuracy_by_task": _group_accuracy(predictions, "task"),
        "accuracy_by_subject": _group_accuracy(predictions, "subject"),
        "prediction_distribution": {
            label: count / total_predictions
            for label, count in sorted(distribution_counts.items(), key=lambda item: item[0])
        },
    }


def save_predictions(predictions: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def save_submission(predictions: list[dict[str, Any]], path: Path) -> Path:
    return save_submission_from_predictions(predictions, path)


def evaluate_dataset(
    dataset: Any,
    scorer: Any,
    output_path: Path | None = None,
    batch_size: int = 1,
    progress_label: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path | None]:
    if batch_size <= 0:
        raise ValueError("Evaluation batch size must be positive.")
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        dataset = list(dataset)

    predictions: list[dict[str, Any]] = []
    total_items = len(dataset)
    total_batches = (total_items + batch_size - 1) // batch_size
    started_at = time.monotonic()
    if progress_label:
        print(f"{progress_label}: evaluating {total_items} items in {total_batches} batches", flush=True)
    for start_index in range(0, total_items, batch_size):
        batch_number = start_index // batch_size + 1
        batch_items = [dataset[index] for index in range(start_index, min(start_index + batch_size, total_items))]
        batch_predictions = _predict_batch(scorer, batch_items)
        predictions.extend(
            _build_prediction_row(item, result)
            for item, result in zip(batch_items, batch_predictions)
        )
        if progress_label and (batch_number == 1 or batch_number == total_batches or batch_number % 10 == 0):
            elapsed = time.monotonic() - started_at
            processed = min(start_index + batch_size, total_items)
            print(
                f"{progress_label}: batch {batch_number}/{total_batches} "
                f"items {processed}/{total_items} elapsed {elapsed:.1f}s",
                flush=True,
            )

    saved_path = None
    if output_path is not None:
        saved_path = save_predictions(predictions, output_path)
    return compute_metrics(predictions), predictions, saved_path


def _predict_batch(scorer: Any, batch_items: list[dict[str, Any]]) -> list[Any]:
    if not batch_items:
        return []
    if hasattr(scorer, "predict_many"):
        splits = {item["split"] for item in batch_items}
        if len(splits) == 1:
            return scorer.predict_many(
                [item["metadata"] for item in batch_items],
                split=batch_items[0]["split"],
                images=[item["image"] for item in batch_items],
            )
    return [
        scorer.predict(item["metadata"], split=item["split"], image=item["image"])
        for item in batch_items
    ]


def _build_prediction_row(item: dict[str, Any], result: Any) -> dict[str, Any]:
    return {
        "id": item["id"],
        "split": item["split"],
        "answer": item["answer"],
        "predicted_index": result.predicted_index,
        "predicted_label": result.predicted_label,
        "num_choices": item["num_choices"],
        "choices": item["choices"],
        "task": item["task"],
        "subject": item["subject"],
        "choice_scores": [
            {
                "choice_index": score.choice_index,
                "label": score.label,
                "raw_score": score.raw_score,
                "normalized_score": score.normalized_score,
            }
            for score in result.choice_scores
        ],
    }
