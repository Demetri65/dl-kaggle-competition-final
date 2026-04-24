from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from src.config import load_experiment_config
from src.modeling.scoring import FormulationScorer, build_prediction_result, normalize_yes_no_score
from src.prompting.formatter import PromptFormatter
from src.training.evaluate import compute_metrics, evaluate_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_prediction_result_normalizes_scores_and_selects_argmax() -> None:
    result = build_prediction_result(["0", "1", "2"], [0.2, 1.1, -0.4])
    assert result.predicted_index == 1
    assert result.predicted_label == "1"
    assert abs(sum(choice.normalized_score for choice in result.choice_scores) - 1.0) < 1e-6


def test_candidate_yes_no_aggregation_prefers_higher_normalized_yes_score() -> None:
    candidate_a = normalize_yes_no_score(yes_score=2.0, no_score=0.5)
    candidate_b = normalize_yes_no_score(yes_score=0.7, no_score=0.6)
    assert candidate_a > candidate_b


def test_predict_many_restricted_reconstructs_batched_scores() -> None:
    scorer = object.__new__(FormulationScorer)
    scorer.formatter = PromptFormatter(
        load_experiment_config(
            REPO_ROOT,
            "f03_multimodal_letter",
            cli_overrides=[
                "formulation.mode=restricted_index_scoring",
                "prompting.template=strict_index_output",
            ],
        )
    )
    seen: dict[str, object] = {}

    def fake_score_completions(prompt_texts: list[str], completion_texts: list[str], images: list[object | None]) -> list[float]:
        seen["prompt_count"] = len(prompt_texts)
        seen["completion_count"] = len(completion_texts)
        seen["completion_texts"] = completion_texts
        return [0.1, 1.2, -0.3, 0.8, 0.2]

    scorer.score_completions = fake_score_completions  # type: ignore[method-assign]

    results = scorer.predict_many(
        [
            {
                "question": "Question one?",
                "choices": ["A", "B", "C"],
                "num_choices": 3,
            },
            {
                "question": "Question two?",
                "choices": ["A", "B"],
                "num_choices": 2,
            },
        ],
        split="val",
        images=[None, None],
    )

    assert seen["prompt_count"] == 5
    assert seen["completion_count"] == 5
    assert seen["completion_texts"] == [" 0", " 1", " 2", " 0", " 1"]
    assert [result.predicted_index for result in results] == [1, 0]
    assert [len(result.choice_scores) for result in results] == [3, 2]


def test_predict_many_restricted_letter_scores_only_valid_letter_labels() -> None:
    scorer = object.__new__(FormulationScorer)
    scorer.formatter = PromptFormatter(load_experiment_config(REPO_ROOT, "f03_multimodal_letter"))
    seen: dict[str, object] = {}

    def fake_score_completions(prompt_texts: list[str], completion_texts: list[str], images: list[object | None]) -> list[float]:
        seen["prompt_count"] = len(prompt_texts)
        seen["completion_count"] = len(completion_texts)
        seen["completion_texts"] = completion_texts
        return [0.1, 0.2, 1.2, -0.3]

    scorer.score_completions = fake_score_completions  # type: ignore[method-assign]

    result = scorer.predict_many(
        [
            {
                "question": "Question one?",
                "choices": ["A", "B", "C", "D"],
                "num_choices": 4,
            },
        ],
        split="val",
        images=[None],
    )[0]

    assert seen["prompt_count"] == 4
    assert seen["completion_count"] == 4
    assert seen["completion_texts"] == [" A", " B", " C", " D"]
    assert result.predicted_index == 2
    assert result.predicted_label == "C"


def test_predict_many_candidate_yes_no_reconstructs_batched_scores() -> None:
    scorer = object.__new__(FormulationScorer)
    scorer.formatter = PromptFormatter(load_experiment_config(REPO_ROOT, "f04_candidate_yes_no"))
    seen: dict[str, object] = {}

    def fake_score_completions(prompt_texts: list[str], completion_texts: list[str], images: list[object | None]) -> list[float]:
        seen["prompt_count"] = len(prompt_texts)
        seen["completion_count"] = len(completion_texts)
        seen["completion_texts"] = completion_texts
        return [
            0.2, 0.0,
            1.1, -0.2,
            -0.5, 0.1,
            -0.1, 0.3,
            0.9, 0.0,
        ]

    scorer.score_completions = fake_score_completions  # type: ignore[method-assign]

    results = scorer.predict_many(
        [
            {
                "question": "Question one?",
                "choices": ["A", "B", "C"],
                "num_choices": 3,
            },
            {
                "question": "Question two?",
                "choices": ["A", "B"],
                "num_choices": 2,
            },
        ],
        split="val",
        images=[None, None],
    )

    assert seen["prompt_count"] == 10
    assert seen["completion_count"] == 10
    assert seen["completion_texts"] == [" Yes", " No"] * 5
    assert [result.predicted_index for result in results] == [1, 1]
    assert [result.predicted_label for result in results] == ["1", "1"]


def test_evaluate_dataset_uses_eval_batching_without_changing_prediction_rows() -> None:
    class RecordingScorer:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def predict_many(self, records: list[dict[str, object]], split: str, images: list[object | None]):
            self.batch_sizes.append(len(records))
            return [
                build_prediction_result(record["labels"], record["raw_scores"])
                for record in records
            ]

    dataset = [
        {
            "id": "a",
            "split": "val",
            "answer": 1,
            "num_choices": 3,
            "choices": ["A", "B", "C"],
            "task": "closed choice",
            "subject": "science",
            "image": None,
            "metadata": {"labels": ["0", "1", "2"], "raw_scores": [0.1, 1.0, -0.2]},
        },
        {
            "id": "b",
            "split": "val",
            "answer": 0,
            "num_choices": 2,
            "choices": ["A", "B"],
            "task": "diagram",
            "subject": "earth",
            "image": None,
            "metadata": {"labels": ["0", "1"], "raw_scores": [0.6, 0.2]},
        },
        {
            "id": "c",
            "split": "val",
            "answer": 1,
            "num_choices": 2,
            "choices": ["True", "False"],
            "task": "diagram",
            "subject": "earth",
            "image": None,
            "metadata": {"labels": ["0", "1"], "raw_scores": [0.1, 0.7]},
        },
    ]
    scorer = RecordingScorer()

    metrics, predictions, _ = evaluate_dataset(dataset, scorer, batch_size=2)

    assert scorer.batch_sizes == [2, 1]
    assert [row["id"] for row in predictions] == ["a", "b", "c"]
    assert [row["predicted_index"] for row in predictions] == [1, 0, 1]
    assert metrics["overall_accuracy"] == 1.0


def test_score_completions_chunks_large_flattened_batches() -> None:
    scorer = object.__new__(FormulationScorer)
    scorer.max_completion_batch_size = 2
    seen_chunk_sizes: list[int] = []

    def fake_score_completion_chunk(
        prompt_texts: list[str],
        completion_texts: list[str],
        images: list[object | None],
    ) -> list[float]:
        seen_chunk_sizes.append(len(prompt_texts))
        assert len(prompt_texts) == len(completion_texts) == len(images)
        return [float(index) for index in range(len(prompt_texts))]

    scorer._score_completion_chunk = fake_score_completion_chunk  # type: ignore[method-assign]

    scores = scorer.score_completions(
        ["p0", "p1", "p2", "p3", "p4"],
        ["c0", "c1", "c2", "c3", "c4"],
        [None, None, None, None, None],
    )

    assert seen_chunk_sizes == [2, 2, 1]
    assert scores == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_encode_inputs_uses_nested_images_for_each_prompt() -> None:
    scorer = object.__new__(FormulationScorer)
    scorer.device = torch.device("cpu")
    seen: dict[str, object] = {}

    def fake_processor(**kwargs: object) -> dict[str, object]:
        seen["kwargs"] = kwargs
        return {"input_ids": torch.tensor([[1], [2]], dtype=torch.long)}

    scorer.processor = fake_processor
    image_a = Image.new("RGB", (8, 8), color=(255, 255, 255))
    image_b = Image.new("RGB", (8, 8), color=(0, 0, 0))

    batch = scorer._encode_inputs(["first", "second"], [image_a, image_b])

    assert "input_ids" in batch
    assert seen["kwargs"]["images"] == [[image_a], [image_b]]


def test_metrics_include_grouped_accuracy_and_prediction_distribution() -> None:
    predictions = [
        {
            "id": "a",
            "answer": 0,
            "predicted_index": 0,
            "predicted_label": "0",
            "num_choices": 4,
            "choices": ["A", "B", "C", "D"],
            "task": "closed choice",
            "subject": "science",
        },
        {
            "id": "b",
            "answer": 1,
            "predicted_index": 0,
            "predicted_label": "0",
            "num_choices": 4,
            "choices": ["A", "B", "C", "D"],
            "task": "closed choice",
            "subject": "science",
        },
        {
            "id": "c",
            "answer": 0,
            "predicted_index": 0,
            "predicted_label": "0",
            "num_choices": 2,
            "choices": ["True", "False"],
            "task": "diagram",
            "subject": "earth",
        },
    ]
    metrics = compute_metrics(predictions)
    assert metrics["overall_accuracy"] == 2 / 3
    assert metrics["accuracy_by_num_choices"]["4"] == 0.5
    assert metrics["accuracy_by_task"]["diagram"] == 1.0
    assert metrics["accuracy_by_subject"]["science"] == 0.5
    assert metrics["two_choice_accuracy"] == 1.0
    assert metrics["two_choice_support"] == 1
    assert metrics["yes_no_true_false_accuracy"] == 1.0
    assert metrics["yes_no_true_false_support"] == 1
    assert metrics["prediction_distribution"]["0"] == 1.0
