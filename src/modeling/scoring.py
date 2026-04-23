from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from src.prompting.formatter import PromptFormatter

DEFAULT_MAX_COMPLETION_BATCH_SIZE = 8


@dataclass
class ChoiceScore:
    choice_index: int
    label: str
    raw_score: float
    normalized_score: float


@dataclass
class PredictionResult:
    predicted_index: int
    predicted_label: str
    choice_scores: list[ChoiceScore]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["choice_scores"] = [asdict(choice_score) for choice_score in self.choice_scores]
        return payload


def normalize_yes_no_score(yes_score: float, no_score: float) -> float:
    max_score = max(yes_score, no_score)
    denominator = max_score + math.log(math.exp(yes_score - max_score) + math.exp(no_score - max_score))
    return yes_score - denominator


def softmax_normalize(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def build_prediction_result(labels: list[str], raw_scores: list[float]) -> PredictionResult:
    normalized_scores = softmax_normalize(raw_scores)
    best_index = max(range(len(raw_scores)), key=lambda index: raw_scores[index])
    choice_scores = [
        ChoiceScore(
            choice_index=index,
            label=labels[index],
            raw_score=float(raw_scores[index]),
            normalized_score=float(normalized_scores[index]),
        )
        for index in range(len(raw_scores))
    ]
    return PredictionResult(
        predicted_index=best_index,
        predicted_label=labels[best_index],
        choice_scores=choice_scores,
    )


def sequence_logprob_scores(logits: Any, labels: Any) -> Any:
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)
    token_log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = torch.gather(token_log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = shift_labels != -100
    return (gathered * mask).sum(dim=1)


class FormulationScorer:
    def __init__(
        self,
        model: Any,
        processor: Any,
        formatter: PromptFormatter,
        max_completion_batch_size: int = DEFAULT_MAX_COMPLETION_BATCH_SIZE,
    ) -> None:
        self.model = model
        self.processor = processor
        self.formatter = formatter
        self.device = next(model.parameters()).device
        self.max_completion_batch_size = max_completion_batch_size

    def predict(
        self,
        record: Mapping[str, Any],
        split: str,
        image: Any | None,
    ) -> PredictionResult:
        return self.predict_many([record], split=split, images=[image])[0]

    def predict_many(
        self,
        records: list[Mapping[str, Any]],
        split: str,
        images: list[Any | None],
    ) -> list[PredictionResult]:
        if len(records) != len(images):
            raise ValueError("Record and image batches must be the same length.")
        if not records:
            return []
        if self.formatter.formulation_mode == "candidate_yes_no":
            return self._predict_many_candidate_yes_no(records, split=split, images=images)
        return self._predict_many_restricted(records, split=split, images=images)

    def _predict_many_restricted(
        self,
        records: list[Mapping[str, Any]],
        split: str,
        images: list[Any | None],
    ) -> list[PredictionResult]:
        prompt_texts: list[str] = []
        completion_texts: list[str] = []
        flattened_images: list[Any | None] = []
        plans: list[tuple[list[str], int, int]] = []
        for record, image in zip(records, images):
            labels = self.formatter.valid_labels(int(record["num_choices"]))
            prompt_text = self.formatter.render_prompt(record, split=split)
            start_index = len(completion_texts)
            prompt_texts.extend([prompt_text] * len(labels))
            completion_texts.extend([f" {label}" for label in labels])
            flattened_images.extend([image] * len(labels))
            plans.append((labels, start_index, len(labels)))

        flattened_scores = self.score_completions(prompt_texts, completion_texts, flattened_images)
        return [
            build_prediction_result(labels, flattened_scores[start_index : start_index + score_count])
            for labels, start_index, score_count in plans
        ]

    def _predict_many_candidate_yes_no(
        self,
        records: list[Mapping[str, Any]],
        split: str,
        images: list[Any | None],
    ) -> list[PredictionResult]:
        prompt_texts: list[str] = []
        completion_texts: list[str] = []
        flattened_images: list[Any | None] = []
        plans: list[tuple[list[str], int, int]] = []
        for record, image in zip(records, images):
            num_choices = int(record["num_choices"])
            labels = self.formatter.valid_labels(num_choices)
            start_index = len(completion_texts)
            for candidate_index in range(num_choices):
                prompt_text = self.formatter.render_prompt(record, split=split, candidate_index=candidate_index)
                prompt_texts.extend([prompt_text, prompt_text])
                completion_texts.extend([self.formatter.yes_text(), self.formatter.no_text()])
                flattened_images.extend([image, image])
            plans.append((labels, start_index, num_choices))

        flattened_scores = self.score_completions(prompt_texts, completion_texts, flattened_images)
        results: list[PredictionResult] = []
        for labels, start_index, num_choices in plans:
            candidate_scores: list[float] = []
            for candidate_index in range(num_choices):
                yes_score = flattened_scores[start_index + candidate_index * 2]
                no_score = flattened_scores[start_index + candidate_index * 2 + 1]
                candidate_scores.append(normalize_yes_no_score(yes_score, no_score))
            results.append(build_prediction_result(labels, candidate_scores))
        return results

    def score_completions(
        self,
        prompt_texts: list[str],
        completion_texts: list[str],
        images: list[Any | None],
    ) -> list[float]:
        if len(prompt_texts) != len(completion_texts):
            raise ValueError("Prompt and completion batches must be the same length.")
        if len(prompt_texts) != len(images):
            raise ValueError("Prompt and image batches must be the same length.")
        if self.max_completion_batch_size <= 0:
            raise ValueError("Completion batch size cap must be positive.")

        scores: list[float] = []
        for start_index in range(0, len(prompt_texts), self.max_completion_batch_size):
            end_index = start_index + self.max_completion_batch_size
            scores.extend(
                self._score_completion_chunk(
                    prompt_texts[start_index:end_index],
                    completion_texts[start_index:end_index],
                    images[start_index:end_index],
                )
            )
        return scores

    def _score_completion_chunk(
        self,
        prompt_texts: list[str],
        completion_texts: list[str],
        images: list[Any | None],
    ) -> list[float]:
        full_texts = [prompt + completion for prompt, completion in zip(prompt_texts, completion_texts)]
        full_batch = self._encode_inputs(full_texts, images)
        prompt_batch = self._encode_inputs(prompt_texts, images)
        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        for row_index, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[row_index, :prompt_length] = -100

        model_inputs = {key: value for key, value in full_batch.items() if key != "token_type_ids"}
        with torch.inference_mode():
            outputs = self.model(**model_inputs)
        scores = sequence_logprob_scores(outputs.logits, labels)
        return scores.detach().cpu().tolist()

    def _encode_inputs(self, texts: list[str], images: list[Any | None]) -> dict[str, Any]:
        import torch

        kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }
        if images and any(image is not None for image in images):
            # Pass images in the explicit per-prompt nested shape expected by Idefics/SmolVLM processors.
            kwargs["images"] = [[image] for image in images]
        batch = self.processor(**kwargs)
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
