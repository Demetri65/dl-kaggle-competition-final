from __future__ import annotations

from collections import Counter
from typing import Any

from torch.utils.data import WeightedRandomSampler

from src.data.dataset import ScienceQADataset


def _inverse_frequency_weights(values: list[Any]) -> list[float]:
    counts = Counter(values)
    return [1.0 / counts[value] for value in values]


def build_loss_weights(dataset: ScienceQADataset, mode: str) -> list[float] | None:
    if mode != "weighted_loss_answer_index":
        return None
    answers = [item.example.answer for item in dataset.items]
    return _inverse_frequency_weights(answers)


def build_training_sampler(dataset: ScienceQADataset, mode: str) -> WeightedRandomSampler | None:
    if mode == "balanced_answer_index":
        answers = [item.example.answer for item in dataset.items]
        weights = _inverse_frequency_weights(answers)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    if mode == "stratified_num_choices_task":
        groups = [(item.example.num_choices, item.example.task or "unknown") for item in dataset.items]
        weights = _inverse_frequency_weights(groups)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return None

