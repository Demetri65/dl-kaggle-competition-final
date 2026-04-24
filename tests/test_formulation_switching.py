from __future__ import annotations

from pathlib import Path

from src.config import load_experiment_config
from src.prompting.formatter import PromptFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]


RECORD = {
    "id": "train_0010",
    "question": "Which option is correct?",
    "choices": ["Alpha", "Beta", "Gamma", "Delta"],
    "num_choices": 4,
    "answer": 2,
}


def test_index_formulation_switches_target_and_labels() -> None:
    formatter = PromptFormatter(
        load_experiment_config(
            REPO_ROOT,
            "f03_multimodal_letter",
            cli_overrides=[
                "formulation.mode=restricted_index_scoring",
                "prompting.template=strict_index_output",
            ],
        )
    )
    assert formatter.valid_labels(4) == ["0", "1", "2", "3"]
    assert formatter.render_target_text(RECORD) == " 2"


def test_letter_formulation_switches_target_and_labels() -> None:
    formatter = PromptFormatter(load_experiment_config(REPO_ROOT, "f03_multimodal_letter"))
    assert formatter.valid_labels(4) == ["A", "B", "C", "D"]
    assert formatter.render_target_text(RECORD) == " C"


def test_candidate_yes_no_formulation_uses_candidate_specific_targets() -> None:
    formatter = PromptFormatter(load_experiment_config(REPO_ROOT, "f04_candidate_yes_no"))
    yes_prompt = formatter.render_prompt(RECORD, split="val", candidate_index=1)
    assert "Candidate answer:" in yes_prompt
    assert formatter.render_target_text(RECORD, candidate_index=2) == " Yes"
    assert formatter.render_target_text(RECORD, candidate_index=1) == " No"
