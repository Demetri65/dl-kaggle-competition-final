from __future__ import annotations

from pathlib import Path

from src.config import load_experiment_config
from src.prompting.formatter import PromptFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]


SAMPLE_RECORD = {
    "id": "train_0001",
    "image_path": "images/train/train_0001.png",
    "question": "What does the diagram show?",
    "choices": ["Evaporation", "Condensation", "Precipitation"],
    "num_choices": 3,
    "answer": 1,
    "hint": "Think about clouds.",
    "lecture": "Water changes state in the atmosphere.",
    "solution": "The diagram shows water vapor turning into droplets.",
    "task": "closed choice",
    "grade": "grade5",
    "subject": "science",
    "topic": "weather",
    "category": "earth",
    "skill": "interpret diagrams",
}


def test_prompt_respects_enabled_fields() -> None:
    config = load_experiment_config(
        REPO_ROOT,
        "f03_multimodal_letter",
        cli_overrides=["fields.image=false", "fields.hint=false", "fields.lecture=false"],
    )
    formatter = PromptFormatter(config)
    prompt = formatter.render_prompt(SAMPLE_RECORD, split="val")
    assert "<image>" not in prompt
    assert "Hint:" not in prompt
    assert "Lecture:" not in prompt
    assert "Question:" in prompt
    assert "Choices:" in prompt


def test_solution_is_train_only() -> None:
    config = load_experiment_config(
        REPO_ROOT,
        "f03_multimodal_letter",
        cli_overrides=["fields.solution=true"],
    )
    formatter = PromptFormatter(config)
    train_prompt = formatter.render_prompt(SAMPLE_RECORD, split="train")
    val_prompt = formatter.render_prompt(SAMPLE_RECORD, split="val")
    assert "Solution:" in train_prompt
    assert "Solution:" not in val_prompt


def test_f03_multimodal_letter_prompt_matches_stage0_definition() -> None:
    formatter = PromptFormatter(load_experiment_config(REPO_ROOT, "f03_multimodal_letter"))

    prompt = formatter.render_prompt(SAMPLE_RECORD, split="val")

    assert "<image>" in prompt
    assert "Question:" in prompt
    assert "Choices:" in prompt
    assert "A. Evaporation" in prompt
    assert "B. Condensation" in prompt
    assert "Hint:" in prompt
    assert "Lecture:" in prompt
    assert "Candidate answer:" not in prompt
    assert "Task:" not in prompt
    assert "Grade:" not in prompt
    assert "Subject:" not in prompt
    assert "Topic:" not in prompt
    assert "Category:" not in prompt
    assert "Skill:" not in prompt
    assert "Answer with the single correct letter." in prompt
    assert "Valid outputs are: A, B, C." in prompt


def test_f04_candidate_yes_no_prompt_matches_stage0_definition() -> None:
    formatter = PromptFormatter(load_experiment_config(REPO_ROOT, "f04_candidate_yes_no"))

    prompt = formatter.render_prompt(SAMPLE_RECORD, split="val", candidate_index=1)

    assert "<image>" in prompt
    assert "Question:" in prompt
    assert "Candidate answer:" in prompt
    assert "Condensation" in prompt
    assert "Choices:" not in prompt
    assert "Hint:" not in prompt
    assert "Lecture:" not in prompt
    assert "Task:" not in prompt
    assert "Grade:" not in prompt
    assert "Subject:" not in prompt
    assert "Topic:" not in prompt
    assert "Category:" not in prompt
    assert "Skill:" not in prompt
    assert "Is the candidate answer correct? Reply with Yes or No only." in prompt
    assert "Valid outputs" not in prompt


def test_x11_f04_candidate_choices_prompt_adds_full_choices_for_eval_screen() -> None:
    formatter = PromptFormatter(load_experiment_config(REPO_ROOT, "x11_f04_candidate_choices"))

    prompt = formatter.render_prompt(SAMPLE_RECORD, split="val", candidate_index=1)

    assert "<image>" in prompt
    assert "Question:" in prompt
    assert "Choices:" in prompt
    assert "0. Evaporation" in prompt
    assert "Candidate answer:" in prompt
    assert "Condensation" in prompt
    assert "Hint:" not in prompt
    assert "Lecture:" not in prompt
    assert "Task:" not in prompt
    assert "Grade:" not in prompt
    assert "Subject:" not in prompt
    assert "Topic:" not in prompt
    assert "Category:" not in prompt
    assert "Skill:" not in prompt
    assert "Is the candidate answer correct? Reply with Yes or No only." in prompt
    assert "Valid outputs" not in prompt


def test_x12_f04_candidate_hint_prompt_adds_hint_for_eval_screen() -> None:
    formatter = PromptFormatter(load_experiment_config(REPO_ROOT, "x12_f04_candidate_hint"))

    prompt = formatter.render_prompt(SAMPLE_RECORD, split="val", candidate_index=1)

    assert "<image>" in prompt
    assert "Question:" in prompt
    assert "Choices:" not in prompt
    assert "Candidate answer:" in prompt
    assert "Condensation" in prompt
    assert "Hint:" in prompt
    assert "Lecture:" not in prompt
    assert "Task:" not in prompt
    assert "Grade:" not in prompt
    assert "Subject:" not in prompt
    assert "Topic:" not in prompt
    assert "Category:" not in prompt
    assert "Skill:" not in prompt
    assert "Is the candidate answer correct? Reply with Yes or No only." in prompt
    assert "Valid outputs" not in prompt
