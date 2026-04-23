from __future__ import annotations

from dataclasses import dataclass


CHOICE_LETTERS = "ABCDEFGHIJ"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    supports_formulations: tuple[str, ...]
    compact: bool = False
    strong_output_constraint: bool = False
    silent_reasoning_cue: bool = False


PROMPT_TEMPLATES = {
    "strict_index_output": PromptTemplate(
        name="strict_index_output",
        supports_formulations=("restricted_index_scoring",),
    ),
    "strict_letter_output": PromptTemplate(
        name="strict_letter_output",
        supports_formulations=("restricted_letter_scoring",),
    ),
    "candidate_yes_no": PromptTemplate(
        name="candidate_yes_no",
        supports_formulations=("candidate_yes_no",),
    ),
    "compact_prompt": PromptTemplate(
        name="compact_prompt",
        supports_formulations=("restricted_index_scoring", "restricted_letter_scoring"),
        compact=True,
    ),
    "strong_output_constraint": PromptTemplate(
        name="strong_output_constraint",
        supports_formulations=("restricted_index_scoring", "restricted_letter_scoring"),
        strong_output_constraint=True,
    ),
    "silent_reasoning_cue": PromptTemplate(
        name="silent_reasoning_cue",
        supports_formulations=("restricted_index_scoring", "restricted_letter_scoring"),
        silent_reasoning_cue=True,
    ),
}


def get_prompt_template(name: str) -> PromptTemplate:
    try:
        return PROMPT_TEMPLATES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt template: {name}") from exc

