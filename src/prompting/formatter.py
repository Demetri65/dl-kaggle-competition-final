from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.config import ExperimentConfig
from src.prompting.templates import CHOICE_LETTERS, PromptTemplate, get_prompt_template


@dataclass
class PromptFormatter:
    config: ExperimentConfig

    @property
    def template(self) -> PromptTemplate:
        return get_prompt_template(self.config.prompting.template)

    @property
    def formulation_mode(self) -> str:
        return self.config.formulation.mode

    def valid_labels(self, num_choices: int) -> list[str]:
        if self.formulation_mode == "restricted_index_scoring":
            return [str(index) for index in range(num_choices)]
        if self.formulation_mode == "restricted_letter_scoring":
            return [CHOICE_LETTERS[index] for index in range(num_choices)]
        if self.formulation_mode == "candidate_yes_no":
            return [str(index) for index in range(num_choices)]
        raise ValueError(f"Unsupported formulation mode: {self.formulation_mode}")

    def completion_for_choice(self, choice_index: int) -> str:
        if self.formulation_mode == "restricted_index_scoring":
            return f" {choice_index}"
        if self.formulation_mode == "restricted_letter_scoring":
            return f" {CHOICE_LETTERS[choice_index]}"
        raise ValueError(f"Choice completion is not used for formulation: {self.formulation_mode}")

    def yes_text(self) -> str:
        return " Yes"

    def no_text(self) -> str:
        return " No"

    def render_target_text(
        self,
        record: Mapping[str, Any],
        candidate_index: int | None = None,
    ) -> str:
        answer = record.get("answer")
        if answer is None:
            raise ValueError("Cannot render a target without an answer.")
        answer_index = int(answer)
        if self.formulation_mode == "candidate_yes_no":
            if candidate_index is None:
                raise ValueError("candidate_yes_no targets require candidate_index.")
            return self.yes_text() if candidate_index == answer_index else self.no_text()
        return self.completion_for_choice(answer_index)

    def render_prompt(
        self,
        record: Mapping[str, Any],
        split: str,
        candidate_index: int | None = None,
    ) -> str:
        sections: list[str] = []
        if self.config.fields.image:
            sections.append("<image>")

        question = self._clean_text(record.get("question"))
        if self.config.fields.question and question:
            sections.append(self._format_section("Question", question))

        if self.config.fields.choices:
            choices = record.get("choices") or []
            sections.append(self._format_section("Choices", self._render_choices(choices)))

        for field_name, label in (
            ("hint", "Hint"),
            ("lecture", "Lecture"),
            ("task", "Task"),
            ("grade", "Grade"),
            ("subject", "Subject"),
            ("topic", "Topic"),
            ("category", "Category"),
            ("skill", "Skill"),
        ):
            if getattr(self.config.fields, field_name):
                value = self._clean_text(record.get(field_name))
                if value:
                    sections.append(self._format_section(label, value))

        if self.config.fields.solution and split == "train":
            solution = self._clean_text(record.get("solution"))
            if solution:
                sections.append(self._format_section("Solution", solution))

        if self.template.silent_reasoning_cue:
            sections.append("Think carefully before answering, but only output the final answer.")

        if self.formulation_mode == "candidate_yes_no":
            if candidate_index is None:
                raise ValueError("candidate_yes_no prompts require candidate_index.")
            candidate_text = record["choices"][candidate_index]
            sections.append(self._format_section("Candidate answer", str(candidate_text)))

        sections.append(self._answer_instruction(record))
        separator = "\n" if self.template.compact else "\n\n"
        return separator.join(section for section in sections if section)

    def _render_choices(self, choices: list[str]) -> str:
        display_mode = "restricted_letter_scoring" if self.formulation_mode == "restricted_letter_scoring" else "restricted_index_scoring"
        rendered: list[str] = []
        for index, choice in enumerate(choices):
            if display_mode == "restricted_letter_scoring":
                label = CHOICE_LETTERS[index]
            else:
                label = str(index)
            rendered.append(f"{label}. {choice}")
        return "\n".join(rendered)

    def _answer_instruction(self, record: Mapping[str, Any]) -> str:
        valid_labels = self.valid_labels(int(record["num_choices"]))
        if self.formulation_mode == "restricted_index_scoring":
            instruction = "Answer with the single correct index."
        elif self.formulation_mode == "restricted_letter_scoring":
            instruction = "Answer with the single correct letter."
        else:
            instruction = "Is the candidate answer correct? Reply with Yes or No only."

        if self.template.strong_output_constraint:
            instruction += f" Valid outputs: {', '.join(valid_labels)}."
        elif self.formulation_mode != "candidate_yes_no":
            instruction += f" Valid outputs are: {', '.join(valid_labels)}."

        return f"{instruction}\nAnswer:"

    def _format_section(self, label: str, value: str) -> str:
        if self.template.compact:
            return f"{label}: {value}"
        return f"{label}:\n{value}"

    @staticmethod
    def _clean_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
