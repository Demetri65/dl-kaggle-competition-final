from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.config import ExperimentConfig
from src.data.transforms import ImageTransform

if TYPE_CHECKING:
    from src.prompting.formatter import PromptFormatter


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SPLIT_FILES = {"train", "val", "test"}


@dataclass(frozen=True)
class ScienceQAExample:
    id: str
    image_path: str | None
    question: str
    choices: list[str]
    num_choices: int
    image_caption: str | None = None
    answer: int | None = None
    hint: str | None = None
    lecture: str | None = None
    solution: str | None = None
    task: str | None = None
    grade: str | None = None
    subject: str | None = None
    topic: str | None = None
    category: str | None = None
    skill: str | None = None

    def to_prompt_fields(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_path": self.image_path,
            "image_caption": self.image_caption,
            "question": self.question,
            "choices": self.choices,
            "num_choices": self.num_choices,
            "answer": self.answer,
            "hint": self.hint,
            "lecture": self.lecture,
            "solution": self.solution,
            "task": self.task,
            "grade": self.grade,
            "subject": self.subject,
            "topic": self.topic,
            "category": self.category,
            "skill": self.skill,
        }


def _normalize_optional_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _parse_choices(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(choice) for choice in value]
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError(f"choices must decode to a list, got {type(parsed)!r}")
        return [str(choice) for choice in parsed]
    raise TypeError(f"Unsupported choices value: {type(value)!r}")


def _row_to_example(row: pd.Series) -> ScienceQAExample:
    choices = _parse_choices(row["choices"])
    answer = row.get("answer")
    answer_index = None if pd.isna(answer) else int(answer)
    raw_num_choices = row.get("num_choices")
    num_choices = len(choices) if pd.isna(raw_num_choices) else int(raw_num_choices)
    return ScienceQAExample(
        id=str(row["id"]),
        image_path=_normalize_optional_text(row.get("image_path")),
        image_caption=_normalize_optional_text(row.get("image_caption")),
        question=str(row["question"]).strip(),
        choices=choices,
        num_choices=num_choices,
        answer=answer_index,
        hint=_normalize_optional_text(row.get("hint")),
        lecture=_normalize_optional_text(row.get("lecture")),
        solution=_normalize_optional_text(row.get("solution")),
        task=_normalize_optional_text(row.get("task")),
        grade=_normalize_optional_text(row.get("grade")),
        subject=_normalize_optional_text(row.get("subject")),
        topic=_normalize_optional_text(row.get("topic")),
        category=_normalize_optional_text(row.get("category")),
        skill=_normalize_optional_text(row.get("skill")),
    )


def load_split_dataframe(data_dir: Path, split: str, limit: int | None = None) -> pd.DataFrame:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unsupported split: {split}")
    csv_path = data_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")
    dataframe = pd.read_csv(csv_path)
    if limit is not None:
        dataframe = dataframe.head(limit).copy()
    return dataframe


def load_split_examples(data_dir: Path, split: str, limit: int | None = None) -> list[ScienceQAExample]:
    dataframe = load_split_dataframe(data_dir, split, limit=limit)
    return [_row_to_example(row) for _, row in dataframe.iterrows()]


def _caption_cache_path(cache_dir: str, split: str) -> Path:
    return Path(cache_dir) / f"{split}.jsonl"


def load_caption_cache(cache_dir: str, split: str) -> dict[str, str]:
    cache_path = _caption_cache_path(cache_dir, split)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing image caption cache for split {split!r}: {cache_path}. "
            "Run scripts/generate_image_captions.py before enabling fields.image_caption."
        )
    captions: dict[str, str] = {}
    with cache_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("id", "")).strip()
            caption = _normalize_optional_text(row.get("image_caption"))
            if example_id and caption:
                captions[example_id] = caption
            elif example_id:
                captions[example_id] = ""
            else:
                raise ValueError(f"Caption cache row {line_number} in {cache_path} is missing id.")
    return captions


def attach_image_captions(
    examples: list[ScienceQAExample],
    config: ExperimentConfig,
    split: str,
) -> list[ScienceQAExample]:
    if not (config.captioning.enabled and config.fields.image_caption):
        return examples
    captions = load_caption_cache(config.captioning.cache_dir, split)
    missing_ids = [
        example.id
        for example in examples
        if example.image_path and example.id not in captions
    ]
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise FileNotFoundError(
            f"Caption cache for split {split!r} is missing {len(missing_ids)} image captions "
            f"(examples: {preview}). Run scripts/generate_image_captions.py for this split."
        )
    return [
        replace(example, image_caption=captions.get(example.id) or None)
        for example in examples
    ]


def combine_examples(*example_lists: Iterable[ScienceQAExample]) -> list[ScienceQAExample]:
    combined: list[ScienceQAExample] = []
    for examples in example_lists:
        combined.extend(list(examples))
    return combined


class ImagePathResolver:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._name_index: dict[str, Path] | None = None

    def resolve(self, rel_path: str) -> Path:
        direct = self.data_dir / rel_path
        if direct.exists():
            return direct
        if self._name_index is None:
            self._name_index = {
                path.name: path
                for path in self.data_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            }
        resolved = self._name_index.get(Path(rel_path).name)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(f"Image not found under {self.data_dir}: {rel_path}")


@dataclass(frozen=True)
class DatasetItem:
    example: ScienceQAExample
    split: str
    candidate_index: int | None
    prompt_text: str
    target_text: str | None
    sample_weight: float


class ScienceQADataset(Dataset):
    def __init__(
        self,
        examples: list[ScienceQAExample],
        config: ExperimentConfig,
        formatter: "PromptFormatter",
        split: str,
        image_transform: ImageTransform | None = None,
    ) -> None:
        self.examples = list(examples)
        self.config = config
        self.formatter = formatter
        self.split = split
        self.image_transform = image_transform
        self.data_dir = Path(config.data.data_dir)
        self.image_resolver = ImagePathResolver(self.data_dir)
        self.items = self._build_items()

    def _build_items(self) -> list[DatasetItem]:
        items: list[DatasetItem] = []
        is_training = self.split == "train"
        if is_training and self.config.formulation.mode == "candidate_yes_no":
            for example in self.examples:
                for candidate_index in range(example.num_choices):
                    items.append(
                        DatasetItem(
                            example=example,
                            split=self.split,
                            candidate_index=candidate_index,
                            prompt_text=self.formatter.render_prompt(
                                example.to_prompt_fields(),
                                split=self.split,
                                candidate_index=candidate_index,
                            ),
                            target_text=self.formatter.render_target_text(
                                example.to_prompt_fields(),
                                candidate_index=candidate_index,
                            ),
                            sample_weight=1.0,
                        )
                    )
            return items

        for example in self.examples:
            prompt_text = ""
            if self.config.formulation.mode != "candidate_yes_no":
                prompt_text = self.formatter.render_prompt(example.to_prompt_fields(), split=self.split)
            items.append(
                DatasetItem(
                    example=example,
                    split=self.split,
                    candidate_index=None,
                    prompt_text=prompt_text,
                    target_text=self.formatter.render_target_text(example.to_prompt_fields()) if is_training else None,
                    sample_weight=1.0,
                )
            )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, rel_path: str | None) -> Image.Image | None:
        if not self.config.fields.image:
            return None
        if not rel_path:
            image = self._blank_image()
            if self.image_transform is not None:
                image = self.image_transform(image)
            return image
        image = Image.open(self.image_resolver.resolve(rel_path)).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image

    def _blank_image(self) -> Image.Image:
        target_edge = max(1, int(self.config.image.target_long_edge))
        return Image.new("RGB", (target_edge, target_edge), color=(255, 255, 255))

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        example = item.example
        image = self._load_image(example.image_path)
        return {
            "id": example.id,
            "split": self.split,
            "dataset_index": index,
            "image": image,
            "prompt_text": item.prompt_text,
            "target_text": item.target_text,
            "candidate_index": item.candidate_index,
            "choices": example.choices,
            "num_choices": example.num_choices,
            "answer": example.answer,
            "task": example.task,
            "subject": example.subject,
            "sample_weight": item.sample_weight,
            "metadata": example.to_prompt_fields(),
        }
