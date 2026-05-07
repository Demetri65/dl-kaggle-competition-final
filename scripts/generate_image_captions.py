from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ExperimentConfig, load_experiment_config
from src.data.dataset import ImagePathResolver, ScienceQAExample, load_caption_cache, load_split_examples
from src.data.transforms import build_image_transform
from src.modeling.load_model import resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate self-caption cache files for competition images.")
    parser.add_argument("--experiment", default="ta01_dora_caption_context_512_aug", help="Experiment config to source model/image/caption settings from.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, help="Optional per-split example limit for smoke generation.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Config override in key=value form.")
    parser.add_argument("--force-refresh", action="store_true", help="Regenerate selected rows even if cached captions exist.")
    return parser.parse_args()


def _cache_path(config: ExperimentConfig, split: str) -> Path:
    return Path(config.captioning.cache_dir) / f"{split}.jsonl"


def _load_existing_cache(config: ExperimentConfig, split: str) -> dict[str, str]:
    try:
        return load_caption_cache(config.captioning.cache_dir, split)
    except FileNotFoundError:
        return {}


def _load_model_and_processor(config: ExperimentConfig) -> tuple[Any, Any, torch.device]:
    processor = AutoProcessor.from_pretrained(
        config.model.model_id,
        local_files_only=config.model.local_files_only,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(config)
    model = AutoModelForImageTextToText.from_pretrained(
        config.model.model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        local_files_only=config.model.local_files_only,
    )
    if not torch.cuda.is_available():
        model.to(torch.device("cpu"))
    model.eval()
    return model, processor, next(model.parameters()).device


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _caption_batch(
    model: Any,
    processor: Any,
    device: torch.device,
    images: list[Any],
    prompt: str,
    max_new_tokens: int,
) -> list[str]:
    prompt_texts = [f"<image>\n{prompt}" for _ in images]
    batch = processor(
        text=prompt_texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    batch = _move_to_device(batch, device)
    input_lengths = batch["attention_mask"].sum(dim=1)
    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    tokenizer = processor.tokenizer
    captions: list[str] = []
    for row_index, prompt_length in enumerate(input_lengths.tolist()):
        generated_ids = output_ids[row_index][int(prompt_length):]
        caption = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        captions.append(" ".join(caption.split()))
    return captions


def _write_cache(config: ExperimentConfig, split: str, examples: list[ScienceQAExample], captions: dict[str, str]) -> Path:
    path = _cache_path(config, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    with path.open("w") as handle:
        for example in examples:
            seen_ids.add(example.id)
            handle.write(
                json.dumps(
                    {
                        "id": example.id,
                        "image_path": example.image_path,
                        "image_caption": captions.get(example.id, ""),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        for example_id, caption in sorted(captions.items(), key=lambda item: item[0]):
            if example_id in seen_ids:
                continue
            handle.write(
                json.dumps(
                    {
                        "id": example_id,
                        "image_path": None,
                        "image_caption": caption,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def generate_split_captions(
    config: ExperimentConfig,
    split: str,
    model: Any,
    processor: Any,
    device: torch.device,
    limit: int | None,
    force_refresh: bool,
) -> Path:
    examples = load_split_examples(Path(config.data.data_dir), split, limit=limit)
    captions = {} if force_refresh else _load_existing_cache(config, split)
    resolver = ImagePathResolver(Path(config.data.data_dir))
    transform = build_image_transform(config.image, augment=False)
    pending_examples = [
        example
        for example in examples
        if example.image_path and (force_refresh or example.id not in captions)
    ]

    for start_index in range(0, len(pending_examples), config.captioning.batch_size):
        batch_examples = pending_examples[start_index:start_index + config.captioning.batch_size]
        images = [
            transform(Image.open(resolver.resolve(example.image_path)).convert("RGB"))
            for example in batch_examples
            if example.image_path
        ]
        generated = _caption_batch(
            model,
            processor,
            device,
            images,
            prompt=config.captioning.prompt,
            max_new_tokens=config.captioning.max_new_tokens,
        )
        for example, caption in zip(batch_examples, generated):
            captions[example.id] = caption
        print(
            f"{split}: captioned {min(start_index + len(batch_examples), len(pending_examples))}/{len(pending_examples)} new images",
            flush=True,
        )

    for example in examples:
        if not example.image_path:
            captions.setdefault(example.id, "")
    path = _write_cache(config, split, examples, captions)
    print(f"{split}: wrote caption cache {path}")
    return path


def main() -> None:
    args = parse_args()
    config = load_experiment_config(REPO_ROOT, args.experiment, cli_overrides=list(args.overrides))
    force_refresh = args.force_refresh or config.captioning.force_refresh
    model, processor, device = _load_model_and_processor(config)
    for split in args.splits:
        generate_split_captions(
            config,
            split,
            model,
            processor,
            device,
            limit=args.limit,
            force_refresh=force_refresh,
        )


if __name__ == "__main__":
    main()
