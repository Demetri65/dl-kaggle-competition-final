from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import PeftModel
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.config import ExperimentConfig, resolve_effective_config, resolve_eval_artifact_dir
from src.data.dataset import ScienceQADataset, attach_image_captions, combine_examples, load_split_examples
from src.data.transforms import build_image_transform
from src.modeling.load_model import (
    ModelBundle,
    load_model_and_processor,
    load_model_and_processor_from_artifacts,
)
from src.modeling.lora import (
    LoraApplicationResult,
    apply_lora,
    count_trainable_parameters,
    find_target_modules,
    resolve_target_modules,
)
from src.modeling.scoring import FormulationScorer, sequence_logprob_scores
from src.prompting.formatter import PromptFormatter
from src.results import append_results_row, build_results_row, ensure_directory
from src.training.evaluate import evaluate_dataset, save_submission
from src.training.sampler import build_loss_weights, build_training_sampler


class TrainingCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_texts = [row["prompt_text"] for row in batch]
        target_texts = [row["target_text"] or "" for row in batch]
        full_texts = [prompt + target for prompt, target in zip(prompt_texts, target_texts)]
        images = [row["image"] for row in batch]
        full_batch = self._encode(full_texts, images)
        prompt_batch = self._encode(prompt_texts, images)

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        for row_index, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[row_index, :prompt_length] = -100

        full_batch["labels"] = labels
        full_batch["dataset_indices"] = torch.tensor([row["dataset_index"] for row in batch], dtype=torch.long)
        return full_batch

    def _encode(self, texts: list[str], images: list[Any | None]) -> dict[str, torch.Tensor]:
        kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }
        if images and any(image is not None for image in images):
            kwargs["images"] = images
        return self.processor(**kwargs)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _autocast_context(config: ExperimentConfig):
    enabled = torch.cuda.is_available() and (config.training.bf16 or config.training.fp16)
    dtype = torch.bfloat16 if config.training.bf16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def compute_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    dataset_indices: torch.Tensor,
    sample_weights: list[float] | None,
) -> torch.Tensor:
    per_sample_loss = -sequence_logprob_scores(logits, labels)
    if sample_weights is None:
        return per_sample_loss.mean()
    weight_tensor = torch.tensor(
        [sample_weights[index] for index in dataset_indices.tolist()],
        dtype=per_sample_loss.dtype,
        device=per_sample_loss.device,
    )
    return (per_sample_loss * weight_tensor).sum() / weight_tensor.sum()


def _checkpoint_root(output_dir: Path) -> Path:
    return output_dir / "checkpoints"


def _sorted_checkpoint_dirs(checkpoint_root: Path) -> list[Path]:
    if not checkpoint_root.exists():
        return []
    return sorted(
        [path for path in checkpoint_root.iterdir() if path.is_dir()],
        key=lambda path: (path.stat().st_mtime, path.name),
    )


def _latest_checkpoint_dir(checkpoint_root: Path) -> Path | None:
    checkpoints = _sorted_checkpoint_dirs(checkpoint_root)
    return checkpoints[-1] if checkpoints else None


def _prune_checkpoints(checkpoint_root: Path, max_to_keep: int) -> None:
    checkpoints = _sorted_checkpoint_dirs(checkpoint_root)
    for checkpoint_dir in checkpoints[:-max_to_keep]:
        shutil.rmtree(checkpoint_dir)


def _save_training_checkpoint(
    model_bundle: ModelBundle,
    config: ExperimentConfig,
    checkpoint_root: Path,
    epoch_index: int,
    global_step: int,
    reason: str,
) -> Path:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoint_root / f"{reason}_epoch{epoch_index + 1:03d}_step{global_step:06d}"
    model_dir = checkpoint_dir / "model"
    processor_dir = checkpoint_dir / "processor"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_bundle.model.save_pretrained(model_dir)
    model_bundle.processor.save_pretrained(processor_dir)
    _write_resolved_config(checkpoint_dir, config)
    with (checkpoint_dir / "checkpoint_metadata.yaml").open("w") as handle:
        yaml.safe_dump(
            {
                "epoch": epoch_index + 1,
                "global_step": global_step,
                "reason": reason,
            },
            handle,
            sort_keys=False,
        )
    _prune_checkpoints(checkpoint_root, config.training.checkpointing.max_to_keep)
    print(f"Saved training checkpoint: {checkpoint_dir}", flush=True)
    return checkpoint_dir


def _resolve_training_checkpoint_dir(repo_root: Path, checkpoint_ref: str) -> Path:
    checkpoint_dir = Path(checkpoint_ref)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = repo_root / checkpoint_dir
    model_dir = checkpoint_dir / "model"
    if not (model_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Training checkpoint is missing adapter artifacts: {model_dir}. "
            "Expected a checkpoint directory containing model/adapter_config.json."
        )
    return checkpoint_dir


def _load_lora_from_training_checkpoint(
    model: Any,
    config: ExperimentConfig,
    checkpoint_dir: Path,
) -> LoraApplicationResult:
    if not config.lora.enabled:
        raise ValueError("runtime.resume_from_checkpoint requires lora.enabled=true.")
    target_modules = resolve_target_modules(config.lora)
    matched_modules, unmatched_targets = find_target_modules(model, target_modules)
    peft_model = PeftModel.from_pretrained(
        model,
        checkpoint_dir / "model",
        is_trainable=True,
    )
    trainable_parameters = count_trainable_parameters(peft_model)
    if trainable_parameters > config.lora.trainable_parameter_cap:
        raise ValueError(
            "Resumed LoRA setup exceeds trainable parameter cap: "
            f"{trainable_parameters} > {config.lora.trainable_parameter_cap}"
        )
    print(f"Resumed trainable LoRA adapter from checkpoint: {checkpoint_dir}")
    print(f"Trainable parameter count: {trainable_parameters}")
    return LoraApplicationResult(
        model=peft_model,
        trainable_parameters=trainable_parameters,
        matched_modules=matched_modules,
        unmatched_targets=unmatched_targets,
        target_modules=target_modules,
        adapter_enabled=True,
    )


def _build_dataset(
    config: ExperimentConfig,
    formatter: PromptFormatter,
    split: str,
    limit: int | None = None,
    examples: list[Any] | None = None,
) -> ScienceQADataset:
    image_transform = build_image_transform(config.image, augment=split == "train")
    dataset_examples = examples
    if dataset_examples is None:
        dataset_examples = _load_examples_for_config(config, split, limit=limit)
    return ScienceQADataset(
        dataset_examples,
        config=config,
        formatter=formatter,
        split=split,
        image_transform=image_transform,
    )


def _load_examples_for_config(
    config: ExperimentConfig,
    split: str,
    limit: int | None = None,
) -> list[Any]:
    examples = load_split_examples(Path(config.data.data_dir), split, limit=limit)
    return attach_image_captions(examples, config, split)


def _build_training_datasets(
    config: ExperimentConfig,
    formatter: PromptFormatter,
) -> tuple[ScienceQADataset, ScienceQADataset | None, ScienceQADataset | None]:
    train_examples = _load_examples_for_config(config, "train", limit=config.runtime.max_train_examples)
    val_examples = _load_examples_for_config(config, "val", limit=config.runtime.max_val_examples)

    if config.runtime.final_retrain:
        train_examples = combine_examples(train_examples, val_examples)
        val_dataset = None
    else:
        val_dataset = _build_dataset(config, formatter, split="val", examples=val_examples)

    train_dataset = _build_dataset(config, formatter, split="train", examples=train_examples)
    test_dataset = None
    if config.runtime.predict_test:
        test_dataset = _build_dataset(
            config,
            formatter,
            split="test",
            limit=config.runtime.max_test_examples,
        )
    return train_dataset, val_dataset, test_dataset


def _build_eval_datasets(
    config: ExperimentConfig,
    formatter: PromptFormatter,
) -> tuple[ScienceQADataset | None, ScienceQADataset | None]:
    val_dataset = None
    if config.runtime.max_val_examples != 0:
        val_dataset = _build_dataset(
            config,
            formatter,
            split="val",
            limit=config.runtime.max_val_examples,
        )
    test_dataset = None
    if config.runtime.predict_test:
        test_dataset = _build_dataset(
            config,
            formatter,
            split="test",
            limit=config.runtime.max_test_examples,
        )
    return val_dataset, test_dataset


def _run_training_epochs(
    model_bundle: ModelBundle,
    config: ExperimentConfig,
    train_dataset: ScienceQADataset,
    extra_epochs: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    checkpoint_root: Path | None = None,
) -> tuple[torch.optim.Optimizer, Any]:
    total_epochs = extra_epochs if extra_epochs is not None else config.training.epochs
    if total_epochs <= 0:
        raise ValueError("Training loop was requested with zero epochs. Use config.is_eval_only to skip training.")

    collator = TrainingCollator(model_bundle.processor)
    sampler = build_training_sampler(train_dataset, config.sampling.mode)
    sample_weights = build_loss_weights(train_dataset, config.sampling.mode)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.runtime.num_workers,
        collate_fn=collator,
    )
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            params=[parameter for parameter in model_bundle.model.parameters() if parameter.requires_grad],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    total_steps = max(1, math.ceil(len(train_loader) * total_epochs / config.training.gradient_accumulation))
    if scheduler is None:
        warmup_steps = int(total_steps * config.training.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    scaler = GradScaler(enabled=torch.cuda.is_available() and config.training.fp16)
    model_bundle.model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch_index in range(total_epochs):
        running_loss = 0.0
        for step_index, batch in enumerate(train_loader, start=1):
            batch = _move_to_device(batch, model_bundle.device)
            labels = batch.pop("labels")
            dataset_indices = batch.pop("dataset_indices")

            with _autocast_context(config):
                outputs = model_bundle.model(**batch)
                loss = compute_weighted_loss(
                    outputs.logits,
                    labels,
                    dataset_indices,
                    sample_weights,
                )
                loss = loss / config.training.gradient_accumulation

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()
            if step_index % config.training.gradient_accumulation == 0 or step_index == len(train_loader):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % config.runtime.logging_steps == 0:
                    print(
                        f"epoch={epoch_index + 1} step={global_step} "
                        f"loss={running_loss / config.runtime.logging_steps:.4f}"
                    )
                    running_loss = 0.0
                checkpointing = config.training.checkpointing
                if (
                    checkpoint_root is not None
                    and checkpointing.enabled
                    and checkpointing.save_steps > 0
                    and global_step % checkpointing.save_steps == 0
                ):
                    _save_training_checkpoint(
                        model_bundle,
                        config,
                        checkpoint_root,
                        epoch_index,
                        global_step,
                        reason="step",
                    )
        if checkpoint_root is not None and config.training.checkpointing.enabled and config.training.checkpointing.save_epochs:
            _save_training_checkpoint(
                model_bundle,
                config,
                checkpoint_root,
                epoch_index,
                global_step,
                reason="epoch",
            )
    return optimizer, scheduler


def _run_hard_example_follow_up(
    model_bundle: ModelBundle,
    config: ExperimentConfig,
    train_dataset: ScienceQADataset,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    checkpoint_root: Path | None = None,
) -> None:
    hard_config = config.sampling.hard_example_follow_up
    if not hard_config.enabled:
        return

    scorer = _build_scorer(model_bundle, train_dataset.formatter, config)
    eval_dataset = ScienceQADataset(
        train_dataset.examples,
        config=config,
        formatter=train_dataset.formatter,
        split="train_eval",
        image_transform=train_dataset.image_transform,
    )
    _, predictions, _ = evaluate_dataset(eval_dataset, scorer)
    hard_ids = [
        prediction["id"]
        for prediction in predictions
        if prediction.get("answer") is not None and prediction["predicted_index"] != prediction["answer"]
    ]
    if hard_config.max_examples is not None:
        hard_ids = hard_ids[: hard_config.max_examples]
    if not hard_ids:
        return

    hard_examples = [example for example in train_dataset.examples if example.id in set(hard_ids)]
    follow_up_dataset = ScienceQADataset(
        hard_examples,
        config=config,
        formatter=train_dataset.formatter,
        split="train",
        image_transform=train_dataset.image_transform,
    )
    _run_training_epochs(
        model_bundle,
        config,
        follow_up_dataset,
        extra_epochs=hard_config.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_root=checkpoint_root,
    )


def _build_scorer(
    model_bundle: ModelBundle,
    formatter: PromptFormatter,
    config: ExperimentConfig,
) -> FormulationScorer:
    return FormulationScorer(
        model_bundle.model,
        model_bundle.processor,
        formatter,
        max_completion_batch_size=config.scoring.max_completion_batch_size,
    )


def _write_run_metadata(output_dir: Path, config: ExperimentConfig, summary: dict[str, Any]) -> dict[str, str]:
    resolved_config_path = _write_resolved_config(output_dir, config)
    metadata_paths = {
        "resolved_config_path": str(resolved_config_path),
        "run_summary_path": str(output_dir / "run_summary.yaml"),
    }
    with Path(metadata_paths["run_summary_path"]).open("w") as handle:
        yaml.safe_dump({**summary, **metadata_paths}, handle, sort_keys=False)
    return metadata_paths


def _write_resolved_config(output_dir: Path, config: ExperimentConfig) -> Path:
    resolved_config_path = output_dir / "resolved_config.yaml"
    with resolved_config_path.open("w") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    return resolved_config_path


def run_experiment(repo_root: Path, config: ExperimentConfig) -> dict[str, Any]:
    config = resolve_effective_config(repo_root, config)
    set_random_seed(config.runtime.seed)
    formatter = PromptFormatter(config)
    output_dir = config.output_dir
    ensure_directory(output_dir)
    (output_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    _write_resolved_config(output_dir, config)
    train_dataset = None

    if config.runtime.eval_artifact_dir:
        print(f"Running artifact-backed eval-only mode for {config.run_name}.")
        eval_artifact_dir = resolve_eval_artifact_dir(repo_root, config.runtime.eval_artifact_dir)
        val_dataset, test_dataset = _build_eval_datasets(config, formatter)
        model_bundle = load_model_and_processor_from_artifacts(config, eval_artifact_dir)
        trainable_parameters = 0
        lora_result = LoraApplicationResult(
            model=model_bundle.model,
            trainable_parameters=0,
            matched_modules=[],
            unmatched_targets=[],
            target_modules=[],
            adapter_enabled=(eval_artifact_dir / "model" / "adapter_config.json").exists(),
        )
    else:
        if config.is_eval_only:
            val_dataset, test_dataset = _build_eval_datasets(config, formatter)
        else:
            train_dataset, val_dataset, test_dataset = _build_training_datasets(config, formatter)
        model_bundle = load_model_and_processor(config)
        if config.runtime.resume_from_checkpoint:
            checkpoint_dir = _resolve_training_checkpoint_dir(repo_root, config.runtime.resume_from_checkpoint)
            lora_result = _load_lora_from_training_checkpoint(
                model_bundle.model,
                config,
                checkpoint_dir,
            )
        else:
            lora_result = apply_lora(
                model_bundle.model,
                config.lora,
                will_train=not config.is_eval_only,
            )
        model_bundle.model = lora_result.model
        trainable_parameters = lora_result.trainable_parameters

    if config.is_eval_only:
        print(f"Running eval-only mode for {config.run_name}. Training loop is skipped.")
    else:
        checkpoint_root = _checkpoint_root(output_dir) if config.training.checkpointing.enabled else None
        optimizer, scheduler = _run_training_epochs(
            model_bundle,
            config,
            train_dataset,
            checkpoint_root=checkpoint_root,
        )
        _run_hard_example_follow_up(
            model_bundle,
            config,
            train_dataset,
            optimizer,
            scheduler,
            checkpoint_root=checkpoint_root,
        )

        model_dir = output_dir / "model"
        processor_dir = output_dir / "processor"
        model_bundle.model.save_pretrained(model_dir)
        model_bundle.processor.save_pretrained(processor_dir)

    latest_checkpoint = _latest_checkpoint_dir(_checkpoint_root(output_dir))
    scorer = _build_scorer(model_bundle, formatter, config)
    val_metrics = None
    val_predictions_path = None
    if val_dataset is not None:
        val_output_path = output_dir / "artifacts" / "val_predictions.jsonl" if config.runtime.save_predictions else None
        val_metrics, _, val_predictions_path = evaluate_dataset(
            val_dataset,
            scorer,
            output_path=val_output_path,
            batch_size=config.training.eval_batch_size,
            progress_label="val",
        )

    test_predictions_path = None
    submission_path = None
    if test_dataset is not None:
        test_output_path = output_dir / "artifacts" / "test_predictions.jsonl" if config.runtime.save_predictions else None
        _, test_predictions, test_predictions_path = evaluate_dataset(
            test_dataset,
            scorer,
            output_path=test_output_path,
            batch_size=config.training.eval_batch_size,
            progress_label="test",
        )
        submission_path = save_submission(test_predictions, output_dir / "artifacts" / "submission.csv")

    summary = {
        "trainable_parameters": trainable_parameters,
        "matched_lora_modules": lora_result.matched_modules,
        "unmatched_lora_targets": lora_result.unmatched_targets,
        "target_lora_modules": lora_result.target_modules,
        "adapter_enabled": lora_result.adapter_enabled,
        "run_mode": "eval_only" if config.is_eval_only else "train_eval",
        "val_metrics": val_metrics,
        "val_predictions_path": str(val_predictions_path) if val_predictions_path else None,
        "test_predictions_path": str(test_predictions_path) if test_predictions_path else None,
        "submission_path": str(submission_path) if submission_path else None,
        "eval_artifact_dir": config.runtime.eval_artifact_dir,
        "resume_from_checkpoint": config.runtime.resume_from_checkpoint,
        "latest_checkpoint_dir": str(latest_checkpoint) if latest_checkpoint else None,
    }
    summary.update(_write_run_metadata(output_dir, config, summary))
    append_results_row(build_results_row(repo_root, config, summary))
    return summary
