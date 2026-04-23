from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.config import load_experiment_config
from src.data.dataset import ScienceQAExample
from src.modeling.lora import LoraApplicationResult
from src.modeling.load_model import _configure_processor, load_model_and_processor_from_artifacts
from src.prompting.formatter import PromptFormatter
from src.training import train as train_module


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
        self.image_processor = SimpleNamespace()


def test_configure_processor_only_sets_missing_pad_token() -> None:
    processor = FakeProcessor()
    processor.image_processor.do_image_splitting = True
    processor.image_processor.do_resize = True
    processor.image_processor.size = {"longest_edge": 2048}
    processor.image_processor.max_image_size = {"longest_edge": 512}

    config = load_experiment_config(REPO_ROOT, "f02_multimodal_index")
    _configure_processor(processor, config)

    assert processor.tokenizer.pad_token == "<eos>"
    assert processor.image_processor.do_image_splitting is True
    assert processor.image_processor.do_resize is True
    assert processor.image_processor.size == {"longest_edge": 2048}
    assert processor.image_processor.max_image_size == {"longest_edge": 512}


class FakeModel:
    def __init__(self, source: str | Path) -> None:
        self.source = source
        self.config = SimpleNamespace(use_cache=True)
        self.gradient_checkpointing_enabled = False

    def to(self, device: object) -> "FakeModel":
        self.device = device
        return self

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    def parameters(self) -> list[object]:
        return []


def test_artifact_loader_prefers_saved_processor_and_adapter(monkeypatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "outputs" / "source_run"
    model_dir = artifact_dir / "model"
    processor_dir = artifact_dir / "processor"
    model_dir.mkdir(parents=True)
    processor_dir.mkdir(parents=True)
    (model_dir / "adapter_config.json").write_text("{}")

    config = load_experiment_config(
        REPO_ROOT,
        "f02_multimodal_index",
        cli_overrides=["training.epochs=0", f"runtime.eval_artifact_dir={artifact_dir}"],
    )
    calls: dict[str, object] = {}

    def fake_processor_loader(source: str | Path, **kwargs: object) -> FakeProcessor:
        calls["processor"] = source
        return FakeProcessor()

    def fake_model_loader(source: str | Path, **kwargs: object) -> FakeModel:
        calls["model"] = source
        return FakeModel(source)

    def fake_adapter_loader(base_model: FakeModel, source: str | Path) -> FakeModel:
        calls["adapter"] = source
        return base_model

    monkeypatch.setattr("src.modeling.load_model.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "src.modeling.load_model.AutoProcessor.from_pretrained",
        fake_processor_loader,
    )
    monkeypatch.setattr(
        "src.modeling.load_model.AutoModelForImageTextToText.from_pretrained",
        fake_model_loader,
    )
    monkeypatch.setattr(
        "src.modeling.load_model.PeftModel.from_pretrained",
        fake_adapter_loader,
    )

    bundle = load_model_and_processor_from_artifacts(config, artifact_dir)

    assert calls["processor"] == processor_dir
    assert calls["model"] == config.model.model_id
    assert calls["adapter"] == model_dir
    assert bundle.model.gradient_checkpointing_enabled is True


def test_artifact_loader_falls_back_to_saved_model_dir_without_adapter(monkeypatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "outputs" / "source_run"
    model_dir = artifact_dir / "model"
    model_dir.mkdir(parents=True)

    config = load_experiment_config(
        REPO_ROOT,
        "f02_multimodal_index",
        cli_overrides=["training.epochs=0", f"runtime.eval_artifact_dir={artifact_dir}"],
    )
    calls: dict[str, object] = {}

    def fake_processor_loader(source: str | Path, **kwargs: object) -> FakeProcessor:
        calls["processor"] = source
        return FakeProcessor()

    def fake_model_loader(source: str | Path, **kwargs: object) -> FakeModel:
        calls["model"] = source
        return FakeModel(source)

    monkeypatch.setattr("src.modeling.load_model.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "src.modeling.load_model.AutoProcessor.from_pretrained",
        fake_processor_loader,
    )
    monkeypatch.setattr(
        "src.modeling.load_model.AutoModelForImageTextToText.from_pretrained",
        fake_model_loader,
    )

    load_model_and_processor_from_artifacts(config, artifact_dir)

    assert calls["processor"] == config.model.model_id
    assert calls["model"] == model_dir


def test_eval_only_dataset_builder_skips_train_split(monkeypatch) -> None:
    config = load_experiment_config(
        REPO_ROOT,
        "f02_multimodal_index",
        cli_overrides=["training.epochs=0", "runtime.predict_test=true"],
    )
    formatter = PromptFormatter(config)
    seen_splits: list[str] = []

    def fake_load_split_examples(data_dir: Path, split: str, limit: int | None = None) -> list[ScienceQAExample]:
        seen_splits.append(split)
        if split == "train":
            raise AssertionError("eval-only dataset construction should not load the train split")
        return [
            ScienceQAExample(
                id=f"{split}_001",
                image_path=None,
                question="Which answer is correct?",
                choices=["A", "B"],
                num_choices=2,
                answer=0 if split == "val" else None,
            )
        ]

    monkeypatch.setattr(train_module, "load_split_examples", fake_load_split_examples)

    val_dataset, test_dataset = train_module._build_eval_datasets(config, formatter)

    assert seen_splits == ["val", "test"]
    assert len(val_dataset) == 1
    assert test_dataset is not None
    assert len(test_dataset) == 1


def test_candidate_yes_no_eval_dataset_keeps_one_item_per_example() -> None:
    config = load_experiment_config(
        REPO_ROOT,
        "f04_candidate_yes_no",
        cli_overrides=["training.epochs=0"],
    )
    formatter = PromptFormatter(config)
    dataset = train_module._build_dataset(
        config,
        formatter,
        split="val",
        examples=[
            ScienceQAExample(
                id="val_001",
                image_path=None,
                question="Which answer is correct?",
                choices=["A", "B", "C"],
                num_choices=3,
                answer=1,
            )
        ],
    )

    assert len(dataset) == 1
    item = dataset[0]
    assert item["candidate_index"] is None
    assert item["prompt_text"] == ""
    assert item["metadata"]["num_choices"] == 3


def test_missing_image_path_uses_blank_image_when_image_field_enabled() -> None:
    config = load_experiment_config(
        REPO_ROOT,
        "f02_multimodal_index",
        cli_overrides=["image.target_long_edge=32"],
    )
    formatter = PromptFormatter(config)
    dataset = train_module._build_dataset(
        config,
        formatter,
        split="val",
        examples=[
            ScienceQAExample(
                id="val_002",
                image_path=None,
                question="Which answer is correct?",
                choices=["A", "B"],
                num_choices=2,
                answer=0,
            )
        ],
    )

    item = dataset[0]
    assert item["image"] is not None
    assert item["image"].size == (32, 32)


def test_plain_eval_only_run_uses_eval_datasets_without_train_split(monkeypatch, tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    config = load_experiment_config(
        REPO_ROOT,
        "f02_multimodal_index",
        cli_overrides=["training.epochs=0", "lora.enabled=false", "runtime.predict_test=true"],
        output_dir=str(output_root),
    )
    seen: dict[str, object] = {}

    def fail_build_training_datasets(*args: object, **kwargs: object) -> object:
        raise AssertionError("plain eval-only runs should not build training datasets")

    def fake_build_eval_datasets(*args: object, **kwargs: object) -> tuple[list[str], list[str]]:
        seen["used_eval_builder"] = True
        return ["val_dataset"], ["test_dataset"]

    monkeypatch.setattr(train_module, "_build_training_datasets", fail_build_training_datasets)
    monkeypatch.setattr(train_module, "_build_eval_datasets", fake_build_eval_datasets)
    monkeypatch.setattr(
        train_module,
        "load_model_and_processor",
        lambda config: SimpleNamespace(model=object(), processor=object(), device="cpu", dtype=None),
    )
    monkeypatch.setattr(
        train_module,
        "apply_lora",
        lambda model, config, will_train: LoraApplicationResult(
            model=model,
            trainable_parameters=0,
            matched_modules=[],
            unmatched_targets=[],
            target_modules=[],
            adapter_enabled=False,
        ),
    )
    monkeypatch.setattr(train_module, "FormulationScorer", lambda model, processor, formatter: object())

    def fake_evaluate_dataset(dataset: object, scorer: object, output_path: Path | None = None, batch_size: int = 1):
        calls = seen.setdefault("datasets", [])
        assert isinstance(calls, list)
        calls.append(dataset)
        seen["eval_batch_size"] = batch_size
        if dataset == ["val_dataset"]:
            return {"overall_accuracy": 1.0}, [], None
        return {}, [], None

    monkeypatch.setattr(train_module, "evaluate_dataset", fake_evaluate_dataset)
    monkeypatch.setattr(train_module, "save_submission", lambda predictions, path: path)
    monkeypatch.setattr(train_module, "append_results_row", lambda row: None)

    summary = train_module.run_experiment(REPO_ROOT, config)

    assert seen["used_eval_builder"] is True
    assert seen["datasets"] == [["val_dataset"], ["test_dataset"]]
    assert seen["eval_batch_size"] == config.training.eval_batch_size
    assert summary["run_mode"] == "eval_only"
