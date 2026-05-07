from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import load_experiment_config
from src.modeling import lora as lora_module


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeParameter:
    def __init__(self, size: int, requires_grad: bool = True) -> None:
        self.size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.size


class FakeModel:
    def __init__(self, trainable_size: int = 123) -> None:
        self.trainable_size = trainable_size

    def named_modules(self):
        return [
            ("model.text_model.layers.0.self_attn.q_proj", object()),
            ("model.text_model.layers.0.self_attn.v_proj", object()),
            ("model.text_model.layers.0.mlp.gate_proj", object()),
            ("model.text_model.layers.0.mlp.up_proj", object()),
            ("model.text_model.layers.0.mlp.down_proj", object()),
        ]

    def parameters(self):
        return [FakeParameter(self.trainable_size)]


def test_apply_lora_passes_dora_to_peft_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_lora_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(lora_module, "LoraConfig", fake_lora_config)
    monkeypatch.setattr(lora_module, "get_peft_model", lambda model, config: model)
    config = load_experiment_config(
        REPO_ROOT,
        "c03_balanced_answer",
        cli_overrides=["lora.use_dora=true"],
    )

    result = lora_module.apply_lora(FakeModel(), config.lora, will_train=True)

    assert captured["use_dora"] is True
    assert result.trainable_parameters == 123


def test_lora_trainable_parameter_cap_still_raises(monkeypatch) -> None:
    monkeypatch.setattr(lora_module, "LoraConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(lora_module, "get_peft_model", lambda model, config: FakeModel(trainable_size=6_000_000))
    config = load_experiment_config(REPO_ROOT, "ta02_rank16_dora_qv_mlp_capcheck")

    with pytest.raises(ValueError, match="exceeds trainable parameter cap"):
        lora_module.apply_lora(FakeModel(), config.lora, will_train=True)
