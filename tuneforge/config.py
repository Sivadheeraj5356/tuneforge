from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import math


@dataclass
class TrainConfig:
    model: str
    dataset: str
    output_dir: str = "runs/tune"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    load_in_4bit: bool = True
    learning_rate: float = 2e-4
    epochs: float = 1.0
    max_steps: int = -1
    per_device_batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 1024
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 20
    seed: int = 42
    gradient_checkpointing: bool = True
    packing: bool = False
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    text_field: str | None = None

    def validate(self) -> list[str]:
        errs: list[str] = []
        for name in ("model", "dataset", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                errs.append(f"{name} must be a nonempty string")
        for name in ("lora_r", "lora_alpha", "per_device_batch_size", "grad_accum",
                     "logging_steps", "save_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                errs.append(f"{name} must be a positive integer")
        for name in ("learning_rate", "epochs"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                errs.append(f"{name} must be finite and > 0")
        if not isinstance(self.lora_dropout, (int, float)) or not 0 <= self.lora_dropout < 1:
            errs.append("lora_dropout must be in [0, 1)")
        if not isinstance(self.warmup_ratio, (int, float)) or not 0 <= self.warmup_ratio <= 1:
            errs.append("warmup_ratio must be in [0, 1]")
        if type(self.max_steps) is not int or (self.max_steps != -1 and self.max_steps < 1):
            errs.append("max_steps must be -1 or a positive integer")
        if type(self.max_seq_len) is not int or self.max_seq_len < 32:
            errs.append("max_seq_len must be an integer >= 32")
        if not isinstance(self.target_modules, (list, tuple)) or not self.target_modules or any(
                not isinstance(x, str) or not x.strip() for x in self.target_modules):
            errs.append("target_modules must be a nonempty list of names")
        for name in ("load_in_4bit", "gradient_checkpointing", "packing"):
            if type(getattr(self, name)) is not bool:
                errs.append(f"{name} must be a boolean")
        return errs

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.grad_accum

    def to_json(self) -> str:
        d = asdict(self)
        d["target_modules"] = list(self.target_modules)
        return json.dumps(d, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "TrainConfig":
        d = json.loads(s)
        if isinstance(d.get("target_modules"), list):
            d["target_modules"] = tuple(d["target_modules"])
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f)
        if not isinstance(d, dict):
            raise ValueError("config must be a YAML mapping")
        for key in ("learning_rate", "epochs", "warmup_ratio", "lora_dropout"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = float(d[key])
                except ValueError:
                    pass
        return cls.from_json(json.dumps(d))
