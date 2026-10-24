"""Training configuration: one dataclass, YAML/JSON round-trip, validation.

Pure stdlib so configs can be built, checked, and diffed anywhere, including
machines without a GPU or torch installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json


@dataclass
class TrainConfig:
    model: str                                  # HF id or local path
    dataset: str                                # HF id, or local:/path.jsonl
    output_dir: str = "runs/tune"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    load_in_4bit: bool = True                   # QLoRA-style NF4 base weights
    # optimization
    learning_rate: float = 2e-4
    epochs: float = 1.0
    max_steps: int = -1                         # -1 = use epochs
    per_device_batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 1024
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 20
    seed: int = 42
    gradient_checkpointing: bool = True
    packing: bool = False
    # data mapping: which record fields form prompt/completion
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    text_field: str | None = None               # set for raw-text SFT instead

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.lora_r < 1:
            errs.append("lora_r must be >= 1")
        if not (0.0 <= self.lora_dropout < 1.0):
            errs.append("lora_dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            errs.append("learning_rate must be > 0")
        if self.per_device_batch_size < 1 or self.grad_accum < 1:
            errs.append("batch sizes must be >= 1")
        if self.max_seq_len < 32:
            errs.append("max_seq_len must be >= 32")
        if self.text_field and self.text_field == self.prompt_field:
            errs.append("text_field and prompt_field must differ")
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
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        # minimal YAML subset: flat "key: value" lines, no nesting
        d: dict = {}
        for line in open(path):
            line = line.split("#", 1)[0].rstrip()
            if not line.strip() or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if v.startswith("[") and v.endswith("]"):
                d[k] = tuple(x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip())
            elif v.lower() in ("true", "false"):
                d[k] = v.lower() == "true"
            else:
                try:
                    d[k] = int(v)
                except ValueError:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v.strip("'\"")
        return cls(**d)
