"""Pre-training checks: config validity, dataset shape, dependency
availability, and memory fit, so a bad run fails before GPU time starts.
Pure stdlib plus lazy lookups, so it runs anywhere including CPU boxes.
"""
from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass, field

from tuneforge.config import TrainConfig

DEPENDENCIES = (
    "torch", "transformers", "datasets", "peft", "trl",
    "bitsandbytes", "yaml",
)

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class PreflightResult:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def ok(self) -> bool:
        return all(c.status != FAIL for c in self.checks)

    def counts(self) -> dict[str, int]:
        out = {PASS: 0, WARN: 0, FAIL: 0}
        for c in self.checks:
            out[c.status] += 1
        return out


def _config_checks(cfg: TrainConfig, result: PreflightResult) -> None:
    errs = cfg.validate()
    if errs:
        for e in errs:
            result.add("config", FAIL, e)
    else:
        result.add("config", PASS, f"{cfg.model} with r={cfg.lora_r}, "
                                   f"{cfg.epochs} epoch(s), seq {cfg.max_seq_len}")


def _looks_like_path(dataset: str) -> bool:
    if os.path.exists(dataset):
        return True
    if dataset.endswith((".json", ".jsonl", ".csv")):
        return True
    if dataset.startswith((".", "~", os.sep)):
        return True
    return dataset.count(os.sep) >= 2


def _dataset_checks(cfg: TrainConfig, result: PreflightResult) -> None:
    ds = cfg.dataset[len("local:"):] if cfg.dataset.startswith("local:") else cfg.dataset
    if not _looks_like_path(ds):
        result.add("dataset", WARN,
                   f"{ds} is a hub name; it is downloaded at train time and "
                   f"was not checked here")
        return
    if not os.path.exists(ds):
        result.add("dataset", FAIL, f"{ds} does not exist")
        return
    if ds.endswith(".csv"):
        result.add("dataset", WARN, "CSV files need a text_field; "
                                    "JSONL with prompt/completion is the tested path")
        return
    try:
        with open(ds) as f:
            lines = f.read().splitlines()
    except OSError as exc:
        result.add("dataset", FAIL, f"cannot read {ds}: {exc}")
        return
    rows = [ln for ln in lines if ln.strip()]
    if not rows:
        result.add("dataset", FAIL, f"{ds} is empty")
        return
    try:
        first = json.loads(rows[0])
    except json.JSONDecodeError as exc:
        result.add("dataset", FAIL, f"first row is not JSON: {exc}")
        return
    if not isinstance(first, dict):
        result.add("dataset", FAIL, "rows must be JSON objects")
        return
    if cfg.text_field is not None:
        if cfg.text_field in first:
            result.add("dataset", PASS,
                       f"{len(rows)} rows, text field '{cfg.text_field}' present")
        else:
            result.add("dataset", FAIL,
                       f"text_field '{cfg.text_field}' missing from first row "
                       f"(has {sorted(first)})")
        return
    missing = [k for k in (cfg.prompt_field, cfg.completion_field) if k not in first]
    if missing:
        result.add("dataset", FAIL,
                   f"fields {missing} missing from first row (has {sorted(first)})")
    else:
        result.add("dataset", PASS,
                   f"{len(rows)} rows with {cfg.prompt_field}/{cfg.completion_field}")
    if len(rows) < 100:
        result.add("dataset", WARN,
                   f"only {len(rows)} rows; expect overfitting with more than a "
                   f"few steps")


def _dependency_checks(result: PreflightResult) -> None:
    missing = [d for d in DEPENDENCIES if importlib.util.find_spec(d) is None]
    if missing:
        result.add("dependencies", FAIL,
                   f"not importable: {', '.join(missing)}")
    else:
        result.add("dependencies", PASS,
                   f"all {len(DEPENDENCIES)} training dependencies importable")


def _detect_gpu_gb() -> float | None:
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return round(props.total_memory / 1e9, 1)
    except Exception:
        return None


def _memory_checks(cfg: TrainConfig, gpu_gb: float | None,
                   result: PreflightResult) -> None:
    from tuneforge.vram import KNOWN_MODELS, plan
    if gpu_gb is None:
        gpu_gb = _detect_gpu_gb()
    if gpu_gb is None:
        result.add("memory", WARN,
                   "no GPU detected and --gpu-gb not given; memory fit not "
                   "checked. Training needs an NVIDIA/AMD/Intel GPU or a Mac.")
        return
    info = KNOWN_MODELS.get(cfg.model)
    if info is None:
        result.add("memory", WARN,
                   f"{cfg.model} is not in the known-model table; run "
                   f"`tuneforge plan {cfg.model} --gpu-gb {gpu_gb} "
                   f"--params-b ... --layers ... --hidden ...` to check fit")
        return
    p = plan(info, gpu_gb, load_in_4bit=cfg.load_in_4bit, lora_r=cfg.lora_r,
             batch=cfg.per_device_batch_size, seq_len=cfg.max_seq_len)
    if p.fits:
        result.add("memory", PASS,
                   f"estimated {p.total_gb} GB vs {gpu_gb} GB GPU "
                   f"(headroom {p.headroom_gb} GB)")
    else:
        result.add("memory", FAIL,
                   f"estimated {p.total_gb} GB vs {gpu_gb} GB GPU; "
                   f"first advice: {p.advice[0] if p.advice else 'shrink the plan'}")


def run_preflight(cfg: TrainConfig, gpu_gb: float | None = None) -> PreflightResult:
    result = PreflightResult()
    _config_checks(cfg, result)
    _dataset_checks(cfg, result)
    _dependency_checks(result)
    _memory_checks(cfg, gpu_gb, result)
    return result
