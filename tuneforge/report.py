"""Reproducible run cards: a markdown record of what was trained, from the
files a run leaves behind (config, adapter config, trainer state), plus the
machine it ran on. Missing files are listed, not guessed.
"""
from __future__ import annotations

import json
import os
import platform
import sys


def _read_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _read_text(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _env_lines() -> list[str]:
    lines = [f"- Python {platform.python_version()} on {platform.platform()}"]
    try:
        import torch
        lines.append(f"- torch {torch.__version__}")
        if torch.cuda.is_available():
            lines.append(f"- GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        lines.append("- torch not importable on this machine")
    return lines


def build_run_card(run_dir: str) -> str:
    missing: list[str] = []
    out: list[str] = []
    out.append(f"# Run card: {os.path.basename(os.path.abspath(run_dir))}")
    out.append("")

    config_text = None
    for name in ("config.yaml", "config.yml", "config.json"):
        config_text = _read_text(os.path.join(run_dir, name))
        if config_text is not None:
            break
    if config_text is None:
        missing.append("training config (config.yaml/config.json)")

    adapter = _read_json(os.path.join(run_dir, "adapter_config.json"))
    state = _read_json(os.path.join(run_dir, "trainer_state.json"))
    if state is None:
        for root, _dirs, files in os.walk(run_dir):
            if "trainer_state.json" in files:
                state = _read_json(os.path.join(root, "trainer_state.json"))
                break
    if state is None:
        missing.append("trainer_state.json (metrics)")

    out.append("## Model and adapter")
    if adapter:
        out.append(f"- base model: {adapter.get('base_model_name_or_path', 'unknown')}")
        out.append(f"- LoRA r={adapter.get('r')}, alpha={adapter.get('lora_alpha')}, "
                   f"dropout={adapter.get('lora_dropout')}")
        targets = adapter.get("target_modules")
        if targets:
            out.append(f"- target modules: {', '.join(targets)}")
    else:
        out.append("- adapter_config.json not found; adapter settings not recorded")
        missing.append("adapter_config.json")
    out.append("")

    if config_text is not None:
        out.append("## Training config")
        out.append("```")
        out.append(config_text.strip())
        out.append("```")
        out.append("")

    out.append("## Metrics")
    if state:
        history = state.get("log_history") or []
        train_losses = [h for h in history if "loss" in h]
        eval_entries = [h for h in history if "eval_loss" in h]
        if train_losses:
            last = train_losses[-1]
            out.append(f"- final train loss {last['loss']} at step "
                       f"{last.get('step', '?')}")
        if eval_entries:
            last = eval_entries[-1]
            out.append(f"- final eval loss {last['eval_loss']} at step "
                       f"{last.get('step', '?')}")
        if not train_losses and not eval_entries:
            out.append("- trainer_state.json has no loss entries")
        out.append(f"- total steps recorded: {state.get('global_step', '?')}")
    else:
        out.append("- not recorded")
    out.append("")

    out.append("## Environment")
    out.extend(_env_lines())
    out.append("")

    if missing:
        out.append("## Not recorded")
        for m in missing:
            out.append(f"- {m}")
        out.append("")
    return "\n".join(out)


def write_run_card(run_dir: str, out_path: str | None = None) -> str:
    if out_path is None:
        out_path = os.path.join(run_dir, "RUN_CARD.md")
    card = build_run_card(run_dir)
    with open(out_path, "w") as f:
        f.write(card)
    return out_path
