"""Eval harness: compare a base model against its fine-tuned adapter on a
held-out set, with exact-match and task-specific scorers.

Scorers are pure stdlib and individually testable; generation needs the GPU
stack and is loaded lazily.
"""
from __future__ import annotations

import json
import re


def normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def normalize_sql(s: str) -> str:
    s = s.strip().rstrip(";")
    s = re.sub(r"\s+", " ", s).lower()
    return s


def sql_exact_match(pred: str, gold: str) -> bool:
    return normalize_sql(pred) == normalize_sql(gold)


def extract_after_marker(text: str, marker: str, stops: tuple[str, ...] = ()) -> str:
    """Cut generation at a marker and optional stop strings; strips code
    fences. Shared by SQL/JSON/function-call style tasks."""
    if marker and marker in text:
        text = text.split(marker, 1)[0]
    for stop in stops:
        if stop in text:
            text = text.split(stop, 1)[0]
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def compare_runs(base_result: dict, tuned_result: dict) -> dict:
    """Summarize two score dicts into a before/after table."""
    out = {"n": base_result.get("n"), "metrics": {}}
    for k, v in base_result.items():
        if isinstance(v, (int, float)) and k != "n":
            out["metrics"][k] = {
                "base": v,
                "finetuned": tuned_result.get(k),
                "delta": (tuned_result.get(k) - v)
                if isinstance(tuned_result.get(k), (int, float)) else None,
            }
    return out


def run_generations(model, tok, prompts: list[str], max_new_tokens: int = 192) -> list[str]:
    """Greedy generation for a list of prompts (GPU path)."""
    import torch
    outs = []
    for p in prompts:
        inputs = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True))
    return outs
