"""Dataset loading and formatting for SFT. Tokenizer-free helpers are pure
stdlib so they can be tested anywhere; loading touches HF datasets lazily.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass


@dataclass
class Example:
    prompt: str
    completion: str


def render_chat(messages: list[dict], template: str = "chatml") -> str:
    """Render a chat conversation to a training string. ChatML-style by
    default; extend templates here rather than per-script."""
    if template != "chatml":
        raise ValueError(f"unknown template: {template}")
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(out)


def pack_sequences(texts: list[str], max_len_tokens_estimate: int,
                   chars_per_token: float = 4.0) -> list[str]:
    """Greedy first-fit packing of short samples into longer training
    sequences, using a character budget as a token estimate. Packing raises
    GPU utilization for short-example datasets; the estimate only decides the
    packing, never truncates content."""
    budget = int(max_len_tokens_estimate * chars_per_token)
    bins: list[list[str]] = []
    sizes: list[int] = []
    for t in sorted(texts, key=len, reverse=True):
        placed = False
        for i, s in enumerate(sizes):
            if s + len(t) + 1 <= budget:
                bins[i].append(t)
                sizes[i] += len(t) + 1
                placed = True
                break
        if not placed:
            bins.append([t])
            sizes.append(len(t))
    return ["\n".join(b) for b in bins]


def train_eval_split(rows: list[dict], eval_fraction: float = 0.05,
                     seed: int = 42, key: str | None = None) -> tuple[list, list]:
    """Deterministic split; when `key` names a field, all rows sharing a key
    value stay on the same side (blocks near-duplicate leakage, e.g. same
    database schema or same dialogue)."""
    rng = random.Random(seed)
    if key:
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(str(r.get(key, "")), []).append(r)
        keys = list(groups)
        rng.shuffle(keys)
        cut = max(1, int(len(keys) * (1 - eval_fraction)))
        train = [r for k in keys[:cut] for r in groups[k]]
        eval_rows = [r for k in keys[cut:] for r in groups[k]]
        return train, eval_rows
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - eval_fraction))
    return [rows[i] for i in idx[:cut]], [rows[i] for i in idx[cut:]]


def load_rows(dataset: str, max_rows: int | None = None) -> list[dict]:
    """Load from HF hub id or local:/path (.jsonl or .json)."""
    if dataset.startswith("local:"):
        path = dataset[len("local:"):]
        if path.endswith(".jsonl"):
            rows = [json.loads(l) for l in open(path) if l.strip()]
        else:
            rows = json.load(open(path))
    else:
        from datasets import load_dataset  # lazy: needs the lib + network
        rows = list(load_dataset(dataset, split="train"))
    return rows[:max_rows] if max_rows else rows
