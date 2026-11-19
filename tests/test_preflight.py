import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import json

import pytest

from tuneforge.cli import main
from tuneforge.config import TrainConfig
from tuneforge.preflight import run_preflight


def _dataset(tmp_path, rows=120, field="prompt"):
    ds = tmp_path / "train.jsonl"
    ds.write_text("\n".join(
        json.dumps({field: f"p{i}", "completion": f"c{i}"}) for i in range(rows)))
    return str(ds)


def _cfg(tmp_path, **kw):
    base = dict(model="Qwen/Qwen3-1.7B", dataset=_dataset(tmp_path))
    base.update(kw)
    return TrainConfig(**base)


def _patch_deps(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())


def test_preflight_clean_run_passes(tmp_path, monkeypatch, capsys):
    _patch_deps(monkeypatch)
    rc = main(["preflight", "--model", "Qwen/Qwen3-1.7B",
               "--dataset", _dataset(tmp_path), "--gpu-gb", "16"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0 failed" in out


def test_preflight_missing_dataset_fails(tmp_path, monkeypatch, capsys):
    _patch_deps(monkeypatch)
    rc = main(["preflight", "--model", "Qwen/Qwen3-1.7B",
               "--dataset", str(tmp_path / "nope.jsonl"), "--gpu-gb", "16"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "does not exist" in out


def test_preflight_wrong_fields_fail(tmp_path, monkeypatch, capsys):
    _patch_deps(monkeypatch)
    rc = main(["preflight", "--model", "Qwen/Qwen3-1.7B",
               "--dataset", _dataset(tmp_path, field="instruction"),
               "--gpu-gb", "16"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "missing from first row" in out


def test_preflight_small_gpu_fails(tmp_path, monkeypatch, capsys):
    _patch_deps(monkeypatch)
    rc = main(["preflight", "--model", "Qwen/Qwen3-8B",
               "--dataset", _dataset(tmp_path), "--gpu-gb", "4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "memory" in out


def test_preflight_hub_dataset_warns_not_fails(tmp_path, monkeypatch, capsys):
    _patch_deps(monkeypatch)
    rc = main(["preflight", "--model", "Qwen/Qwen3-1.7B",
               "--dataset", "tatsu-lab/alpaca", "--gpu-gb", "16"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "WARN" in out


def test_preflight_missing_deps_fail(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    rc = main(["preflight", "--model", "Qwen/Qwen3-1.7B",
               "--dataset", _dataset(tmp_path), "--gpu-gb", "16"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not importable" in out


def test_run_preflight_counts(tmp_path, monkeypatch):
    _patch_deps(monkeypatch)
    result = run_preflight(_cfg(tmp_path), gpu_gb=16)
    counts = result.counts()
    assert counts["fail"] == 0
    assert result.ok
