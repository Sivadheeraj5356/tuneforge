import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuneforge import __version__
from tuneforge import cli, export


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_train_rejects_invalid_config(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text("model: m\ndataset: d\nlora_r: 0\n")
    assert cli.main(["train", "--config", str(p)]) == 2
    assert "invalid config" in capsys.readouterr().err


def test_generate_dispatches_to_models(monkeypatch, capsys):
    import tuneforge.models as models

    seen = {}

    def fake(base, prompt, adapter=None, max_new_tokens=128):
        seen.update(base=base, prompt=prompt, adapter=adapter, n=max_new_tokens)
        return "SELECT 1"

    monkeypatch.setattr(models, "generate", fake)
    rc = cli.main(["generate", "--base", "b", "--adapter", "a",
                   "--prompt", "hi", "--max-new-tokens", "9"])
    assert rc == 0
    assert seen == {"base": "b", "prompt": "hi", "adapter": "a", "n": 9}
    assert "SELECT 1" in capsys.readouterr().out


def test_export_dispatch_with_gguf(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(export, "merge_adapter",
                        lambda b, a, o: calls.append(("merge", b, a, o)) or o)
    monkeypatch.setattr(export, "to_gguf",
                        lambda m, o, quant, llama_cpp_dir:
                        calls.append(("gguf", m, o, quant, llama_cpp_dir)) or o)
    rc = cli.main(["export", "--base", "b", "--adapter", "a",
                   "--merged-out", "m", "--gguf-out", "x.gguf"])
    assert rc == 0
    assert calls == [("merge", "b", "a", "m"),
                     ("gguf", "m", "x.gguf", "q4_k_m", "llama.cpp")]


def test_to_gguf_requires_llama_cpp(tmp_path):
    with pytest.raises(FileNotFoundError):
        export.to_gguf("merged", str(tmp_path / "o.gguf"),
                       llama_cpp_dir=str(tmp_path / "missing"))


def test_to_gguf_missing_quantizer_fails_before_conversion(tmp_path, monkeypatch):
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("")
    ran = []
    monkeypatch.setattr(export.subprocess, "run", lambda *a, **kw: ran.append(a))
    with pytest.raises(FileNotFoundError, match="quantizer"):
        export.to_gguf("merged", str(tmp_path / "o.gguf"), llama_cpp_dir=str(llama))
    assert not ran


def test_to_gguf_f16_respects_output_filename(tmp_path, monkeypatch):
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("")
    def convert(cmd, check):
        from pathlib import Path
        Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF")
    monkeypatch.setattr(export.subprocess, "run", convert)
    output = tmp_path / "output"
    assert export.to_gguf("merged", str(output), quant="f16", llama_cpp_dir=str(llama)) == str(output)
    assert output.read_bytes() == b"GGUF"


def test_to_gguf_quantized_output_and_cleanup(tmp_path, monkeypatch):
    from pathlib import Path
    llama = tmp_path / "llama"
    (llama / "build/bin").mkdir(parents=True)
    (llama / "convert_hf_to_gguf.py").touch()
    (llama / "build/bin/llama-quantize").touch()
    ran = []
    def run(cmd, check):
        ran.append(cmd)
        if "--outfile" in cmd:
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"fp16")
        else:
            Path(cmd[2]).write_bytes(b"q4")
    monkeypatch.setattr(export.subprocess, "run", run)
    output = tmp_path / "output"
    assert export.to_gguf("merged", str(output), llama_cpp_dir=str(llama)) == str(output)
    assert output.read_bytes() == b"q4"
    assert not (tmp_path / "output.f16.gguf").exists()
    assert ran[-1][-1] == "Q4_K_M"


def test_train_without_arguments_returns_error(capsys):
    assert cli.main(["train"]) == 2
    assert "model must" in capsys.readouterr().err


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuneforge.config import TrainConfig


def test_from_yaml_parses_types(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "model: org/m\n"
        "dataset: local:rows.jsonl\n"
        "lora_r: 8\n"
        "learning_rate: 1e-4\n"
        "load_in_4bit: false\n"
        "target_modules: [q_proj, v_proj]\n"
        "output_dir: 'runs/x'\n"
    )
    cfg = TrainConfig.from_yaml(str(p))
    assert cfg.lora_r == 8
    assert cfg.learning_rate == 1e-4
    assert cfg.load_in_4bit is False
    assert cfg.target_modules == ("q_proj", "v_proj")
    assert cfg.output_dir == "runs/x"


def test_from_yaml_ignores_trailing_comments(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: m  # base\ndataset: d\n\n# note\nseed: 7\n")
    cfg = TrainConfig.from_yaml(str(p))
    assert cfg.model == "m"
    assert cfg.seed == 7


def test_shipped_example_configs_are_valid():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("smoke_config.yaml", "tiny_cpu_config.yaml"):
        cfg = TrainConfig.from_yaml(os.path.join(root, "examples", name))
        assert cfg.validate() == []


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace

import pytest

from tuneforge import TrainConfig
from tuneforge.models import load_base


@pytest.mark.parametrize('kwargs', [
    {'model': None}, {'dataset': ''}, {'epochs': 0}, {'max_steps': 0},
    {'warmup_ratio': 2}, {'learning_rate': float('nan')},
    {'target_modules': []}, {'load_in_4bit': 'true'}, {'grad_accum': 1.5},
])
def test_reject_bad_config(kwargs):
    values = {'model': 'm', 'dataset': 'd'}
    values.update(kwargs)
    assert TrainConfig(**values).validate()


@pytest.mark.parametrize('cuda,bf16,expected', [(True, False, 'fp16'),
                                             (True, True, 'bf16'),
                                             (False, False, 'fp32')])
def test_model_dtype_selection(monkeypatch, cuda, bf16, expected):
    seen = {}
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda,
        is_bf16_supported=lambda: bf16, current_device=lambda: 0),
        float16='fp16', bfloat16='bf16', float32='fp32')
    class Loader:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(config=SimpleNamespace(), pad_token='p')
    transformers = SimpleNamespace(AutoModelForCausalLM=Loader, AutoTokenizer=Loader,
                                    BitsAndBytesConfig=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    load_base(TrainConfig(model='m', dataset='d', load_in_4bit=cuda,
                           gradient_checkpointing=False))
    assert seen['dtype'] == expected
    if cuda:
        assert seen['quantization_config']['bnb_4bit_compute_dtype'] == expected


def test_quantized_cpu_fails_with_clear_message(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), float32='fp32'))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=object, AutoTokenizer=object, BitsAndBytesConfig=object))
    with pytest.raises(ValueError, match='CUDA'):
        load_base(TrainConfig(model='m', dataset='d'))


def test_yaml_keeps_hash_inside_quoted_path(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('model: "m#v1"\ndataset: d\n')
    assert TrainConfig.from_yaml(str(path)).model == 'm#v1'
