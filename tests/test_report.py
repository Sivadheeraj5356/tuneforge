import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from tuneforge.cli import main
from tuneforge.report import build_run_card


def _run_dir(tmp_path, with_state=True, with_adapter=True, with_config=True):
    run = tmp_path / "run1"
    run.mkdir()
    if with_config:
        (run / "config.yaml").write_text("model: Qwen/Qwen3-1.7B\nepochs: 3\n")
    if with_adapter:
        (run / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "Qwen/Qwen3-1.7B",
            "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        }))
    if with_state:
        (run / "trainer_state.json").write_text(json.dumps({
            "global_step": 60,
            "log_history": [
                {"loss": 1.2, "step": 20},
                {"loss": 0.7, "step": 40},
                {"eval_loss": 0.9, "step": 60},
            ],
        }))
    return str(run)


def test_run_card_full(tmp_path):
    card = build_run_card(_run_dir(tmp_path))
    assert "Qwen/Qwen3-1.7B" in card
    assert "r=16" in card
    assert "final train loss 0.7" in card
    assert "final eval loss 0.9" in card
    assert "epochs: 3" in card
    assert "Not recorded" not in card


def test_run_card_lists_missing(tmp_path):
    card = build_run_card(_run_dir(tmp_path, with_state=False, with_adapter=False))
    assert "trainer_state.json (metrics)" in card
    assert "adapter_config.json" in card


def test_run_card_finds_state_in_checkpoints(tmp_path):
    run = _run_dir(tmp_path, with_state=False)
    ckpt = os.path.join(run, "checkpoint-20")
    os.makedirs(ckpt)
    with open(os.path.join(ckpt, "trainer_state.json"), "w") as f:
        json.dump({"global_step": 20, "log_history": [{"loss": 1.1, "step": 20}]}, f)
    card = build_run_card(run)
    assert "final train loss 1.1" in card


def test_cli_report_writes_file(tmp_path, capsys):
    run = _run_dir(tmp_path)
    rc = main(["report", run])
    assert rc == 0
    path = os.path.join(run, "RUN_CARD.md")
    assert os.path.exists(path)
    assert "Run card" in open(path).read()


def test_cli_report_custom_out(tmp_path):
    run = _run_dir(tmp_path)
    out = str(tmp_path / "card.md")
    rc = main(["report", run, "--out", out])
    assert rc == 0
    assert os.path.exists(out)
