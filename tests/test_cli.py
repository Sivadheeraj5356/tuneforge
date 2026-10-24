import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuneforge.cli import main


def test_plan_known_model_fits(capsys):
    rc = main(["plan", "Qwen/Qwen3-1.7B", "--gpu-gb", "16"])
    assert rc == 0
    assert "FITS" in capsys.readouterr().out


def test_plan_too_small_gpu_fails(capsys):
    rc = main(["plan", "Qwen/Qwen3-8B", "--gpu-gb", "4"])
    assert rc == 1
    assert "DOES NOT FIT" in capsys.readouterr().out


def test_plan_unknown_model_needs_arch(capsys):
    rc = main(["plan", "some/custom-model", "--gpu-gb", "16"])
    assert rc == 2


def test_eval_compare(tmp_path, capsys):
    import json
    b = tmp_path / "b.json"; t = tmp_path / "t.json"
    b.write_text(json.dumps({"n": 5, "exact_match": 0.4}))
    t.write_text(json.dumps({"n": 5, "exact_match": 0.8}))
    rc = main(["eval", str(b), str(t)])
    assert rc == 0
    assert "delta" in capsys.readouterr().out
