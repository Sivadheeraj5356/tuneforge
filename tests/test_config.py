import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuneforge.config import TrainConfig


def test_defaults_valid():
    assert TrainConfig(model="m", dataset="d").validate() == []


def test_json_round_trip():
    c = TrainConfig(model="m", dataset="d", lora_r=8, target_modules=("q_proj",))
    c2 = TrainConfig.from_json(c.to_json())
    assert c2.lora_r == 8 and c2.target_modules == ("q_proj",)


def test_unknown_key_rejected():
    import pytest
    with pytest.raises(ValueError):
        TrainConfig.from_json('{"model": "m", "dataset": "d", "bogus": 1}')


def test_validation_catches_bad_values():
    assert TrainConfig(model="m", dataset="d", lora_r=0).validate()
    assert TrainConfig(model="m", dataset="d", learning_rate=-1).validate()
    assert TrainConfig(model="m", dataset="d", max_seq_len=8).validate()


def test_effective_batch():
    c = TrainConfig(model="m", dataset="d", per_device_batch_size=4, grad_accum=8)
    assert c.effective_batch_size == 32


def test_yaml_subset(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: Qwen/Qwen3-1.7B\ndataset: gretelai/synthetic_text_to_sql\nlora_r: 8\nload_in_4bit: true\n"
                 "target_modules: [q_proj, v_proj]\nlearning_rate: 1e-4\n")
    c = TrainConfig.from_yaml(str(p))
    assert c.model == "Qwen/Qwen3-1.7B" and c.lora_r == 8
    assert c.load_in_4bit is True and c.target_modules == ("q_proj", "v_proj")
    assert abs(c.learning_rate - 1e-4) < 1e-9
