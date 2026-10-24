import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuneforge.vram import KNOWN_MODELS, ModelInfo, plan, estimate_weights_gb


def test_4bit_weights_quarter_of_fp16():
    info = ModelInfo(2.0, 28, 2048)
    w4 = estimate_weights_gb(info, True)
    w16 = estimate_weights_gb(info, False)
    assert w16 / w4 > 3.5  # ~4x (lora_r=0 so exact)


def test_small_model_fits_free_t4():
    p = plan(KNOWN_MODELS["Qwen/Qwen3-1.7B"], 16.0)
    assert p.fits and p.headroom_gb > 0


def test_oversized_model_advises():
    p = plan(KNOWN_MODELS["Qwen/Qwen3-8B"], 8.0, load_in_4bit=False)
    assert not p.fits and p.advice


def test_gradient_checkpointing_reduces_activations():
    info = KNOWN_MODELS["Qwen/Qwen3-8B"]
    on = plan(info, 80.0, grad_ckpt=True)
    off = plan(info, 80.0, grad_ckpt=False)
    assert on.activations_gb < off.activations_gb


def test_batch_scales_activations():
    info = KNOWN_MODELS["Qwen/Qwen3-1.7B"]
    b2 = plan(info, 80.0, batch=2)
    b8 = plan(info, 80.0, batch=8)
    assert b8.activations_gb > 3 * b2.activations_gb
