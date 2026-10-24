"""Training-memory estimator: predict peak VRAM before a run starts.

Pure-math, no torch import, so plans can be checked on any machine. The model
is intentionally approximate: it separates static weight memory, optimizer
state, adapter state, and activation memory, and reports which knob to turn
when a plan does not fit.
"""
from __future__ import annotations

from dataclasses import dataclass

BYTES_FP32 = 4
BYTES_FP16 = 2
BYTES_NF4 = 0.5        # 4-bit weights, packed
BYTES_INT8 = 1

# attention head dim is almost always 64 or 128; used only for activation scale
_HEAD_DIM = 128


@dataclass
class ModelInfo:
    n_params_b: float          # billions of parameters
    n_layers: int
    hidden: int
    n_trainable_pct: float = 1.0  # percent of params trainable (LoRA: ~1%)


@dataclass
class Plan:
    fits: bool
    total_gb: float
    weights_gb: float
    optimizer_gb: float
    gradients_gb: float
    activations_gb: float
    headroom_gb: float
    advice: list[str]


def estimate_weights_gb(info: ModelInfo, load_in_4bit: bool, lora_r: int = 0) -> float:
    base = info.n_params_b * 1e9 * (BYTES_NF4 if load_in_4bit else BYTES_FP16)
    if lora_r > 0:
        # LoRA adds ~2 * r * (in+out) per targeted linear; approximated from
        # layer count and hidden size for a standard SwiGLU transformer (7
        # projections per layer).
        lora_params = 2 * lora_r * (info.hidden + info.hidden) * 7 * info.n_layers
        base += lora_params * BYTES_FP16
    return base / 1e9


def estimate_optimizer_gb(info: ModelInfo, lora_r: int, paged: bool = True) -> float:
    """AdamW keeps 2 fp32 states per trainable param; LoRA restricts that to
    adapter params. `paged` (8-bit Adam) halves it."""
    if lora_r > 0:
        trainable = 2 * lora_r * (info.hidden + info.hidden) * 7 * info.n_layers
    else:
        trainable = info.n_params_b * 1e9 * (info.n_trainable_pct / 100.0)
    per_param = BYTES_INT8 * 2 if paged else BYTES_FP32 * 2
    return trainable * per_param / 1e9


def estimate_activation_gb(info: ModelInfo, batch: int, seq_len: int,
                           grad_ckpt: bool) -> float:
    """Per-layer activations scale with batch*seq*hidden. Gradient
    checkpointing stores only layer-boundary activations plus recompute
    scratch, roughly dividing by sqrt(n_layers)."""
    per_layer = batch * seq_len * info.hidden * BYTES_FP16 * 10  # ~10 tensors kept
    total = per_layer * info.n_layers
    if grad_ckpt:
        total = per_layer * (info.n_layers ** 0.5) * 4
    return total / 1e9


def plan(info: ModelInfo, gpu_gb: float, *, load_in_4bit: bool = True,
         lora_r: int = 16, batch: int = 4, seq_len: int = 1024,
         grad_ckpt: bool = True) -> Plan:
    w = estimate_weights_gb(info, load_in_4bit, lora_r)
    opt = estimate_optimizer_gb(info, lora_r)
    grads = opt / 2                      # grads tracked at same scale as states
    acts = estimate_activation_gb(info, batch, seq_len, grad_ckpt)
    total = w + opt + grads + acts
    headroom = gpu_gb - total
    advice: list[str] = []
    if headroom < 0:
        if not load_in_4bit:
            advice.append("enable load_in_4bit (NF4) for base weights")
        if not grad_ckpt:
            advice.append("enable gradient_checkpointing")
        if batch > 1:
            advice.append(f"lower per-device batch from {batch} (activations scale linearly)")
        if seq_len > 512:
            advice.append(f"shorten max_seq_len from {seq_len}")
        if lora_r > 8:
            advice.append(f"lower lora_r from {lora_r}")
        if not advice:
            advice.append("model too large for this GPU even fully shrunk")
    return Plan(fits=headroom >= 0, total_gb=round(total, 2),
                weights_gb=round(w, 2), optimizer_gb=round(opt, 2),
                gradients_gb=round(grads, 2), activations_gb=round(acts, 2),
                headroom_gb=round(headroom, 2), advice=advice)


# Small registry of known architectures (approximate, documented as such).
KNOWN_MODELS: dict[str, ModelInfo] = {
    "Qwen/Qwen3-0.6B": ModelInfo(0.6, 28, 1024),
    "Qwen/Qwen3-1.7B": ModelInfo(1.7, 28, 2048),
    "Qwen/Qwen3-4B": ModelInfo(4.0, 36, 2560),
    "Qwen/Qwen3-8B": ModelInfo(8.0, 36, 4096),
    "microsoft/phi-4-mini-instruct": ModelInfo(3.8, 32, 3072),
    "google/gemma-3-1b-it": ModelInfo(1.0, 26, 1152),
    "meta-llama/Llama-3.2-1B-Instruct": ModelInfo(1.24, 16, 2048),
    "meta-llama/Llama-3.2-3B-Instruct": ModelInfo(3.2, 28, 3072),
}
