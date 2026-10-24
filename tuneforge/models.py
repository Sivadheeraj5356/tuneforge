"""Model + adapter loading. Everything heavy (torch/transformers/peft) is
imported inside functions so the package stays importable without a GPU stack.
"""
from __future__ import annotations

from tuneforge.config import TrainConfig


def load_base(cfg: TrainConfig):
    """Load the base model, NF4-quantized when cfg.load_in_4bit, with
    gradient checkpointing for training."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = None
    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        quantization_config=quant,
        dtype=None if quant else torch.bfloat16,
        device_map="auto",
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def add_adapter(model, cfg: TrainConfig):
    """Attach a fresh LoRA adapter per the config."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable/1e6:.1f}M / {total/1e9:.2f}B "
          f"({100*trainable/total:.2f}%)")
    return model


def load_for_inference(base: str, adapter: str | None = None):
    """Load base (+ optional trained adapter) for generation/eval."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    tok = AutoTokenizer.from_pretrained(adapter or base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok
