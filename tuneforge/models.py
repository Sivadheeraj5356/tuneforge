from __future__ import annotations

from tuneforge.config import TrainConfig


def load_base(cfg: TrainConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (
        torch.float16 if cuda else torch.float32)
    if cfg.load_in_4bit and not cuda:
        raise ValueError("4-bit training requires CUDA; set load_in_4bit: false on CPU")
    quant = None
    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        quantization_config=quant,
        dtype=dtype,
        device_map={"": torch.cuda.current_device()} if cuda else None,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def add_adapter(model, cfg: TrainConfig):
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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    try:
        tok = AutoTokenizer.from_pretrained(adapter or base)
    except (OSError, ValueError):
        if not adapter:
            raise
        tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def generate(base: str, prompt: str, adapter: str | None = None,
             max_new_tokens: int = 128) -> str:
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    import torch

    model, tok = load_for_inference(base, adapter)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)
