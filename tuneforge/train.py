"""SFT training entry: config -> trainer. Thin, explicit orchestration over
transformers/peft/trl with tuneforge defaults; no hidden global state.

API compatibility note: written against the stack trl 1.5.x /
transformers 5.x / peft 0.21.0 (see pyproject pins). The underlying
training approach is proven on that stack; tuneforge's own end-to-end
run on it is tracked as the toolkit's smoke test. Key details:
- peft_config is handed to SFTTrainer (it wraps the model itself).
- loss_type="nll": TRL 1.6+ defaults SFT loss to a chunked kernel whose
  fp32-out bf16 GEMM needs sm80+; free-tier T4 is sm75. Plain nll is the
  same loss computed conventionally.
"""
from __future__ import annotations

from tuneforge.config import TrainConfig
from tuneforge.models import load_base


def build_dataset(cfg: TrainConfig, tok):
    from datasets import Dataset
    from tuneforge import data as tdata

    rows = tdata.load_rows(cfg.dataset)
    if cfg.text_field:
        texts = [str(r[cfg.text_field]) for r in rows]
    else:
        texts = [str(r[cfg.prompt_field]) + str(r[cfg.completion_field])
                 for r in rows]
    if cfg.packing:
        texts = tdata.pack_sequences(texts, cfg.max_seq_len)
    return Dataset.from_dict({"text": texts})


def train(cfg: TrainConfig) -> str:
    """Run SFT; returns the output dir containing the trained adapter."""
    errs = cfg.validate()
    if errs:
        raise ValueError("invalid config: " + "; ".join(errs))

    import torch
    model, tok = load_base(cfg)
    ds = build_dataset(cfg, tok)

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    peft_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), task_type="CAUSAL_LM",
    )
    steps_per_epoch = max(1, len(ds) // cfg.effective_batch_size)
    total_steps = int(steps_per_epoch * cfg.epochs)
    args = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        warmup_steps=int(cfg.warmup_ratio * total_steps),
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        max_length=cfg.max_seq_len,
        seed=cfg.seed,
        packing=cfg.packing,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataset_text_field="text",
        loss_type="nll",   # sm75 (free T4) safe; see module docstring
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir
