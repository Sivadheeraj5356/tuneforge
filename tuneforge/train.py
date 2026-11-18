from __future__ import annotations

import math

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
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("training data must contain nonempty text rows")
    return Dataset.from_dict({"text": texts})


def train(cfg: TrainConfig) -> str:
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
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    total_steps = cfg.max_steps if cfg.max_steps > 0 else math.ceil(
        math.ceil(len(ds) / cfg.effective_batch_size) * cfg.epochs)
    args = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        warmup_steps=math.ceil(cfg.warmup_ratio * total_steps),
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        max_length=cfg.max_seq_len,
        seed=cfg.seed,
        packing=cfg.packing,
        bf16=bf16,
        fp16=cuda and not bf16,
        optim="paged_adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataset_text_field="text",
        loss_type="nll",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir
