# tuneforge

Memory-efficient fine-tuning toolkit for open-weight LLMs. Plan the VRAM,
train a QLoRA adapter, evaluate base vs tuned, export to GGUF - one small,
readable codebase instead of scattered notebooks.

Inspired by the same problem space as [Unsloth](https://github.com/unslothai/unsloth)
(Apache-2.0): making LLM fine-tuning fit on free and cheap GPUs. tuneforge is
an independent implementation - no Unsloth code, branding, or performance
claims. It orchestrates the standard OSS stack (transformers, peft, trl,
bitsandbytes) and adds its own planning, data, eval, and export tooling.

## Why

- **Know before you train.** `tuneforge plan` estimates peak VRAM (weights,
  optimizer, gradients, activations) and tells you which knob to turn when a
  plan doesn't fit - before you burn an hour of free-GPU quota.
- **Sane QLoRA defaults.** NF4 4-bit base weights, paged optimizers, gradient
  checkpointing, LoRA on all projections. Small models train on a free Colab
  T4.
- **Evaluation is a first-class step.** Held-out splitting that blocks
  near-duplicate leakage, exact-match scorers, and a base-vs-tuned comparison
  table so "did fine-tuning help?" has a number, not a vibe.
- **Export to real serving.** Merge the adapter and convert to GGUF for
  llama.cpp.

## Install

```bash
pip install -e .          # training stack
pip install -e .[dev]     # + tests
```

Planning, config, data utilities, and tests run without torch or a GPU.
Training/export need a CUDA GPU.

## Quickstart

Check the plan fits your GPU:

```bash
tuneforge plan Qwen/Qwen3-1.7B --gpu-gb 16
#   weights: 0.90 GB (NF4)  optimizer: 0.05  gradients: 0.03  activations: 3.55
#   total: 4.53 GB vs GPU 16.0 GB -> FITS (headroom 11.47 GB)
```

Train (config YAML):

```yaml
model: Qwen/Qwen3-1.7B
dataset: gretelai/synthetic_text_to_sql
output_dir: runs/qwen3-sql
lora_r: 16
lora_alpha: 32
load_in_4bit: true
per_device_batch_size: 4
grad_accum: 4
max_seq_len: 1024
learning_rate: 2e-4
```

```bash
tuneforge train --config train.yaml
```

Or from Python:

```python
from tuneforge import TrainConfig
from tuneforge.train import train

cfg = TrainConfig(model="Qwen/Qwen3-1.7B",
                  dataset="gretelai/synthetic_text_to_sql")
train(cfg)   # adapter lands in cfg.output_dir
```

Evaluate base vs tuned (see `examples/text2sql_eval.py` for a complete
held-out harness with exact-match and sandboxed SQL execution accuracy):

```bash
tuneforge eval base_scores.json tuned_scores.json
```

Export for llama.cpp:

```bash
tuneforge export --base Qwen/Qwen3-1.7B --adapter runs/qwen3-sql \
  --merged-out runs/qwen3-sql-merged --gguf-out qwen3-sql.q4_k_m.gguf
```

## What it does and doesn't claim

- It uses battle-tested libraries for the math (peft/bitsandbytes/trl); its
  own code is the planning, data, eval, and export layers.
- Memory numbers from `plan` are estimates with the model documented in
  `tuneforge/vram.py`. Measure your own run before quoting them.
- No speedup claims. If a benchmark ever ships here, it will come with the
  script and hardware it was measured on.

## Reference deployment

Developed alongside a real project: QLoRA fine-tune of Qwen3-1.7B on
text-to-SQL (6000 train rows, 300 schema-held-out rows, zero schema overlap),
trained on a free Colab T4. See `examples/`.

## License

Apache-2.0. Component licenses: model weights and datasets carry their own
licenses - check each model/dataset card before training or redistributing.
