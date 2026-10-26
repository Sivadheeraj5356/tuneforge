# Validation log

Exact commands and observed results for tuneforge's verified runs. Quality
claims live in README.md; this file is the mechanics record.

## 2026-10-25 - real QLoRA smoke (free Colab T4, Qwen/Qwen3-1.7B)

    pip uninstall -y torchao   # Colab torchao 0.10 workaround, see README
    python examples/prep_smoke.py --rows 200 --out smoke_rows.jsonl
    tuneforge train --config examples/smoke_config.yaml
    tuneforge export --base Qwen/Qwen3-1.7B --adapter runs/smoke \
      --merged-out runs/smoke_merged

Observed: 20 QLoRA steps in 82.4s, batch 2 x seq 512, train_loss 1.082;
adapter 34.9MB saved; fp16 merged safetensors 3.441GB exported.
Full logs/configs: private Drive backup (smoke_evidence.tar.gz).

## 2026-10-25 - GGUF chain (synthetic tiny Qwen3-arch model, CPU)

    tuneforge export --base <tiny> --adapter <random-lora> \
      --merged-out <m> --gguf-out <m>.gguf --quant q4_k_m

Observed: merge -> F16 GGUF -> Q4_K_M quantize -> llama.cpp (rev 74bca39)
loads and generates. Mechanics only - random weights, no quality claim.

## 2026-10-26 - CPU train + merge (tiny random model, no GPU/token)

Model: Qwen3Config(vocab=~152k, hidden 128, 2 layers, heads 4/2, ctx 512),
random init, 39.4M params, saved locally; tokenizer Qwen/Qwen3-0.6B
(public, unauthenticated download). Data: 32 local rows, smoke format.

    tuneforge train --config examples/tiny_cpu_config.yaml
    # loss 11.93 -> 11.90 -> 11.86 -> 11.82 -> 11.80 -> 11.77 (6 steps,
    # batch 2 x seq 128, LoRA r=4); adapter + checkpoint + tokenizer saved
    tuneforge export --base /tmp/tiny-qwen3 --adapter /tmp/runs/tiny \
      --merged-out /tmp/runs/tiny_merged
    # merged dir valid; safetensors diff vs base shows LoRA deltas folded
    # into target-module weights (attn q/k/v/o + mlp gate/up/down changed,
    # max abs delta ~0.002; all other tensors identical)

Also verified 2026-10-26: fresh-venv `pip install -e .[dev]` resolves all
declared deps (torch 2.14.1, transformers 5.20.2, trl 1.5.1, peft 0.21.0,
bitsandbytes 0.50.2, accelerate 1.15.0, datasets 5.1.0) and 24/24 tests
pass; `tuneforge plan` output matches the README quickstart byte-for-byte;
`tuneforge eval base.json tuned.json` runs as documented;
examples/text2sql_eval.py runs end-to-end on CPU with the tiny model
(scorers sanity-checked: gold-as-pred passes sql_exact_match and
exec_match).

Not covered by any run: merged-model inference quality on a real base
(weights/adapter gated on a valid HF token + GPU window).
