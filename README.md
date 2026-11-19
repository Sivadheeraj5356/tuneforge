# TuneForge

A fine-tuning toolkit for open-weight language models: estimate memory, prepare data, train a LoRA/QLoRA adapter, compare evaluation scores, generate text and export a merged model or GGUF.

TuneForge is an independent project in the same problem space as [Unsloth](https://github.com/unslothai/unsloth). It uses Transformers, PEFT, TRL and bitsandbytes. It does not copy Unsloth's code or implement its custom GPU kernels. It makes no speedup or feature-parity claim.

## Semester evaluation: start here

The Python toolkit has a hosted browser workbench at https://tuneforge-web.vercel.app/. No API key or paid service is needed for the CPU demonstration.

The UI provides a working VRAM planner, local JSON/JSONL data validation, YAML configuration download, score comparison and export commands. It does not run training on a server. GPU training still runs in your own Python environment or Colab when a free GPU is available. To serve the UI locally, run `python -m http.server 8080 --directory web` and open http://localhost:8080.

```bash
git clone https://github.com/Sivadheeraj5356/tuneforge.git
cd tuneforge
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/cpu_demo.py
```

Model training in the CPU demo is entirely local.

The demo creates a tiny randomly initialized Qwen3-architecture model and a local tokenizer, trains a LoRA adapter for two steps on eight synthetic SQL examples, reloads the adapter for generation, and saves a merged model. Outputs are in `runs/cpu-demo/`. It ends with `CPU demo complete`. This demonstrates the real pipeline, not useful SQL ability. Random-model generation is not a quality benchmark.

Suggested demonstration order:

1. Run `tuneforge --help` and explain the workflow.
2. Run `tuneforge plan Qwen/Qwen3-1.7B --gpu-gb 16` to inspect the memory estimate.
3. Run the tests and CPU demo above. Show the saved adapter and merged-model directories.
4. Explain the separate real-model results below and their limitations.

## Results and verification

| Check | Result | Meaning |
| --- | --- | --- |
| November 19, 2026 regression suite | 50 tests passed on Python 3.10 | Config, CLI, memory planning, data utilities, scoring, GPU dtype selection and export failure paths |
| November 19, 2026 CPU demonstration | Train, adapter save/reload, generation and merge passed | Tiny random model, mechanics only; no GPU and no quality claim |
| November 6, 2026 recorded real-model run | Base execution accuracy 0.30, tuned 0.53 on 100 held-out examples | Qwen3-1.7B text-to-SQL, free Colab T4; not rerun by the November 19 CPU checks |
| Same real-model run | Exact match 0.08 to 0.33; Q4_K_M GGUF 1,107,408,736 bytes | Historical run evidence, not a general performance guarantee |

The recorded real run used 1,500 training rows across 1,459 schemas and 400 schema-held-out rows across 384 schemas, with no schema overlap. Evaluation sampled 100 held-out rows. Training ran for 300 steps with final train loss 0.5493. The exported GGUF was loaded in llama.cpp and generated SQL on a held-out prompt. The execution scorer uses generated database contexts, not a production SQL benchmark.

The private artifact repository holds the adapter, GGUF, evaluation scores and run log:
https://huggingface.co/sivadheeraj5356/tuneforge-qwen3-1.7b-text2sql

Historical commands and results are in [examples/VALIDATION.md](examples/VALIDATION.md). Access to the private artifacts is separate from access to this code repository. Do not claim that the CPU demo reproduces the real-model accuracy numbers.

## Training a real model

The tested library API is Transformers 5.20.2, PEFT 0.21.0 and TRL 1.5.1. November 19 CPU verification used PyTorch 2.14.1+cpu. The earlier T4 run used PyTorch 2.11.0+cu130 and bitsandbytes 0.50.2. Hardware and library differences can affect results.

Prepare the example data and train on a CUDA GPU:

```bash
python examples/prep_smoke.py --rows 200 --out smoke_rows.jsonl
tuneforge plan Qwen/Qwen3-1.7B --gpu-gb 16
tuneforge train --config examples/smoke_config.yaml
```

Data preparation downloads a public Hugging Face dataset. The default 4-bit configuration needs CUDA. CPU training works with `load_in_4bit: false`: precision falls back to FP32 and the train/save/generate/merge pipeline runs without a GPU, as the CPU demo shows. It is practical for small models and pipeline checks; a free Colab GPU remains the realistic route for billion-parameter models. For comparison, Unsloth's documented CPU support covers chat and data preparation only - its training path requires an NVIDIA, AMD or Intel GPU, or a Mac.

Training settings are in `TrainConfig`: model, dataset, output directory, adapter rank/alpha/dropout, target modules, batch size, gradient accumulation, sequence length, learning rate, epochs or maximum steps, and packing. JSON and YAML configs are supported. Local datasets use `local:path.jsonl` or `local:path.json`, with `prompt` and `completion` fields by default. For raw-text SFT, set `text_field` to the dataset's text field.

Packing is delegated to TRL once, using token lengths. Precision is selected by GPU support: BF16 where supported, FP16 on other CUDA GPUs, FP32 on CPU. Quantized training uses paged 8-bit AdamW. The planner reports estimates, not measured peak VRAM; it does not guarantee that a run will fit.

Some Colab installations contain incompatible `torchao` versions. TuneForge does not use torchao quantization. The earlier verified run required uninstalling torchao or updating it to a compatible version; inspect the error and your environment before changing installed packages.

## Preflight and run cards

`preflight` checks a training plan before any GPU time is spent: config validity, dataset existence and field names for local files, whether the training dependencies import, and the estimated memory fit against a GPU you name or the one it detects. It exits nonzero when something would fail at train time.

```bash
tuneforge preflight --config examples/smoke_config.yaml --gpu-gb 16
```

Hub datasets are downloaded at train time, so preflight can only note them; it cannot check their fields ahead of time.

`report` writes a `RUN_CARD.md` for a finished run from the files the run leaves behind: the training config, the PEFT adapter config, and the trainer state (from the run directory or a checkpoint inside it), plus the Python, torch and GPU it can see on the machine. Anything it cannot find is listed under "Not recorded" rather than guessed.

```bash
tuneforge report runs/smoke
```

## Generate and evaluate

```bash
tuneforge generate --base Qwen/Qwen3-1.7B --adapter runs/smoke \
  --prompt 'Write SQL to select all rows from users:' --max-new-tokens 64

tuneforge eval base_scores.json tuned_scores.json
```

`eval` compares numeric metrics in two score JSON files. It does not run model generation itself. [examples/text2sql_eval.py](examples/text2sql_eval.py) is the task-specific generation/scoring example. Use the same held-out prompts for base and tuned models. Group the split by schema or another task-specific key to avoid overlap; the generic split helper does not detect arbitrary semantic duplicates.

## Export

Merge the adapter into the base model:

```bash
tuneforge export --base Qwen/Qwen3-1.7B --adapter runs/smoke \
  --merged-out runs/smoke-merged
```

For GGUF, separately clone and build [llama.cpp](https://github.com/ggml-org/llama.cpp) using its instructions, and install its converter requirements. The converter and `build/bin/llama-quantize` binary must exist. Installing `llama-cpp-python` alone does not provide that checkout.

```bash
tuneforge export --base Qwen/Qwen3-1.7B --adapter runs/smoke \
  --merged-out runs/smoke-merged --gguf-out runs/smoke.q4_k_m.gguf \
  --llama-cpp llama.cpp --quant q4_k_m
```

A missing quantizer now fails explicitly instead of silently returning an unquantized F16 file. GGUF output must be outside the merged-model directory. Use `--quant f16` to request an unquantized GGUF. November 19 export regression tests mock converter processes; real-weight GGUF validation is the separately recorded November 6 run.

## Scope

Implemented: memory planning, validated config loading, local/Hugging Face datasets, LoRA/QLoRA training, score comparison, base/adapter generation, model merge and llama.cpp conversion orchestration.

Not implemented: custom CUDA kernels, Unsloth-equivalent acceleration, distributed training orchestration, hosted GPU training, hosted inference, or one-click GPU provisioning. This is a small independent fine-tuning project, not a replacement for every Unsloth feature.

## License

Apache-2.0. Model weights, datasets and dependencies have their own licenses. Check their terms before redistributing trained artifacts.
