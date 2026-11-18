from __future__ import annotations

import argparse
import sys

from tuneforge import __version__


def _cmd_plan(args) -> int:
    from tuneforge.vram import KNOWN_MODELS, ModelInfo, plan
    if args.model in KNOWN_MODELS:
        info = KNOWN_MODELS[args.model]
    else:
        if args.params_b is None or args.layers is None or args.hidden is None:
            print("unknown model: pass --params-b --layers --hidden", file=sys.stderr)
            return 2
        info = ModelInfo(args.params_b, args.layers, args.hidden)
    p = plan(info, args.gpu_gb, load_in_4bit=not args.no_4bit,
             lora_r=args.lora_r, batch=args.batch, seq_len=args.seq_len)
    print(f"model: {args.model}")
    print(f"  weights:     {p.weights_gb:7.2f} GB ({'NF4 4-bit' if not args.no_4bit else 'fp16'})")
    print(f"  optimizer:   {p.optimizer_gb:7.2f} GB (paged AdamW, adapter only)")
    print(f"  gradients:   {p.gradients_gb:7.2f} GB")
    print(f"  activations: {p.activations_gb:7.2f} GB (batch {args.batch} x seq {args.seq_len})")
    print(f"  total:       {p.total_gb:7.2f} GB vs GPU {args.gpu_gb} GB -> "
          f"{'FITS' if p.fits else 'DOES NOT FIT'} (headroom {p.headroom_gb} GB)")
    for a in p.advice:
        print(f"  advice: {a}")
    return 0 if p.fits else 1


def _cmd_train(args) -> int:
    from tuneforge.config import TrainConfig
    from tuneforge.train import train
    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig(
        model=args.model, dataset=args.dataset)
    errs = cfg.validate()
    if errs:
        print("invalid config:", "; ".join(errs), file=sys.stderr)
        return 2
    out = train(cfg)
    print(f"adapter saved to {out}")
    return 0


def _cmd_eval(args) -> int:
    from tuneforge.evals import compare_runs
    import json
    with open(args.base_scores) as f:
        base = json.load(f)
    with open(args.tuned_scores) as f:
        tuned = json.load(f)
    print(json.dumps(compare_runs(base, tuned), indent=2))
    return 0


def _cmd_export(args) -> int:
    from tuneforge.export import merge_adapter, to_gguf
    merged = merge_adapter(args.base, args.adapter, args.merged_out)
    print(f"merged model at {merged}")
    if args.gguf_out:
        path = to_gguf(merged, args.gguf_out, quant=args.quant,
                       llama_cpp_dir=args.llama_cpp)
        print(f"GGUF at {path}")
    return 0


def _cmd_generate(args) -> int:
    from tuneforge.models import generate
    print(generate(args.base, args.prompt, adapter=args.adapter,
                   max_new_tokens=args.max_new_tokens))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tuneforge",
        description="Memory-efficient LLM fine-tuning toolkit")
    ap.add_argument("--version", action="version",
                    version=f"tuneforge {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="estimate VRAM for a training plan")
    p.add_argument("model")
    p.add_argument("--gpu-gb", type=float, required=True)
    p.add_argument("--params-b", type=float)
    p.add_argument("--layers", type=int)
    p.add_argument("--hidden", type=int)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--no-4bit", action="store_true")
    p.set_defaults(fn=_cmd_plan)

    t = sub.add_parser("train", help="run QLoRA SFT")
    t.add_argument("--config", help="YAML config path")
    t.add_argument("--model")
    t.add_argument("--dataset")
    t.set_defaults(fn=_cmd_train)

    e = sub.add_parser("eval", help="compare base vs tuned score files")
    e.add_argument("base_scores")
    e.add_argument("tuned_scores")
    e.set_defaults(fn=_cmd_eval)

    x = sub.add_parser("export", help="merge adapter and optionally convert to GGUF")
    x.add_argument("--base", required=True)
    x.add_argument("--adapter", required=True)
    x.add_argument("--merged-out", required=True)
    x.add_argument("--gguf-out")
    x.add_argument("--quant", default="q4_k_m")
    x.add_argument("--llama-cpp", default="llama.cpp")
    x.set_defaults(fn=_cmd_export)

    g = sub.add_parser("generate", help="run one prompt through base or tuned model")
    g.add_argument("--base", required=True)
    g.add_argument("--adapter")
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.set_defaults(fn=_cmd_generate)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
