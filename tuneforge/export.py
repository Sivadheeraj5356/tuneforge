from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def merge_adapter(base: str, adapter: str, out_dir: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(adapter)
    except (OSError, ValueError):
        tok = AutoTokenizer.from_pretrained(base)
    tok.save_pretrained(out_dir)
    return out_dir


def to_gguf(merged_dir: str, out_file: str, quant: str = "q4_k_m",
            llama_cpp_dir: str = "llama.cpp") -> str:
    convert = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert):
        raise FileNotFoundError(
            f"llama.cpp converter not found at {convert}; "
            "clone https://github.com/ggml-org/llama.cpp first")
    output = Path(out_file)
    source = Path(merged_dir).resolve()
    if output.resolve().is_relative_to(source):
        raise ValueError("GGUF output must be outside the merged model directory")
    f16 = str(output.with_name(output.stem + ".f16.gguf")) if quant.lower() != "f16" else str(output)
    quant_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
    if quant.lower() != "f16" and not os.path.isfile(quant_bin):
        raise FileNotFoundError(f"quantizer not found at {quant_bin}; build llama.cpp first")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, convert, merged_dir, "--outfile", f16,
                    "--outtype", "f16"], check=True)
    if quant.lower() != "f16":
        subprocess.run([quant_bin, f16, str(output), quant.upper()], check=True)
        if not output.is_file():
            raise RuntimeError("quantizer did not create the requested GGUF")
        os.remove(f16)
    if not output.is_file():
        raise RuntimeError("converter did not create the requested GGUF")
    return str(output)
