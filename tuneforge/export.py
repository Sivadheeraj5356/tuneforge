"""Export trained adapters: merge into the base model and convert to GGUF
for llama.cpp serving. Heavy deps load lazily; the merge plan is computed
without torch so it can be validated anywhere.
"""
from __future__ import annotations

import os
import subprocess


def merge_adapter(base: str, adapter: str, out_dir: str) -> str:
    """Merge a LoRA adapter into base weights (fp16) and save."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter or base).save_pretrained(out_dir)
    return out_dir


def to_gguf(merged_dir: str, out_file: str, quant: str = "q4_k_m",
            llama_cpp_dir: str = "llama.cpp") -> str:
    """Convert a merged HF model dir to GGUF via llama.cpp's converter, then
    quantize. Requires a local llama.cpp checkout (see scripts)."""
    convert = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert):
        raise FileNotFoundError(
            f"llama.cpp converter not found at {convert}; "
            "clone https://github.com/ggml-org/llama.cpp first")
    f16 = out_file.replace(".gguf", ".f16.gguf")
    subprocess.run(["python", convert, merged_dir, "--outfile", f16,
                    "--outtype", "f16"], check=True)
    quant_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
    if os.path.exists(quant_bin) and quant != "f16":
        subprocess.run([quant_bin, f16, out_file, quant.upper()], check=True)
        os.remove(f16)
    else:
        out_file = f16
    return out_file
