"""tuneforge: memory-efficient fine-tuning toolkit for open-weight LLMs.

Public surface is import-safe without torch; heavy deps load lazily inside
functions so planning (VRAM estimate, config checks, data prep) works on any
machine and only training/export need the full stack.
"""
from tuneforge.config import TrainConfig
from tuneforge.vram import ModelInfo, Plan, plan, KNOWN_MODELS

__version__ = "0.1.0"
__all__ = ["TrainConfig", "ModelInfo", "Plan", "plan", "KNOWN_MODELS", "__version__"]
