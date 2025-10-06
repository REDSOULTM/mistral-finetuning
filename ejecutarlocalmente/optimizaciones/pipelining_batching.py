"""
Pequeñas palancas de batching/pipelining: num_threads, cudnn.benchmark y pad_side del tokenizer.
No interfiere con tu loop si no lo llamas explícitamente.
"""

from __future__ import annotations
import os
import sys
from typing import Any

_ENABLED = False

def _log(msg: str) -> None:
    print(f"[pipelining_batching] {msg}", file=sys.stderr)

def enable(num_threads: int | None = None) -> None:
    """
    Ajusta algunos flags globales (seguros) para throughput.
    """
    global _ENABLED
    if _ENABLED:
        return
    try:
        import torch  # type: ignore
        # Cuantos hilos en CPU (tokenización o pre/post)
        if num_threads is not None and num_threads > 0:
            try:
                torch.set_num_threads(num_threads)  # type: ignore[attr-defined]
                _log(f"torch.set_num_threads({num_threads})")
            except Exception:
                pass
        # Heurística para mejor selección de kernels en CNN (no afecta atención)
        try:
            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass
    _ENABLED = True
    _log("flags básicos de rendimiento activados (threads/cudnn.benchmark).")

def tune_tokenizer(tokenizer: Any) -> None:
    """
    Para tokenizers de HuggingFace: asegura padding lado 'left' en chat (mejor para batch).
    """
    try:
        if hasattr(tokenizer, "padding_side"):
            if tokenizer.padding_side != "left":
                tokenizer.padding_side = "left"
                _log("tokenizer.padding_side = 'left'")
    except Exception as e:
        _log(f"no-op tune_tokenizer: {type(e).__name__}: {e}")
