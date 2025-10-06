"""
Intenta acelerar con torch.compile en modelos HF; para vLLM, solo hints no disruptivos.
No rompe si torch/vllm no están instalados.
"""

from __future__ import annotations
import os
import sys
from typing import Any, Optional

_ENABLED = False

def _log(msg: str) -> None:
    print(f"[torch_compile_vllm] {msg}", file=sys.stderr)

def enable(mode: str = "max-autotune") -> None:
    """
    Prepara entorno para compilación: inductor/ cudagraphs según corresponda.
    """
    global _ENABLED
    if _ENABLED:
        return
    # Hints genéricos; si el backend no los usa, se ignoran
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCH_COMPILE_DYNAMIC_SHAPES", "1")
    os.environ.setdefault("TORCH_LOGS", "error")
    # Para vLLM algunos builds leen estos flags:
    os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTENTION", "1")
    _ENABLED = True
    _log(f"hints de compilación establecidos (mode={mode}).")

def try_compile_model(model: Any, fullgraph: bool = False) -> Any:
    """
    Envuelve un modelo HF con torch.compile si está disponible.
    Devuelve el mismo modelo si no.
    """
    try:
        import torch  # type: ignore
        if hasattr(torch, "compile"):
            compiled = torch.compile(model, mode="max-autotune", fullgraph=fullgraph)  # type: ignore[attr-defined]
            _log("torch.compile aplicado al modelo.")
            return compiled
        else:
            _log("torch.compile no disponible; no-op.")
            return model
    except Exception as e:
        _log(f"no-op torch.compile: {type(e).__name__}: {e}")
        return model

def apply_vllm_hints(kwargs: dict) -> dict:
    """
    Si usas vLLM, puedes pasar los kwargs por aquí para añadir sugerencias suaves.
    """
    try:
        kwargs.setdefault("enforce_eager", False)  # preferir kernels compilados cuando aplique
        _log("hints para vLLM aplicados (enforce_eager=False).")
    except Exception as e:
        _log(f"no-op hints vLLM: {type(e).__name__}: {e}")
    return kwargs
