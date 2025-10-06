"""
Sugerencias para cuantización de la KV cache en motores que lo soportan (p.ej. vLLM).
No fuerza nada: expone utilidades para aplicar en tiempo de construcción de engine.
"""

from __future__ import annotations
import os
import sys
from typing import Any, Dict

_ENABLED = False

def _log(msg: str) -> None:
    print(f"[quantizacion_kv_cache] {msg}", file=sys.stderr)

def enable() -> None:
    """Marca intención de usar KV cache cuantizada cuando sea posible."""
    global _ENABLED
    if _ENABLED:
        return
    # Variables de entorno comunes en backends modernos
    # (si el backend no las usa, simplemente serán ignoradas)
    os.environ.setdefault("KV_CACHE_QUANT", "fp8")  # hint genérico
    os.environ.setdefault("KV_CACHE_DTYPE", "fp8")
    _ENABLED = True
    _log("preferencia KV cache cuantizada = fp8 (si el backend lo soporta).")

def apply_vllm_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Si estás creando un vLLM LLM(...), puedes pasar el dict resultante:
      kwargs = apply_vllm_kwargs(kwargs)
      llm = LLM(**kwargs)
    No falla si vllm no está instalado.
    """
    try:
        # vLLM >= 0.5 usa CacheConfig via kwargs; versiones previas ignorarán claves desconocidas
        kwargs.setdefault("kv_cache_dtype", "fp8")  # preferencia
        kwargs.setdefault("tensor_parallel_size", kwargs.get("tensor_parallel_size", 1))
        _log("vLLM kwargs sugeridos: kv_cache_dtype=fp8.")
    except Exception as e:
        _log(f"no-op al sugerir kwargs vLLM: {type(e).__name__}: {e}")
    return kwargs
