"""
Activa defaults seguros para inferencia incremental (use_cache) y contextos de performance.
Incluye un contexto de inferencia y un pequeño helper para Transformers.
"""

from __future__ import annotations
import sys
from contextlib import contextmanager

_ENABLED = False

def _log(msg: str) -> None:
    print(f"[cache_incremental] {msg}", file=sys.stderr)

@contextmanager
def inference_mode():
    """
    Contexto robusto que intenta usar torch.inference_mode() si existe;
    cae a no_grad() si no.
    """
    try:
        import torch  # type: ignore
        if hasattr(torch, "inference_mode"):
            with torch.inference_mode():  # type: ignore[attr-defined]
                yield
                return
        else:
            with torch.no_grad():
                yield
                return
    except Exception:
        # torch no está; no hacemos nada
        yield

def enable() -> None:
    """No parcha agresivo nada; solo registra disponibilidad."""
    global _ENABLED
    if _ENABLED:
        return
    _ENABLED = True
    _log("contexto inference_mode listo (use_cache sugerido en helpers).")

def transformers_generate_defaults(model) -> None:
    """
    Establece defaults apropiados si 'model' es un Transformers-like.
    No falla si no lo es.
    """
    try:
        # Ajustes casuales seguros
        if hasattr(model, "config"):
            cfg = model.config
            if getattr(cfg, "use_cache", None) is not True:
                cfg.use_cache = True
        _log("Transformers: use_cache=True sugerido en config.")
    except Exception as e:
        _log(f"no-op en transformers_generate_defaults: {type(e).__name__}: {e}")
