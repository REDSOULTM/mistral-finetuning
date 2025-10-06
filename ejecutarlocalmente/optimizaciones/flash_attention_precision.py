"""
Configura precisión de matmul y habilita Flash-Attention si está disponible.
Seguro y opcional: se degrada a no-op si torch/flash-attn no existen.
"""

from __future__ import annotations
import os
import sys

_ENABLED = False

def _log(msg: str) -> None:
    print(f"[flash_attention_precision] {msg}", file=sys.stderr)

def enable() -> None:
    global _ENABLED
    if _ENABLED:
        return
    try:
        import torch  # type: ignore
        # Permitir TF32 (mejor throughput en A100/RTX 30/40, precisión suficiente para inferencia)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass
        # Matmul precision (PyTorch 2.x)
        try:
            torch.set_float32_matmul_precision("high")  # "high" ~ TF32/FP16 preferidos
        except Exception:
            pass

        # Intentar habilitar flash-attn si existe
        try:
            import flash_attn  # type: ignore  # noqa: F401
            os.environ.setdefault("USE_FLASH_ATTENTION", "1")
            _log("Flash-Attention detectado y marcado para uso.")
        except Exception:
            os.environ.setdefault("USE_FLASH_ATTENTION", "0")
            _log("Flash-Attention no detectado; continúo sin él.")

        _ENABLED = True
        _log("TF32 y precisión de matmul configuradas.")
    except Exception as e:
        _log(f"no-op (torch no disponible o error menor: {type(e).__name__}: {e})")
