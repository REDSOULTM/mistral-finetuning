"""
Reducción de ruido en prompts y cacheo de pretokenización de plantillas estáticas.
"""

from __future__ import annotations
import re
import sys
from functools import lru_cache
from typing import Any, Dict, Tuple

_ENABLED = False

def _log(msg: str) -> None:
    print(f"[prompt_diet] {msg}", file=sys.stderr)

_WS_RE = re.compile(r"[ \t\u00A0]+")

def diet_prompt(text: str, max_consecutive_newlines: int = 2) -> str:
    """
    - Colapsa espacios consecutivos.
    - Recorta líneas.
    - Limita saltos de línea consecutivos.
    No toca contenido dentro de bloques de código triple.
    """
    parts = text.split("```")
    for i in range(0, len(parts), 2):  # solo segmentos fuera de code blocks
        chunk = parts[i]
        # normalizar espacios
        chunk = "\n".join(_WS_RE.sub(" ", ln).strip() for ln in chunk.splitlines())
        # limitar \n consecutivos
        lines = [ln for ln in chunk.splitlines()]
        out = []
        nl = 0
        for ln in lines:
            if ln == "":
                nl += 1
                if nl <= max_consecutive_newlines:
                    out.append("")
            else:
                nl = 0
                out.append(ln)
        parts[i] = "\n".join(out).strip()
    return "```".join(parts).strip()

@lru_cache(maxsize=64)
def pretokenize_template(system: str, user_prefix: str, assistant_prefix: str, sep: str = "\n") -> Tuple[str, str, str, str]:
    """
    Cachea piezas estáticas frecuentes.
    """
    sys_clean = diet_prompt(system)
    up_clean = diet_prompt(user_prefix)
    as_clean = diet_prompt(assistant_prefix)
    tpl = sep.join([sys_clean, up_clean, as_clean]).strip()
    return sys_clean, up_clean, as_clean, tpl

def enable() -> None:
    global _ENABLED
    if _ENABLED:
        return
    _ENABLED = True
    _log("diet_prompt y pretokenize_template habilitados.")
