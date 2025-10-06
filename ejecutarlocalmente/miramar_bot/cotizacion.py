"""Guarda las cotizaciones."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .configuracion import COTIZACIONES_FILE


def guardar_cotizacion(state: Dict[str, Any]) -> None:
    """Anexa la cotización recibida al archivo JSON asegurando escritura atómica."""
    data: List[Dict[str, Any]] = []
    if os.path.exists(COTIZACIONES_FILE):
        try:
            with open(COTIZACIONES_FILE, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            data = []

    data.append(state)
    tmp_path = COTIZACIONES_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, COTIZACIONES_FILE)
    print(f"\nTransporte Miramar: 📁 Cotización guardada en {COTIZACIONES_FILE}")


__all__ = ["guardar_cotizacion"]
