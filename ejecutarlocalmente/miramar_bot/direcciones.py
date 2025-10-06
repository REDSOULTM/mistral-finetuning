"""Para detectar y normalizar direcciones."""

from __future__ import annotations

import urllib.parse
from typing import Dict, Optional

import requests

try:  # pragma: no cover - optional dependency
    from postal.parser import parse_address
except ImportError:  # libpostal may be unavailable in tests
    parse_address = None  # type: ignore


def detect_and_extract_address(text: str) -> Optional[Dict[str, str]]:
    """Detecta y normaliza direcciones usando libpostal y Nominatim."""
    raw = (text or "").strip()
    if not raw or len(raw) < 5:
        return None

    # Intentar usar libpostal si está disponible
    if parse_address is not None:
        try:
            parts = dict(parse_address(raw))  # type: ignore[arg-type]
        except Exception:
            return None

        if not parts.get("road") and not parts.get("house_number") and not parts.get("city"):
            return None

        query = ""
        if parts.get("road"):
            query += parts["road"]
        if parts.get("house_number"):
            query += f" {parts['house_number']}"
        if parts.get("city"):
            query += f", {parts['city']}"
        if not query:
            query = raw

        url = (
            "https://nominatim.openstreetmap.org/search?format=json&addressdetails=1&limit=1&q="
            f"{urllib.parse.quote(query)}"
        )
        try:
            resp = requests.get(url, headers={"User-Agent": "MiramarBot/1.0"}, timeout=5)
            data = resp.json()
        except Exception:
            return None

        if not data:
            return None

        item = data[0]
        address = item.get("address", {})
        lat = item.get("lat")
        lon = item.get("lon")

        street = address.get("road") or parts.get("road") or ""
        number = address.get("house_number") or parts.get("house_number") or ""
        comuna = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or parts.get("city")
            or ""
        )
        coords = f"{lat},{lon}" if lat and lon else ""
        maps_link = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}" if lat and lon else ""

        return {
            "street": street,
            "number": number,
            "comuna": comuna,
            "coords": coords,
            "maps_link": maps_link,
            "raw": raw,
        }
    
    # Si libpostal no está disponible, retornar None para que utilidades.py use el fallback manual
    return None


__all__ = ["detect_and_extract_address"]
