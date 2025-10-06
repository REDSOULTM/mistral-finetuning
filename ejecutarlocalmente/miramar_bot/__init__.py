"""Punto de entrada de alto nivel para el paquete miramar_bot."""

from .nucleo import (
    MiramarSellerBot,
    extract_structured_pairs,
    guardar_cotizacion,
    load_model_and_tokenizer,
    main,
    sanitize_print,
    STATE_FIELDS,
)

__all__ = [
    "MiramarSellerBot",
    "extract_structured_pairs",
    "guardar_cotizacion",
    "load_model_and_tokenizer",
    "sanitize_print",
    "STATE_FIELDS",
    "main",
]
