"""Configuración base y rutas empleadas por el bot Miramar."""

from __future__ import annotations

import os
from typing import Dict, List, Any

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
PROYECTO_DIR = os.path.abspath(os.path.join(ROOT, ".."))

# Ruta donde deben ubicarse el modelo base y el adaptador LoRA.
BASE_MODEL_DIR = os.path.join(PROYECTO_DIR, "modelos", "mistral-7b-bnb-4bit")  
LORA_ADAPTER_DIR = os.path.join(PROYECTO_DIR, "modelos", "mistral_finetuned_miramar_combined_steps20000", "lora_adapter")

DEFAULT_MODEL_VARIANT = os.environ.get("MIRAMAR_MODEL_VARIANT", "lora_4bit")

# Archivo donde se guardan las cotizaciones generadas.
COTIZACIONES_FILE = os.path.join(PACKAGE_DIR, "cotizaciones.json")

WELCOME_MESSAGE = "Hola, soy Transportes Miramar. Estoy aquí para ayudarte con tu traslado en Chile."

# Feature flag: pedir datos personales antes de cotizar.
ENABLE_PERSONAL_REGISTRATION = False

# Configuración ULTRA-OPTIMIZADA para MÁXIMA VELOCIDAD
LLM_SAMPLING_KWARGS: Dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 30,           # REDUCIDO para velocidad
    "do_sample": True,
    "repetition_penalty": 1.05,
    "max_new_tokens": 30,  # DRÁSTICAMENTE REDUCIDO para máxima velocidad
    "use_cache": True,     # Cache incremental habilitado
}

COMMENT_FIELD = "comentario adicional"

if ENABLE_PERSONAL_REGISTRATION:
    PERSONAL_FIELDS: List[str] = ["nombre", "rut", "correo"]
else:
    PERSONAL_FIELDS = []

TRIP_FIELDS = ["origen", "destino", "fecha", "hora", "regreso", "cantidad"]
PRIMARY_FIELDS = PERSONAL_FIELDS + TRIP_FIELDS
STATE_FIELDS = PRIMARY_FIELDS + [COMMENT_FIELD]

__all__ = [
    "PACKAGE_DIR",
    "ROOT",
    "BASE_MODEL_DIR",
    "LORA_ADAPTER_DIR",
    "DEFAULT_MODEL_VARIANT",
    "COTIZACIONES_FILE",
    "WELCOME_MESSAGE",
    "ENABLE_PERSONAL_REGISTRATION",
    "LLM_SAMPLING_KWARGS",
    "COMMENT_FIELD",
    "PERSONAL_FIELDS",
    "TRIP_FIELDS",
    "PRIMARY_FIELDS",
    "STATE_FIELDS",
]
