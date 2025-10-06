"""Carga el modelo que utilizará el chatbot Miramar."""

from __future__ import annotations

import os

import torch
from unsloth import FastLanguageModel

from .configuracion import (
    BASE_MODEL_DIR,
    DEFAULT_MODEL_VARIANT,
    LORA_ADAPTER_DIR,
)


def _load_lora_variant(device: str):
    use_4bit = device == "cuda"
    if device == "cuda":
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass
    print("Cargando modelo base 4b + LoRA...")
    
    # Configuración básica para Unsloth
    load_kwargs = {
        "model_name": BASE_MODEL_DIR,
        "adapter_name": os.path.join(LORA_ADAPTER_DIR, "lora_adapter"),
        "max_seq_length": 2048,
        "dtype": torch.float16 if use_4bit else torch.float32,
        "load_in_4bit": use_4bit,
        "float8_kv_cache": use_4bit,
        "fast_inference": use_4bit,
    }
    
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    FastLanguageModel.for_inference(model)
    
    # Aplicar compilación si está disponible en CUDA
    if device == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")
            print("Modelo compilado con torch.compile.")
        except Exception as compile_exc:
            print(f"[ADVERTENCIA] No se pudo compilar el modelo: {compile_exc}")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def load_model_and_tokenizer(variant: str = DEFAULT_MODEL_VARIANT):
    """
    Cargar modelo y tokenizer. Solo LoRA 4bit está disponible.
    
    Args:
        variant: Variante del modelo (ignorado, siempre usa LoRA 4bit)
        
    Returns:
        tuple: (model, tokenizer, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("✅ Cargando modelo LoRA 4bit...")
    return _load_lora_variant(device)
