"""Carga el modelo que utilizará el chatbot Miramar."""

from __future__ import annotations

import os

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from .configuracion import (
    BASE_MODEL_DIR,
    DEFAULT_MODEL_VARIANT,
    LORA_ADAPTER_DIR,
    QUANTIZED_AWQ_MODEL_DIR,
)


AWQ_ALIASES = {
    "awq",
    "awq_3b",
    "awq3b",
    "3b",
    "3bit",
    "fusion",
    "fusion_awq",
    "fused_awq",
}


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


def _load_awq_variant() -> tuple[object, AutoTokenizer, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("El modelo cuantizado AWQ requiere GPU disponible.")
    model_dir = QUANTIZED_AWQ_MODEL_DIR
    print(f"Cargando modelo AWQ cuantizado desde {model_dir}...")
    
    device = "cuda"
    
    # Configuración básica de carga sin optimizaciones externas
    load_kwargs = {
        "pretrained_model_name_or_path": model_dir,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Modelo AWQ cargado correctamente.")
    return model, tokenizer, device


def load_model_and_tokenizer(variant: str = DEFAULT_MODEL_VARIANT):
    """
    Cargar modelo y tokenizer según la variante especificada.
    
    Args:
        variant: Variante del modelo a cargar ('lora_4bit' o 'awq_3b')
        
    Returns:
        tuple: (model, tokenizer, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if variant.lower() in AWQ_ALIASES:
        return _load_awq_variant()
    elif variant.lower() in {"lora", "lora_4bit", "4bit", "4b", "base", "basic"}:
        return _load_lora_variant(device)
    else:
        print(f"Variante desconocida '{variant}', usando LoRA 4bit por defecto.")
        return _load_lora_variant(device)
