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
    
    # Aplicar kwargs optimizados de vLLM si están disponibles (para contexto de carga)
    load_kwargs = {
        "model_name": BASE_MODEL_DIR,
        "adapter_name": os.path.join(LORA_ADAPTER_DIR, "lora_adapter"),  # Ruta correcta al adaptador
        "max_seq_length": 2048,
        "dtype": torch.float16 if use_4bit else torch.float32,  # Dtype válido para Unsloth
        "load_in_4bit": use_4bit,
        "float8_kv_cache": use_4bit,
        "fast_inference": use_4bit,
    }
    
    if OPT_HOOKS.get("apply_vllm_kwargs"):
        try:
            load_kwargs.update(OPT_HOOKS["apply_vllm_kwargs"]())
        except Exception:
            pass  # Si falla, usar kwargs por defecto
    
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    FastLanguageModel.for_inference(model)
    
    # Aplicar optimizaciones de velocidad extrema si están disponibles
    if OPT_HOOKS.get("optimize_model_for_speed"):
        try:
            model = OPT_HOOKS["optimize_model_for_speed"](model)
            print("🚀 Modelo optimizado para velocidad extrema.")
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo optimizar modelo para velocidad: {e}")
    
    if OPT_HOOKS.get("optimize_tokenizer_for_speed"):
        try:
            tokenizer = OPT_HOOKS["optimize_tokenizer_for_speed"](tokenizer)
            print("🚀 Tokenizer optimizado para velocidad extrema.")
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo optimizar tokenizer para velocidad: {e}")
    
    # Aplicar optimización de compilación si está disponible
    if device == "cuda":
        if OPT_HOOKS.get("compile_model"):
            try:
                model = OPT_HOOKS["compile_model"](model)
                print("Modelo optimizado con torch.compile vía hooks.")
            except Exception as compile_exc:
                print(f"[ADVERTENCIA] No se pudo aplicar compilación optimizada: {compile_exc}")
                # Fallback al método original
                if hasattr(torch, "compile"):
                    try:
                        model = torch.compile(model, mode="max-autotune")
                        print("Modelo compilado con torch.compile (fallback).")
                    except Exception as fallback_exc:
                        print(f"[ADVERTENCIA] Fallback de compilación también falló: {fallback_exc}")
        elif hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="max-autotune")
                print("Modelo compilado con torch.compile.")
            except Exception as compile_exc:
                print(f"[ADVERTENCIA] No se pudo compilar el modelo con torch.compile: {compile_exc}")
    
    # Aplicar optimizaciones del tokenizer si están disponibles
    if OPT_HOOKS.get("tune_tokenizer"):
        try:
            OPT_HOOKS["tune_tokenizer"](tokenizer)
        except Exception:
            pass
    
    # Aplicar optimizaciones para streaming si están disponibles
    if OPT_HOOKS.get("optimize_for_streaming"):
        try:
            model, tokenizer = OPT_HOOKS["optimize_for_streaming"](model, tokenizer)
            print("📺 Modelo y tokenizer optimizados para streaming.")
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo optimizar para streaming: {e}")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def _load_awq_variant() -> tuple[object, AutoTokenizer, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("El modelo cuantizado AWQ requiere GPU disponible.")
    model_dir = QUANTIZED_AWQ_MODEL_DIR
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"No se encontró el modelo cuantizado AWQ en {model_dir}. Ejecuta el script de fusión/cuántización."
        )

    from awq import AutoAWQForCausalLM

    print("Cargando modelo fusionado cuantizado AWQ 3b...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    
    # Aplicar kwargs optimizados de vLLM si están disponibles
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",  # Corregido: usar "auto" en lugar de "cuda"
        "fuse_layers": True,
    }
    
    if OPT_HOOKS.get("apply_vllm_kwargs"):
        try:
            vllm_kwargs = OPT_HOOKS["apply_vllm_kwargs"]()
            # Solo usar kwargs compatibles con AWQ
            compatible_kwargs = {k: v for k, v in vllm_kwargs.items() 
                               if k in ["device_map", "trust_remote_code"]}
            load_kwargs.update(compatible_kwargs)
        except Exception:
            pass  # Si falla, usar kwargs por defecto
    
    model = AutoAWQForCausalLM.from_quantized(model_dir, **load_kwargs)
    model.eval()
    
    # Aplicar optimizaciones del tokenizer si están disponibles
    if OPT_HOOKS.get("tune_tokenizer"):
        try:
            OPT_HOOKS["tune_tokenizer"](tokenizer)
        except Exception:
            pass
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Modelo AWQ 3b cargado.\n")
    return model, tokenizer, "cuda"


def _load_base_variant(device: str):
    """Carga solo el modelo base sin LoRA (TEMPORAL)."""
    
    print("Cargando modelo base sin LoRA...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Aplicar optimizaciones del tokenizer si están disponibles
    if OPT_HOOKS.get("tune_tokenizer"):
        try:
            OPT_HOOKS["tune_tokenizer"](tokenizer)
        except Exception:
            pass
    
    print("Modelo base cargado.\n")
    return model, tokenizer, device


def load_model_and_tokenizer(variant: str | None = None):
    """Carga el modelo y el tokenizador listos para inferencia."""

    target = (variant or DEFAULT_MODEL_VARIANT or "lora_4bit").strip().lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if target in AWQ_ALIASES:
            return _load_awq_variant()
        elif target in {"base", "base_model", "solo_base", "sin_lora"}:
            return _load_base_variant(device)

        model, tokenizer, device = _load_lora_variant(device)
        print("Modelo 4b + LoRA cargado.\n")
        return model, tokenizer, device
    except Exception as exc:
        import traceback
        print(f"[ADVERTENCIA] No se pudo cargar el modelo ({target}): {exc}")
        print(f"Error completo:")
        traceback.print_exc()
        print(f"→ Ejecutando en modo reducido (sin LLM).")
        return None, None, device


__all__ = ["load_model_and_tokenizer"]
