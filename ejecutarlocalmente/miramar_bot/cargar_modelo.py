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


def _get_optimal_vram_config():
    """
    Detecta automáticamente la VRAM disponible y retorna configuración óptima BALANCEADA.
    Prioriza velocidad de respuesta sobre capacidad máxima.
    """
    if not torch.cuda.is_available():
        return {"max_seq_length": 512, "gpu_memory_utilization": 0.5, "max_num_seqs": 8}
    
    # Detectar VRAM total
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # CONFIGURACIÓN BALANCEADA - Velocidad > Capacidad máxima
    if vram_gb >= 20:  # RTX 4090, A6000, etc.
        return {
            "max_seq_length": 2048,  # SUFICIENTE para prompts largos
            "gpu_memory_utilization": 0.8,  # REDUCIDO para estabilidad
            "max_num_seqs": 32,  # REDUCIDO para latencia baja
            "max_num_batched_tokens": 1024
        }
    elif vram_gb >= 15:  # RTX 4060 Ti 16GB, RTX 4080
        return {
            "max_seq_length": 2048,  # AUMENTADO para manejar prompts largos
            "gpu_memory_utilization": 0.75,  # MODERADO para balance
            "max_num_seqs": 16,  # REDUCIDO para menor latencia individual (de 24 a 16)
            "max_num_batched_tokens": 896  # Optimizado para velocidad (de 1024 a 896)
        }
    elif vram_gb >= 10:  # RTX 3060 Ti 12GB, RTX 4070
        return {
            "max_seq_length": 1536,
            "gpu_memory_utilization": 0.7,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 768
        }
    else:  # RTX 3060 6GB, RTX 4060
        return {
            "max_seq_length": 1024,
            "gpu_memory_utilization": 0.6,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 512
        }


def _load_lora_variant(device: str, disable_compile: bool = False):
    use_4bit = device == "cuda"
    
    # DETECCIÓN AUTOMÁTICA DE VRAM Y CONFIGURACIÓN ÓPTIMA
    vram_gb = 0  # Inicializar variable
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_config = _get_optimal_vram_config()
        print(f"🎮 GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM total: {vram_gb:.1f}GB")
        print(f"🚀 Configuración automática: {vram_config['max_seq_length']} tokens, {vram_config['gpu_memory_utilization']*100:.0f}% VRAM")
        
        if disable_compile:
            print("⚡ MODO DESARROLLO RÁPIDO: torch.compile deshabilitado")
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        try:
            # Flash Attention + precision hints para mejor rendimiento
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass
    else:
        vram_config = {"max_seq_length": 512, "gpu_memory_utilization": 0.5, "max_num_seqs": 16}
    print("Cargando modelo base 4b + LoRA con optimizaciones...")
    
    # Configuración OPTIMIZADA para RTX 4060 Ti 16GB (memoria abundante)
    # 🔧 CONFIGURACIONES POR VRAM DISPONIBLE:
    # 
    # 📱 Para 6GB VRAM (RTX 3060, RTX 4060):
    #    "max_seq_length": 1024, "gpu_memory_utilization": 0.75, "max_num_seqs": 32
    # 
    # 🎮 Para 12GB VRAM (RTX 3060 Ti, RTX 4070):
    #    "max_seq_length": 1536, "gpu_memory_utilization": 0.8, "max_num_seqs": 48
    # 
    # 🚀 Para 16GB VRAM (RTX 4060 Ti, RTX 4080):
    #    "max_seq_length": 2048, "gpu_memory_utilization": 0.95, "max_num_seqs": 128
    #
    # 💎 Para 24GB+ VRAM (RTX 4090, RTX A6000):
    #    "max_seq_length": 4096, "gpu_memory_utilization": 0.95, "max_num_seqs": 256
    
    load_kwargs = {
        "model_name": BASE_MODEL_DIR,
        "adapter_name": os.path.join(LORA_ADAPTER_DIR, "lora_adapter"),
        
        # 🧠 MEMORIA CONVERSACIONAL - Configuración BALANCEADA para velocidad
        "max_seq_length": vram_config["max_seq_length"],
        
        # ⚡ PRECISIÓN Y VELOCIDAD - Configuración SIMPLE
        "dtype": torch.float16 if use_4bit else torch.float32,
        "load_in_4bit": use_4bit,  # Quantización 4-bit esencial
        
        #  OPTIMIZACIONES BÁSICAS - Solo las esenciales para Unsloth
        "use_cache": True,  # Cache incremental esencial
    }
    
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    FastLanguageModel.for_inference(model)
    
    # torch.compile HABILITADO por defecto para máxima velocidad (compilación única)
    compile_enabled = False
    if device == "cuda" and hasattr(torch, "compile") and not disable_compile:
        try:
            print("🔥 Compilando modelo con torch.compile...")
            print("   ⏱️  Esto tomará ~30-60s la primera vez, pero después será ultra-rápido")
            
            # CONFIGURACIÓN OPTIMIZADA para compilación única:
            # - mode="default": Balance velocidad/compilación (no "max-autotune")
            # - fullgraph=False: Evita recompilación en grafos dinámicos
            # - dynamic=True: Maneja diferentes tamaños de entrada sin recompilar
            model = torch.compile(
                model, 
                mode="default",      # Compilación rápida, no agresiva
                fullgraph=False,     # CRÍTICO: evita recompilación
                dynamic=True         # CRÍTICO: maneja tamaños variables
            )
            compile_enabled = True
            print("✅ Modelo compilado exitosamente - Respuestas ultra-rápidas habilitadas!")
        except Exception as compile_exc:
            print(f"⚠️  No se pudo compilar el modelo: {compile_exc}")
            print("   Continuando sin torch.compile...")
    else:
        if disable_compile:
            print("✅ Modelo SIN torch.compile (modo desarrollo rápido).")
        else:
            print("✅ Modelo sin torch.compile (CUDA no disponible).")
    
    # Configurar tokenizer para WhatsApp (optimización de prompts)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configurar para respuestas cortas de WhatsApp
    tokenizer.model_max_length = 2048
    
    print("✅ Optimizaciones aplicadas:")
    print(f"   • Quantización 4-bit + KV cache FP8")
    print(f"   • Flash Attention + precision medium")
    print(f"   • Memoria conversacional: {vram_config['max_seq_length']} tokens")
    if device == "cuda":
        print(f"   • VRAM utilizada: {vram_config['gpu_memory_utilization']*100:.0f}% ({vram_config['gpu_memory_utilization']*vram_gb:.1f}GB de {vram_gb:.1f}GB)")
    print(f"   • Concurrencia máxima: {vram_config['max_num_seqs']} secuencias")
    if compile_enabled:
        print("   • 🚀 torch.compile activado - Respuestas ultra-rápidas (~2-7s)")
    elif disable_compile:
        print("   • MODO DESARROLLO: sin torch.compile (respuestas ~12-22s)")
    else:
        print("   • Sin torch.compile (CUDA no disponible)")
    
    return model, tokenizer, device


def load_model_and_tokenizer(variant: str = DEFAULT_MODEL_VARIANT):
    """
    Cargar modelo y tokenizer. Solo LoRA 4bit está disponible.
    
    Args:
        variant: Variante del modelo. Si contiene "fast", deshabilita torch.compile
        
    Returns:
        tuple: (model, tokenizer, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Detectar modo rápido basado en la variante
    disable_compile = "fast" in variant.lower()
    
    if disable_compile:
        print("✅ Cargando modelo LoRA 4bit (modo desarrollo rápido - sin torch.compile)...")
    else:
        print("✅ Cargando modelo LoRA 4bit (con torch.compile para máxima velocidad)...")
    
    return _load_lora_variant(device, disable_compile=disable_compile)
