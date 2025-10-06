"""
Optimizaciones para latencia ultra-baja en WhatsApp.
Enfocado en reducir el tiempo total de respuesta (no streaming visual).
"""

from __future__ import annotations
import sys
import os
import time
import hashlib
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager

_ENABLED = False

def _log(msg: str) -> None:
    print(f"[latencia_extrema] {msg}", file=sys.stderr)

def enable() -> None:
    """Activa optimizaciones de latencia extrema para respuestas WhatsApp ultra-rápidas."""
    global _ENABLED
    if _ENABLED:
        return
    _ENABLED = True
    
    # Configuraciones para velocidad máxima
    _configure_system_for_speed()
    _configure_cuda_for_latency()
    _configure_memory_optimization()
    
    _log("optimizaciones de latencia extrema activadas para WhatsApp.")

def _configure_system_for_speed() -> None:
    """Configuraciones de sistema para máxima velocidad."""
    try:
        # Optimizar variables de entorno para operaciones rápidas
        os.environ["OMP_NUM_THREADS"] = "1"  # Evitar overhead en ops pequeñas
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Evitar warnings y overhead
        
        _log("sistema configurado para velocidad máxima.")
    except Exception as e:
        _log(f"error configurando sistema: {e}")

def _configure_cuda_for_latency() -> None:
    """Configuraciones CUDA específicas para latencia mínima."""
    try:
        import torch
        if torch.cuda.is_available():
            # Configuraciones agresivas para latencia mínima
            torch.backends.cudnn.benchmark = True  # Optimizar para tamaños fijos
            torch.backends.cudnn.deterministic = False  # Sacrificar reproducibilidad por velocidad
            torch.backends.cuda.matmul.allow_tf32 = True  # TF32 para speed
            torch.backends.cudnn.allow_tf32 = True
            
            # Cache precompilado
            torch.backends.cuda.enable_flash_sdp(True)
            
            _log("CUDA configurado para latencia mínima.")
    except Exception as e:
        _log(f"error configurando CUDA: {e}")

def _configure_memory_optimization() -> None:
    """Optimizaciones de memoria para acceso ultra-rápido."""
    try:
        import torch
        if torch.cuda.is_available():
            # Configurar memoria para acceso rápido
            torch.cuda.empty_cache()  # Limpiar una vez al inicio
            # Configurar allocator para fragmentación mínima
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            
        _log("memoria optimizada para acceso rápido.")
    except Exception as e:
        _log(f"error optimizando memoria: {e}")

# =====================================
# OPTIMIZACIONES DE GENERACIÓN ESPECÍFICAS PARA WHATSAPP
# =====================================

def get_whatsapp_speed_config() -> Dict[str, Any]:
    """Configuración optimizada para respuestas WhatsApp ultra-rápidas."""
    return {
        # Menos tokens = respuesta más rápida para WhatsApp
        "max_new_tokens": 100,      # Reducido significativamente 
        "min_new_tokens": 8,        # Mínimo para respuestas útiles
        
        # Sampling ultra-agresivo para velocidad
        "do_sample": True,
        "temperature": 0.1,         # Muy determinístico = más rápido
        "top_p": 0.7,              # Limitado fuertemente
        "top_k": 10,               # Solo las 10 opciones más probables
        
        # Parada temprana ultra-agresiva
        "early_stopping": True,
        "length_penalty": 1.5,      # Penalizar mucho las respuestas largas
        
        # Optimizaciones de velocidad
        "use_cache": True,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict_in_generate": False,
        "output_scores": False,
        
        # Parada en primera respuesta coherente
        "num_return_sequences": 1,
        "num_beams": 1,            # Sin beam search = más rápido
    }

def compress_prompt_for_whatsapp(prompt: str) -> str:
    """Comprime prompts agresivamente para WhatsApp (velocidad > completitud)."""
    if not prompt:
        return prompt
    
    # Para WhatsApp, queremos respuestas RÁPIDAS, no perfectas
    lines = prompt.split('\n')
    essential_lines = []
    
    # Solo mantener líneas absolutamente esenciales
    for line in lines:
        line_stripped = line.strip()
        
        # Saltar completamente secciones largas
        if any(skip in line_stripped for skip in [
            "Ejemplos:", "Reglas:", "Contexto:", "Guía de estilo:", 
            "Instrucciones:", "Tarea:", "Referencias:"
        ]):
            continue
        
        # Solo mantener líneas críticas muy cortas
        if line_stripped and len(line_stripped) < 150:
            essential_lines.append(line)
    
    compressed = '\n'.join(essential_lines)
    
    # Límite ultra-agresivo para WhatsApp
    if len(compressed) > 400:  # Muy corto para máxima velocidad
        compressed = compressed[:400] + "..."
    
    return compressed

def create_minimal_whatsapp_templates() -> Dict[str, str]:
    """Templates ultra-minimalistas para WhatsApp."""
    return {
        "confirmation": "Confirmo: {data}",
        "question": "¿{field}?",  # Ultra-directo
        "greeting": "Hola, soy Transportes Miramar. ¿Tu viaje?",
        "extract": "Datos de: {input}",
        "goodbye": "Gracias por elegir Transportes Miramar.",
        "error": "Disculpa, ¿puedes repetir?",
    }

# =====================================
# CACHE ULTRA-AGRESIVO PARA WHATSAPP
# =====================================

class WhatsAppResponseCache:
    """Cache especializado para respuestas WhatsApp ultra-rápidas."""
    
    def __init__(self):
        self.cache = {}
        self.max_size = 20  # Cache pequeño pero muy rápido
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, prompt_hash: str) -> Optional[str]:
        """Obtener respuesta cacheada ultra-rápido."""
        if prompt_hash in self.cache:
            self.hit_count += 1
            return self.cache[prompt_hash]
        self.miss_count += 1
        return None
    
    def set(self, prompt_hash: str, response: str) -> None:
        """Cachear respuesta con LRU simple."""
        if len(self.cache) >= self.max_size:
            # Remover el primer elemento (FIFO simple, más rápido que LRU)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[prompt_hash] = response
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Estadísticas de cache."""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate
        }

# Cache global para WhatsApp
_WHATSAPP_CACHE = WhatsAppResponseCache()

def get_whatsapp_cached_response(prompt: str) -> Optional[str]:
    """Verificar cache para respuesta de WhatsApp."""
    prompt_hash = _hash_prompt_fast(prompt)
    return _WHATSAPP_CACHE.get(prompt_hash)

def cache_whatsapp_response(prompt: str, response: str) -> None:
    """Cachear respuesta de WhatsApp."""
    prompt_hash = _hash_prompt_fast(prompt)
    _WHATSAPP_CACHE.set(prompt_hash, response)

def _hash_prompt_fast(prompt: str) -> str:
    """Hash ultra-rápido para prompts de WhatsApp."""
    # Solo usar primeros 100 chars para hash súper rápido
    truncated = prompt[:100] if len(prompt) > 100 else prompt
    return str(hash(truncated))  # hash() nativo es más rápido que hashlib

# =====================================
# OPTIMIZACIONES DE MODELO PARA WHATSAPP
# =====================================

def optimize_model_for_whatsapp(model):
    """Optimizar modelo específicamente para respuestas WhatsApp."""
    if not model:
        return model
    
    try:
        # Modo evaluación estricto
        model.eval()
        
        # Configuraciones agresivas para WhatsApp
        if hasattr(model, 'config'):
            config = model.config
            
            # Deshabilitar TODAS las salidas innecesarias
            config.output_attentions = False
            config.output_hidden_states = False
            config.use_return_dict = False
        
        # Optimización de precisión para velocidad en WhatsApp
        try:
            import torch
            if torch.cuda.is_available():
                # Half precision para máxima velocidad en WhatsApp
                model = model.half()
                _log("modelo convertido a half precision para WhatsApp.")
        except Exception as e:
            _log(f"half precision falló: {e}")
        
        _log("modelo optimizado específicamente para WhatsApp.")
        return model
        
    except Exception as e:
        _log(f"error optimizando modelo para WhatsApp: {e}")
        return model

def compile_model_for_whatsapp_speed(model):
    """Compilar modelo específicamente para velocidad WhatsApp."""
    try:
        import torch
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            # Configuración ultra-agresiva para WhatsApp
            compiled = torch.compile(
                model,
                mode="max-autotune",     # Máxima optimización
                fullgraph=True,         # Compilar todo
                dynamic=False,          # Sin shapes dinámicas
                backend="inductor",     # Backend más rápido
            )
            _log("modelo compilado con configuración ultra-agresiva para WhatsApp.")
            return compiled
    except Exception as e:
        _log(f"compilación para WhatsApp falló: {e}")
    
    return model

# =====================================
# CONTEXT MANAGER PARA WHATSAPP
# =====================================

@contextmanager
def whatsapp_ultra_speed_context():
    """Context manager para máxima velocidad en WhatsApp."""
    import torch
    
    # Configuraciones previas
    prev_gc_enabled = True
    prev_benchmark = torch.backends.cudnn.benchmark if torch.cuda.is_available() else False
    
    try:
        # Configuración ultra-agresiva para WhatsApp
        import gc
        gc.disable()  # Deshabilitar GC durante generación
        prev_gc_enabled = gc.isenabled()
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        yield
        
    finally:
        # Restaurar configuraciones
        if prev_gc_enabled:
            import gc
            gc.enable()
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = prev_benchmark

# =====================================
# FUNCIONES DE INTEGRACIÓN
# =====================================

def apply_whatsapp_speed_optimizations(model, tokenizer):
    """Aplicar TODAS las optimizaciones para velocidad WhatsApp."""
    # Optimizar modelo
    model = optimize_model_for_whatsapp(model)
    model = compile_model_for_whatsapp_speed(model)
    
    # Optimizar tokenizer
    if tokenizer:
        tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def get_whatsapp_inference_kwargs() -> Dict[str, Any]:
    """Obtener kwargs optimizados para inferencia WhatsApp."""
    config = get_whatsapp_speed_config()
    
    # Añadir configuraciones específicas de contexto
    config.update({
        "cache_implementation": "static",
    })
    
    return config

# Stats para debugging
def get_whatsapp_optimization_stats() -> Dict[str, Any]:
    """Obtener estadísticas de optimizaciones WhatsApp."""
    cache_stats = _WHATSAPP_CACHE.get_stats()
    
    return {
        "enabled": _ENABLED,
        "cache_stats": cache_stats,
        "optimizations_active": [
            "prompt_compression",
            "ultra_fast_generation", 
            "aggressive_caching",
            "model_optimization",
            "whatsapp_templates"
        ]
    }

__all__ = [
    "enable",
    "get_whatsapp_speed_config",
    "compress_prompt_for_whatsapp", 
    "create_minimal_whatsapp_templates",
    "optimize_model_for_whatsapp",
    "compile_model_for_whatsapp_speed",
    "whatsapp_ultra_speed_context",
    "apply_whatsapp_speed_optimizations",
    "get_whatsapp_inference_kwargs",
    "get_whatsapp_cached_response",
    "cache_whatsapp_response",
    "get_whatsapp_optimization_stats",
]
