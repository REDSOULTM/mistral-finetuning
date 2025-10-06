"""
Optimizaciones para respuesta inmediata en WhatsApp.
Enfocado en preparar y entregar respuestas lo más rápido posible.
"""

from __future__ import annotations
import sys
import time
from typing import Any, Iterator, Optional, Callable, Dict
from contextlib import contextmanager

_ENABLED = False

def _log(msg: str) -> None:
    print(f"[respuesta_inmediata] {msg}", file=sys.stderr)

def enable() -> None:
    """Activa capacidades de respuesta inmediata para WhatsApp."""
    global _ENABLED
    if _ENABLED:
        return
    _ENABLED = True
    _log("respuesta inmediata para WhatsApp activada.")

class ImmediateResponseOptimizer:
    """Optimizador que prepara respuestas lo más rápido posible para WhatsApp."""
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.stop_tokens = set()
        
        # Configurar tokens de parada para WhatsApp (respuestas cortas)
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id:
            self.stop_tokens.add(tokenizer.eos_token_id)
        
        # Tokens adicionales de parada para chatbot WhatsApp
        stop_phrases = ["</s>", "<|im_end|>", "Usuario:", "Cliente:", "\n\n"]
        for phrase in stop_phrases:
            try:
                token_ids = tokenizer.encode(phrase, add_special_tokens=False)
                if token_ids:
                    self.stop_tokens.update(token_ids)
            except:
                pass
    
    def generate_immediate_response(
        self, 
        prompt: str, 
        max_new_tokens: int = 80,  # Muy corto para WhatsApp
        temperature: float = 0.05,  # Muy determinístico
    ) -> str:
        """
        Genera una respuesta inmediata optimizada para WhatsApp.
        
        Args:
            prompt: Texto de entrada
            max_new_tokens: Máximo número de tokens (muy limitado para velocidad)
            temperature: Temperatura muy baja para determinismo
        
        Returns:
            str: Respuesta completa optimizada
        """
        if self.model is None or self.tokenizer is None:
            return ""
        
        try:
            import torch
            
            # Tokenizar entrada con configuración ultra-rápida
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                max_length=512,      # Limitar contexto para velocidad
                truncation=True,     # Cortar si es muy largo
                padding=False        # Sin padding innecesario
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Configurar generación ultra-rápida para WhatsApp
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 5,         # Mínimo para respuesta útil
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.8,               # Limitado para velocidad
                "top_k": 15,                # Muy limitado
                "early_stopping": True,     # Parar en cuanto sea coherente
                "length_penalty": 2.0,      # Penalizar mucho respuestas largas
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
                "output_scores": False,
                "output_attentions": False,
                "output_hidden_states": False,
                "return_dict_in_generate": False,
                "num_beams": 1,             # Sin beam search para velocidad
            }
            
            start_time = time.time()
            
            with torch.inference_mode():
                # Generar respuesta de una vez (no streaming visual)
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
                
                # Extraer solo la respuesta nueva
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                
                # Decodificar respuesta
                response = self.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                ).strip()
                
                # Limpiar respuesta para WhatsApp
                response = self._clean_whatsapp_response(response)
                
                generation_time = (time.time() - start_time) * 1000
                _log(f"respuesta generada en {generation_time:.1f}ms: {len(response)} chars")
                
                return response
                        
        except Exception as e:
            _log(f"error en generación inmediata: {e}")
            return ""
    
    def _clean_whatsapp_response(self, text: str) -> str:
        """Limpia la respuesta para WhatsApp (corta, directa, sin artifacts)."""
        if not text:
            return ""
        
        # Remover artefactos comunes
        text = text.strip()
        
        # Cortar en primera línea doble (fin natural de respuesta WhatsApp)
        if "\n\n" in text:
            text = text.split("\n\n")[0].strip()
        
        # Cortar en marcadores de fin
        end_markers = ["Usuario:", "Cliente:", "---", "```", "</"]
        for marker in end_markers:
            if marker in text:
                text = text.split(marker)[0].strip()
        
        # Asegurar que termina apropiadamente
        if text and not text.endswith(('.', '!', '?', ':')):
            # Si no termina en puntuación, encontrar el último punto/signo
            for i in range(len(text) - 1, -1, -1):
                if text[i] in '.!?':
                    text = text[:i+1]
                    break
        
        # Límite de caracteres para WhatsApp (mensajes cortos)
        if len(text) > 300:
            text = text[:297] + "..."
        
        return text

    def prepare_fast_context(self, user_state: Dict[str, Any]) -> str:
        """Prepara un contexto mínimo ultra-rápido para generación WhatsApp."""
        # Solo la información absolutamente esencial
        essential_fields = []
        
        for field, value in user_state.items():
            if value and field in ["origen", "destino", "fecha", "cantidad"]:
                essential_fields.append(f"{field}: {value}")
        
        if essential_fields:
            return "Estado: " + ", ".join(essential_fields)
        return ""

# =====================================
# FUNCIONES DE OPTIMIZACIÓN PARA WHATSAPP
# =====================================

def create_immediate_response_optimizer(model, tokenizer, device="cpu") -> ImmediateResponseOptimizer:
    """Factory para crear un optimizador de respuesta inmediata."""
    return ImmediateResponseOptimizer(model, tokenizer, device)

@contextmanager
def immediate_response_context():
    """Context manager para configurar respuesta inmediata óptima."""
    try:
        import torch
        import gc
        
        # Configurar para respuesta inmediata
        prev_gc = gc.isenabled()
        gc.disable()  # Deshabilitar GC durante generación crítica
        
        old_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True
        
        yield
        
        # Restaurar
        torch.backends.cudnn.benchmark = old_benchmark
        if prev_gc:
            gc.enable()
    except:
        yield

def optimize_for_immediate_response(model, tokenizer):
    """Optimiza modelo y tokenizer para respuesta inmediata en WhatsApp."""
    try:
        # Optimizar modelo para respuesta instantánea
        if model and hasattr(model, 'eval'):
            model.eval()
            
            # Configuración específica para respuestas inmediatas
            if hasattr(model, 'config'):
                config = model.config
                config.output_attentions = False
                config.output_hidden_states = False
                config.use_cache = True  # Importante para velocidad
        
        # Optimizar tokenizer para velocidad
        if tokenizer:
            if hasattr(tokenizer, 'padding_side'):
                tokenizer.padding_side = "left"
            
            # Asegurar configuración rápida
            if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
                if hasattr(tokenizer, 'eos_token'):
                    tokenizer.pad_token = tokenizer.eos_token
        
        _log("modelo y tokenizer optimizados para respuesta inmediata.")
        return model, tokenizer
        
    except Exception as e:
        _log(f"error optimizando para respuesta inmediata: {e}")
        return model, tokenizer

def get_whatsapp_quick_config() -> Dict[str, Any]:
    """Configuración específica para respuestas rápidas de WhatsApp."""
    return {
        "max_new_tokens": 60,       # Muy corto para WhatsApp
        "min_new_tokens": 3,        # Mínimo útil
        "temperature": 0.05,        # Muy determinístico
        "top_p": 0.8,
        "top_k": 10,
        "do_sample": True,
        "early_stopping": True,
        "length_penalty": 2.5,      # Fuertemente penalizar respuestas largas
        "repetition_penalty": 1.1,
        "use_cache": True,
        "num_beams": 1,             # Sin beam search
        "output_scores": False,
        "output_attentions": False,
        "output_hidden_states": False,
    }

# =====================================
# FUNCIONES DE CONVENIENCIA
# =====================================

def generate_whatsapp_response(model, tokenizer, prompt: str, device="cpu") -> str:
    """
    Función de conveniencia para generar respuesta inmediata para WhatsApp.
    
    Optimizada para velocidad máxima y respuestas cortas apropiadas para chat.
    """
    optimizer = ImmediateResponseOptimizer(model, tokenizer, device)
    
    with immediate_response_context():
        return optimizer.generate_immediate_response(
            prompt=prompt,
            max_new_tokens=80,
            temperature=0.05
        )

def create_minimal_whatsapp_prompt(base_prompt: str, user_input: str, context: str = "") -> str:
    """Crea un prompt minimalista para WhatsApp ultra-rápido."""
    # Prompt ultra-minimalista para máxima velocidad
    essential = f"Transportes Miramar. Usuario: {user_input}"
    
    if context:
        essential += f" Context: {context}"
    
    essential += " Respuesta:"
    
    return essential

__all__ = [
    "enable",
    "ImmediateResponseOptimizer", 
    "create_immediate_response_optimizer",
    "immediate_response_context",
    "optimize_for_immediate_response",
    "generate_whatsapp_response",
    "get_whatsapp_quick_config",
    "create_minimal_whatsapp_prompt"
]
