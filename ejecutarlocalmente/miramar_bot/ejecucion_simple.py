"""Chat simple y funcional para Transportes Miramar."""

from __future__ import annotations

import re
import time

from .vendedor import MiramarSellerBot
from .cargar_modelo import load_model_and_tokenizer
from .utilidades import sanitize_print, print_bot_turn


def _emit(bot: MiramarSellerBot, *parts: str) -> None:
    text = "\n".join(sanitize_print(part) for part in parts if part)
    if text:
        print_bot_turn(text, bot.state)


def main(model_variant: str | None = None, use_kv_cache: bool = True) -> None:
    """Ejecuta el chat de WhatsApp para Transportes Miramar."""
    print("🔧 Cargando modelo optimizado...")
    start_time = time.perf_counter()
    
    model, tokenizer, device = load_model_and_tokenizer(model_variant or "lora_4bit")
    
    if model is not None and tokenizer is not None:
        # Warmup simple del modelo
        print("🔥 Realizando warmup del modelo...")
        try:
            warmup_prompt = "Hola"
            _ = tokenizer.encode(warmup_prompt, return_tensors="pt")
            print("✅ Warmup completado")
        except Exception as e:
            print(f"⚠️  Warmup fallido, continuando: {e}")
    
    load_time = time.perf_counter() - start_time
    print(f"✅ Modelo cargado y optimizado en {load_time:.2f}s")
    print("✅ Optimizaciones aplicadas:")
    print("   • Quantización 4-bit + KV cache 8-bit")
    print("   • Flash Attention + precision medium")
    print("   • Cache incremental past_key_values")
    print("   • torch.compile max-autotune")
    print()
    
    bot = MiramarSellerBot(model, tokenizer, device, use_kv_cache=use_kv_cache)

    # Mostrar saludo inicial
    print("═" * 60)
    print("🚛 TRANSPORTES MIRAMAR - CHAT WHATSAPP")
    print("═" * 60)
    greeting = bot.initial_prompt()
    if greeting:
        print(f"Transportes Miramar: {greeting}")
    print("(Escribe 'salir' para terminar)")
    print("-" * 60)

    # Loop principal de conversación
    while True:
        print("Cliente: ", end="", flush=True)
        try:
            user_input = input()
        except EOFError:
            print("\nTransporte Miramar: Conversación finalizada.")
            break
        except KeyboardInterrupt:
            print("\nTransporte Miramar: Conversación interrumpida.")
            break

        user_input = (user_input or "").strip()

        # Comandos de salida
        if user_input.lower() in ['salir', 'exit', 'quit'] or re.search(r"\b(chao|adiós|adios|hasta luego|nos vemos|bye)\b", user_input, re.IGNORECASE):
            despedida = bot.generate_goodbye_message(user_input) or "¡Gracias por contactar Transportes Miramar! ¡Que tengas un buen día!"
            print(f"Transportes Miramar: {despedida}")
            break

        if not user_input:
            continue

        # Procesar respuesta del usuario
        inicio = time.perf_counter()
        try:
            # Usar el método de validación avanzada si está disponible
            result = bot.process_user_response_with_validation(user_input)
            
            if result.get("needs_clarification"):
                respuesta = result.get("clarification_message", "")
            else:
                # Datos válidos extraídos
                if result.get("extracted_data"):
                    for key, value in result["extracted_data"].items():
                        bot.state[key] = value
                
                # Obtener siguiente pregunta
                respuesta = bot.get_next_question()
                if not respuesta:
                    respuesta = bot.generate_final_message()
                    
        except Exception as e:
            # Fallback simple si falla el método avanzado
            print(f"⚠️  Error en procesamiento avanzado: {e}")
            respuesta = "Lo siento, hubo un problema procesando tu mensaje. ¿Podrías repetir tu solicitud?"
        
        total_ms = (time.perf_counter() - inicio) * 1000

        # Mostrar respuesta
        if respuesta:
            print(f"Transportes Miramar: {respuesta}")
            print(f"⏱ {total_ms:.0f} ms")
        print("-" * 60)


__all__ = ["main"]
