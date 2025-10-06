"""Bucle principal de la consola para interactuar con el bot."""

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
    """Ejecuta el loop de conversación en la línea de comandos."""
    model, tokenizer, device = load_model_and_tokenizer(model_variant)
    bot = MiramarSellerBot(model, tokenizer, device, use_kv_cache=use_kv_cache)

    # Conversación inicia inmediatamente en modo cotización.
    first_turn = True

    while True:
        print("Cliente: ", end="", flush=True)
        try:
            user_input = input()
        except EOFError:
            print("\nTransporte Miramar: Conversación finalizada por cierre del canal.")
            break
        except KeyboardInterrupt:
            print("\nTransporte Miramar: Conversación interrumpida.")
            break

        user_input = (user_input or "").strip()

        if re.search(r"\b(chao|adiós|adios|hasta luego|nos vemos|bye)\b", user_input, re.IGNORECASE):
            despedida = bot.generate_goodbye_message(user_input) or ""
            print_bot_turn(despedida, bot.state)
            break

        if not user_input:
            continue

        inicio = time.perf_counter()
        respuesta = bot.run_step(user_input)
        total_ms = (time.perf_counter() - inicio) * 1000

        if first_turn:
            inicio_greeting = time.perf_counter()
            greeting = bot.initial_prompt()
            total_ms += (time.perf_counter() - inicio_greeting) * 1000
            if greeting:
                respuesta = (f"{greeting}\n{respuesta}" if respuesta else greeting)
            first_turn = False

        _emit(bot, respuesta)
        if respuesta:
            print(f"⏱ {total_ms:.0f} ms")


__all__ = ["main"]
