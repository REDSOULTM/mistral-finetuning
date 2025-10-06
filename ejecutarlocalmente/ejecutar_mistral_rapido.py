#!/usr/bin/env python3
"""Versión ultra-rápida del bot de Miramar que prioriza velocidad sobre funciones avanzadas."""

from __future__ import annotations

import argparse
import sys
import os
import time

# Configuración para máxima velocidad
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Evitar warnings
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"       # Async CUDA

print("🚀 Iniciando sistema Miramar MODO RÁPIDO...")

def test_extraccion_simple():
    """Prueba rápida de extracción de datos usando sistema híbrido."""
    print("\n🧪 PRUEBA RÁPIDA DE EXTRACCIÓN HÍBRIDA:")
    print("-" * 40)
    
    try:
        from sistema_extraccion_hibrido import extraer_datos_hibrido
        
        # Caso de prueba crítico
        test_input = "Desde Av Curauma Sur 1826 hasta Valparaíso el 10/10 a las 2pm, somos 3 personas"
        print(f"📝 Probando: {test_input}")
        
        # Usar sistema híbrido SIN LLM para máxima velocidad
        resultado = extraer_datos_hibrido(test_input, usar_llm=False)
        
        # Verificar si extrajo datos importantes
        campos_criticos = ['origen', 'destino', 'fecha', 'hora', 'cantidad_personas']
        extraidos = sum(1 for campo in campos_criticos if resultado.get(campo))
        
        if extraidos >= 3:
            print("✅ EXTRACCIÓN HÍBRIDA FUNCIONA CORRECTAMENTE")
        else:
            print("❌ EXTRACCIÓN HÍBRIDA NECESITA AJUSTES")
            
        return resultado
            
    except Exception as e:
        print(f"❌ Error en prueba híbrida: {e}")
        import traceback
        traceback.print_exc()
        return {}


def chat_rapido_con_extraccion():
    """Chat que usa solo extracción de datos sin LLM para máxima velocidad."""
    print("\n💬 CHAT RÁPIDO CON EXTRACCIÓN (SIN LLM):")
    print("Escribe 'exit' para salir")
    print("-" * 40)
    
    try:
        from sistema_extraccion_hibrido import extraer_datos_hibrido
        
        # Estado del bot
        datos_cliente = {
            "origen": "",
            "destino": "",
            "fecha": "",
            "hora": "",
            "regreso": "",
            "cantidad": 0,
            "comentario adicional": ""
        }
        
        # Preguntas template basadas en el estado
        def generar_pregunta_siguiente(datos):
            if not datos["origen"]:
                return "¿Desde dónde inicia tu viaje?"
            elif not datos["destino"]:
                return "¿Hacia dónde deseas llegar?"
            elif not datos["fecha"]:
                return "¿Para qué fecha necesitas el traslado?"
            elif not datos["hora"]:
                return "¿A qué hora necesitas salir?"
            elif not datos["cantidad"] or datos["cantidad"] == 0:
                return "¿Cuántas personas viajarán?"
            else:
                return "¿Necesitas regreso? ¿Algún comentario adicional?"
        
        print("Transportes Miramar: Hola, soy Transportes Miramar. ¿En qué puedo ayudarte con tu traslado?")
        
        while True:
            try:
                user_input = input("\nCliente: ").strip()
                if user_input.lower() in ['exit', 'quit', 'salir']:
                    break
                if not user_input:
                    continue
                
                start_time = time.time()
                
                # Extraer datos del mensaje (SIN LLM para velocidad)
                datos_nuevos = extraer_datos_hibrido(user_input, usar_llm=False)
                
                # Actualizar estado con datos nuevos
                for campo, valor in datos_nuevos.items():
                    if valor and valor != "" and valor != 0:
                        datos_cliente[campo] = valor
                
                # Generar respuesta basada en template
                respuesta = ""
                datos_extraidos = {k: v for k, v in datos_cliente.items() if v and v != "" and v != 0}
                
                if datos_extraidos:
                    respuesta += "Perfecto, tengo: "
                    respuesta += ", ".join([f"{k}: {v}" for k, v in datos_extraidos.items()])
                    respuesta += ". "
                
                # Siguiente pregunta
                siguiente_pregunta = generar_pregunta_siguiente(datos_cliente)
                respuesta += siguiente_pregunta
                
                end_time = time.time()
                
                print(f"\nTransportes Miramar: {respuesta}")
                print(f"⏱ {(end_time - start_time) * 1000:.0f} ms")
                
                # Mostrar estado actual
                print(f"Estado: {datos_cliente}")
                
            except (EOFError, KeyboardInterrupt):
                break
                
        print("\n👋 Chat finalizado.")
        
    except Exception as e:
        print(f"❌ Error en chat rápido: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lanzador RÁPIDO del chat Miramar")
    parser.add_argument(
        "--test-extraction",
        action="store_true",
        help="Ejecuta una prueba rápida de extracción de datos.",
    )
    parser.add_argument(
        "--chat-fast",
        action="store_true", 
        help="Inicia chat rápido con extracción sin LLM.",
    )
    args = parser.parse_args()
    
    if args.test_extraction:
        test_extraccion_simple()
        return
    
    if args.chat_fast:
        chat_rapido_con_extraccion()
        return
        
    # Por defecto, mostrar opciones
    print("\nOpciones disponibles:")
    print("  --test-extraction  : Prueba rápida de extracción")
    print("  --chat-fast        : Chat rápido sin LLM (respuestas < 100ms)")
    print("\nEjemplo: python3 ejecutar_mistral_rapido.py --chat-fast")


if __name__ == "__main__":
    main()
