#!/usr/bin/env python3
"""Lanzador principal del bot de Miramar con modelo LLM Mistral fine-tuned."""

from __future__ import annotations

import argparse
import sys
import os

# Configuración de entorno
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def chat_interactivo():
    """Inicia el chat interactivo con el modelo LLM."""
    try:
        # Importar el sistema principal desde miramar_bot
        from miramar_bot import main
        
        # Iniciar chat con el variant por defecto (lora_4bit con torch.compile)
        main(model_variant="lora_4bit")
        
    except Exception as e:
        print(f"❌ Error al iniciar chat: {e}")
        import traceback
        traceback.print_exc()


def ejecutar_con_gui():
    """Inicia la interfaz gráfica."""
    print("\n🖥️ INICIANDO INTERFAZ GRÁFICA")
    print("-" * 60)
    
    try:
        from gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"❌ Error al iniciar GUI: {e}")
        import traceback
        traceback.print_exc()


def test_modelo():
    """Ejecuta pruebas del modelo."""
    print("\n🧪 EJECUTANDO PRUEBAS DEL MODELO")
    print("-" * 60)
    
    try:
        # Casos de prueba
        casos_prueba = [
            "Necesito transporte desde Av Curauma Sur 1826 hasta Valparaíso el 10/10 a las 2pm, somos 3 personas",
            "De viña del mar a santiago mañana a las 8am, somos 5",
            "Quiero cotizar un viaje de Santiago a Valparaíso el 27 de junio a las 3pm, somos 4 personas sin regreso",
        ]
        
        from miramar_bot.cargar_modelo import load_model_and_tokenizer
        from miramar_bot.vendedor import MiramarSellerBot
        
        print("Cargando modelo...")
        model, tokenizer, device = load_model_and_tokenizer()
        print("✅ Modelo cargado")
        
        # Crear instancia del bot
        bot = MiramarSellerBot(model, tokenizer, device)
        
        for i, caso in enumerate(casos_prueba, 1):
            print(f"\n📝 Caso {i}:")
            print(f"Usuario: {caso}")
            
            # Generar respuesta usando el bot
            respuesta = bot.run_step(caso)
            
            print(f"Bot: {respuesta}")
            print(f"Estado: {bot.state}")
            print("-" * 40)
        
        print("\n✅ Pruebas completadas")
        
    except Exception as e:
        print(f"❌ Error en pruebas: {e}")
        import traceback
        traceback.print_exc()


def test_extraccion():
    """Prueba el sistema de extracción de datos."""
    print("\n🔍 PRUEBA DE EXTRACCIÓN DE DATOS")
    print("-" * 60)
    
    try:
        from extraer_datos_simple import extraer_datos_mensaje_mejorado
        
        casos = [
            "Desde Av Curauma Sur 1826 hasta Valparaíso el 10/10 a las 2pm, somos 3 personas",
            "De viña del mar a santiago, mañana a las 8am, somos 5",
            "Viaje de Santiago a Valparaíso el 27 de junio a las 3pm, somos 4 personas sin regreso",
        ]
        
        for i, caso in enumerate(casos, 1):
            print(f"\n📝 Caso {i}: {caso}")
            resultado = extraer_datos_mensaje_mejorado(caso)
            print(f"Resultado: {resultado}")
            print("-" * 40)
        
        print("\n✅ Pruebas de extracción completadas")
        
    except Exception as e:
        print(f"❌ Error en extracción: {e}")
        import traceback
        traceback.print_exc()


def info_sistema():
    """Muestra información del sistema."""
    print("\n📊 INFORMACIÓN DEL SISTEMA")
    print("-" * 60)
    
    import torch
    
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Dispositivo: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Número de GPUs: {torch.cuda.device_count()}")
    
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bot de Transportes Miramar con LLM Mistral fine-tuned"
    )
    
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Inicia el chat interactivo con el modelo LLM (por defecto)",
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Inicia la interfaz gráfica",
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Ejecuta pruebas del modelo",
    )
    
    parser.add_argument(
        "--test-extraction",
        action="store_true",
        help="Prueba el sistema de extracción de datos",
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Muestra información del sistema",
    )
    
    args = parser.parse_args()
    
    # Banner inicial
    print("🚀 Iniciando sistema de Transportes Miramar...")
    
    # Mostrar información del sistema si se solicita
    if args.info:
        info_sistema()
        return
    
    # Ejecutar según la opción seleccionada
    if args.gui:
        ejecutar_con_gui()
    elif args.test:
        test_modelo()
    elif args.test_extraction:
        test_extraccion()
    elif args.chat or not any(vars(args).values()):
        # Por defecto, iniciar chat
        chat_interactivo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
