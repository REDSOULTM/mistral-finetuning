#!/usr/bin/env python3
"""
Script para probar la extracción de datos del bot Miramar.
Usa el sistema completo de vendedor.py para extraer datos correctamente.
"""

import sys
import os
import json

# Agregar el directorio al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from miramar_bot.vendedor import MiramarSellerBot
from miramar_bot.cargar_modelo import load_model_and_tokenizer


def test_extraccion_datos():
    """
    Prueba específica de extracción de datos usando casos críticos.
    """
    print("🚀 Cargando modelo para prueba de extracción...")
    
    # Cargar modelo
    try:
        model, tokenizer, device = load_model_and_tokenizer("lora_4bit")
        bot = MiramarSellerBot(model, tokenizer, device)
        print("✅ Modelo cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # Casos de prueba críticos
    casos_prueba = [
        "Necesito cotizar desde Av Curauma Sur 1826 hasta Valparaíso el 10/10 a las 2pm, somos 3 personas",
        "De viña del mar a santiago, mañana a las 8am, somos 5",
        "Desde aeropuerto hasta av libertad 1234, para el 15 de octubre",
        "Viaje de Santiago a Valparaíso",
        "Desde mi casa hasta el mall",
        "El 27 de junio a las 3pm, somos 4 personas sin regreso"
    ]
    
    print("\n📋 PRUEBAS DE EXTRACCIÓN DE DATOS:")
    print("=" * 60)
    
    for i, caso in enumerate(casos_prueba, 1):
        print(f"\n🧪 CASO {i}: {caso}")
        print("-" * 40)
        
        # Resetear bot para cada caso
        bot.reset()
        
        # Usar validate_input_llm directamente para ver qué extrae
        datos_extraidos = bot.validate_input_llm(caso)
        print(f"📊 Datos extraídos por LLM: {json.dumps(datos_extraidos, ensure_ascii=False, indent=2)}")
        
        # Usar el flujo completo run_step
        respuesta = bot.run_step(caso)
        print(f"🤖 Respuesta del bot: {respuesta}")
        print(f"📋 Estado final: {json.dumps(bot.state, ensure_ascii=False, indent=2)}")
        
        # Verificar si se extrajo algo
        datos_relevantes = {k: v for k, v in bot.state.items() if v and k != '_address_meta'}
        if datos_relevantes:
            print("✅ EXTRACCIÓN EXITOSA")
        else:
            print("❌ NO SE EXTRAJO NADA")


def test_caso_especifico():
    """
    Prueba el caso específico mencionado por el usuario.
    """
    print("\n🎯 PRUEBA CASO ESPECÍFICO:")
    print("=" * 60)
    
    modelo_input = "Necesito cotizar desde Av Curauma Sur 1826 hasta Valparaíso el 10/10 a las 2pm, somos 3 personas"
    
    try:
        model, tokenizer, device = load_model_and_tokenizer("lora_4bit")
        bot = MiramarSellerBot(model, tokenizer, device)
        
        print(f"📝 Input: {modelo_input}")
        
        # 1. Probar extracción directa
        print("\n1️⃣ EXTRACCIÓN DIRECTA (validate_input_llm):")
        datos_llm = bot.validate_input_llm(modelo_input)
        print(f"   {json.dumps(datos_llm, ensure_ascii=False, indent=3)}")
        
        # 2. Probar flujo completo
        print("\n2️⃣ FLUJO COMPLETO (run_step):")
        respuesta = bot.run_step(modelo_input)
        print(f"   Respuesta: {respuesta}")
        print(f"   Estado: {json.dumps(bot.state, ensure_ascii=False, indent=3)}")
        
        # 3. Verificar extracción de direcciones
        print("\n3️⃣ SISTEMA DE DIRECCIONES:")
        from miramar_bot.direcciones import detect_and_extract_address
        direccion_detectada = detect_and_extract_address(modelo_input)
        print(f"   Dirección detectada: {json.dumps(direccion_detectada, ensure_ascii=False, indent=3)}")
        
        # 4. Análisis de campos
        print("\n4️⃣ ANÁLISIS DE RESULTADOS:")
        campos_esperados = ['origen', 'destino', 'fecha', 'hora', 'cantidad']
        for campo in campos_esperados:
            valor = bot.state.get(campo, '')
            estado = "✅ EXTRAÍDO" if valor else "❌ FALTANTE"
            print(f"   {campo}: '{valor}' - {estado}")
            
    except Exception as e:
        print(f"❌ Error en prueba específica: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔧 DIAGNÓSTICO DE EXTRACCIÓN DE DATOS - BOT MIRAMAR")
    print("=" * 60)
    
    # Ejecutar ambas pruebas
    test_caso_especifico()
    test_extraccion_datos()
    
    print("\n🏁 Diagnóstico completado.")
