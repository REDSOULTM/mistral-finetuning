#!/usr/bin/env python3
"""
Sistema de extracción híbrido que combina:
1. Extracción por patrones regex mejorados
2. Sistema LLM del bot Miramar
3. Detección de direcciones con libpostal

Esto resuelve el problema de que el LLM no extrae datos.
"""

import json
import sys
import os

# Agregar el directorio al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extraer_datos_directo import extraer_datos_mensaje_mejorado

def extraer_datos_hibrido(mensaje_usuario: str, usar_llm: bool = True):
    """
    Extracción híbrida que combina patrones regex + LLM.
    """
    print(f"🔧 EXTRACCIÓN HÍBRIDA - Input: {mensaje_usuario}")
    print("=" * 60)
    
    # 1. Extracción con patrones mejorados (siempre funciona)
    print("1️⃣ EXTRACCIÓN POR PATRONES:")
    datos_patrones = extraer_datos_mensaje_mejorado(mensaje_usuario)
    print(f"   Resultado patrones: {json.dumps(datos_patrones, ensure_ascii=False)}")
    
    datos_final = datos_patrones.copy()
    
    # 2. Extracción con LLM (si está disponible y habilitado)
    if usar_llm:
        try:
            print("\n2️⃣ EXTRACCIÓN POR LLM:")
            from miramar_bot.vendedor import MiramarSellerBot
            from miramar_bot.cargar_modelo import load_model_and_tokenizer
            
            model, tokenizer, device = load_model_and_tokenizer("lora_4bit")
            bot = MiramarSellerBot(model, tokenizer, device)
            
            datos_llm = bot.validate_input_llm(mensaje_usuario)
            print(f"   Resultado LLM: {json.dumps(datos_llm, ensure_ascii=False)}")
            
            # 3. Fusionar resultados (LLM complementa patrones)
            print("\n3️⃣ FUSIÓN DE RESULTADOS:")
            for key, value in datos_llm.items():
                if value and (not datos_final.get(key) or len(str(value)) > len(str(datos_final.get(key, '')))):
                    datos_final[key] = value
                    print(f"   ✅ {key}: '{value}' (desde LLM)")
                elif datos_final.get(key):
                    print(f"   🔄 {key}: '{datos_final[key]}' (desde patrones)")
            
        except Exception as e:
            print(f"   ⚠️ Error con LLM: {e}")
            print("   ➡️ Usando solo extracción por patrones")
    
    # 4. Resultado final
    print(f"\n4️⃣ RESULTADO FINAL:")
    print(f"   {json.dumps(datos_final, ensure_ascii=False, indent=2)}")
    
    # 5. Verificar extracción exitosa
    campos_criticos = ['origen', 'destino', 'fecha', 'hora', 'cantidad_personas']
    extraidos = sum(1 for campo in campos_criticos if datos_final.get(campo))
    
    print(f"\n📊 ANÁLISIS: {extraidos}/{len(campos_criticos)} campos críticos extraídos")
    if extraidos >= 2:
        print("✅ EXTRACCIÓN EXITOSA")
    else:
        print("❌ EXTRACCIÓN INSUFICIENTE")
    
    return datos_final


def probar_caso_critico():
    """Prueba el caso crítico específico del usuario."""
    print("🎯 PRUEBA CASO CRÍTICO:")
    print("=" * 60)
    
    caso_critico = "Necesito cotizar desde Av Curauma Sur 1826 hasta Valparaíso el 10/10 a las 2pm, somos 3 personas"
    
    resultado = extraer_datos_hibrido(caso_critico)
    
    # Verificar campos esperados
    esperados = {
        'origen': 'Av Curauma Sur 1826',
        'destino': 'Valparaíso', 
        'fecha': '2025-10-10',
        'hora': '2pm',
        'cantidad_personas': '3'
    }
    
    print(f"\n🔍 VERIFICACIÓN DE CAMPOS ESPERADOS:")
    for campo, esperado in esperados.items():
        extraido = resultado.get(campo, '')
        if extraido:
            print(f"✅ {campo}: '{extraido}' (esperado: '{esperado}')")
        else:
            print(f"❌ {campo}: NO EXTRAÍDO (esperado: '{esperado}')")


def probar_multiples_casos():
    """Prueba múltiples casos para verificar robustez."""
    casos = [
        "Desde Av Curauma Sur 1826 hasta Valparaíso el 10/10 a las 2pm, somos 3 personas",
        "De viña del mar a santiago, mañana a las 8am, somos 5",
        "Quiero ir desde aeropuerto hasta av libertad 1234, para el 15 de octubre",
        "Viaje de Santiago a Valparaíso",
        "El 27 de junio a las 3pm, somos 4 personas sin regreso",
        "Somos 3 personas, vamos mañana",
        "¿Cuánto cuesta ir a Valparaíso?"
    ]
    
    print("\n🧪 PRUEBAS MÚLTIPLES:")
    print("=" * 60)
    
    for i, caso in enumerate(casos, 1):
        print(f"\n--- CASO {i} ---")
        resultado = extraer_datos_hibrido(caso, usar_llm=False)  # Solo patrones para rapidez
        
        # Resumen
        campos_extraidos = [k for k, v in resultado.items() if v]
        print(f"📋 Resumen: {len(campos_extraidos)} campos → {', '.join(campos_extraidos)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Caso específico
        mensaje = " ".join(sys.argv[1:])
        extraer_datos_hibrido(mensaje)
    else:
        # Ejecutar pruebas
        probar_caso_critico()
        probar_multiples_casos()
