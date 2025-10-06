#!/usr/bin/env python3
"""
Prueba específica de los casos reales de WhatsApp para validar que el sistema funciona.
Casos extraídos de las imágenes proporcionadas por el usuario.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extraer_datos_directo import extraer_datos_mensaje_mejorado

def probar_conversaciones_whatsapp():
    """Prueba los casos específicos de las conversaciones reales de WhatsApp."""
    
    casos_whatsapp = [
        # CONVERSACIÓN 1
        {
            "descripcion": "Conversación 1 - Viaje genérico",
            "input": "Sí, quisiera hacer un viaje desde viña del mar a santiago",
            "esperado": {
                "origen": "viña del mar",
                "destino": "santiago"
            }
        },
        {
            "descripcion": "Conversación 1 - Dirección específica", 
            "input": "Sí, 1 norte 1161, viña del mar",
            "esperado": {
                "origen": "1 norte 1161"
            }
        },
        
        # CONVERSACIÓN 2
        {
            "descripcion": "Conversación 2 - Ruta compleja",
            "input": "Quisiera ir desde Av curauma sur 1826, valparaíso hacia Guindo Santo 1896, La Serena",
            "esperado": {
                "origen": "av curauma sur 1826",
                "destino": "guindo santo 1896, la serena"
            }
        },
        {
            "descripcion": "Conversación 2 - Datos adicionales",
            "input": "Sí, solo ida, somos 4 personas y nos gustaría salir el 27 de junio a las 12hr",
            "esperado": {
                "fecha": "2025-06-27",
                "hora": "12:00",
                "cantidad_personas": "4",
                "regreso": "no"
            }
        },
        
        # CONVERSACIÓN 3
        {
            "descripcion": "Conversación 3 - Direcciones muy complejas",
            "input": "quiero ir a Mall Paseo del Valle - O'Higgins 176, Quillota, Valparaíso, desde Álvarez 70 - 2571923 Viña del Mar, Valparaíso",
            "esperado": {
                "origen": "álvarez 70 - 2571923 viña del mar",
                "destino": "mall paseo del valle - o'higgins 176, quillota"
            }
        },
        {
            "descripcion": "Conversación 3 - Datos grupo grande",
            "input": "sin regreso, 10 personas, para el día 14 de julio a las 8 am",
            "esperado": {
                "fecha": "2025-07-14",
                "hora": "08:00",
                "cantidad_personas": "10",
                "regreso": "no"
            }
        },
        
        # CASOS ADICIONALES TÍPICOS
        {
            "descripcion": "Caso típico - Completo",
            "input": "Necesito cotizar desde Av Curauma Sur 1826 hasta Valparaíso el 10/10 a las 2pm, somos 3 personas",
            "esperado": {
                "origen": "av curauma sur 1826",
                "destino": "valparaíso",
                "fecha": "2025-10-10",
                "hora": "14:00",
                "cantidad_personas": "3"
            }
        }
    ]
    
    print("🎯 PRUEBA DE CONVERSACIONES REALES DE WHATSAPP")
    print("=" * 70)
    
    exitos = 0
    total = len(casos_whatsapp)
    
    for i, caso in enumerate(casos_whatsapp, 1):
        print(f"\n📱 CASO {i}: {caso['descripcion']}")
        print(f"Input: {caso['input']}")
        print("-" * 50)
        
        # Ejecutar extracción
        resultado = extraer_datos_mensaje_mejorado(caso['input'])
        
        # Verificar campos esperados
        campos_correctos = 0
        campos_totales = len(caso['esperado'])
        
        print(f"\n🔍 VERIFICACIÓN:")
        for campo, valor_esperado in caso['esperado'].items():
            valor_extraido = resultado.get(campo, '')
            
            # Comparación flexible (normalizada)
            if valor_extraido:
                extraido_norm = valor_extraido.lower().strip()
                esperado_norm = str(valor_esperado).lower().strip()
                
                # Para fechas y horas, comparación exacta
                if campo in ['fecha', 'hora', 'cantidad_personas']:
                    correcto = extraido_norm == esperado_norm
                else:
                    # Para texto, verificar que contenga palabras clave
                    palabras_esperadas = esperado_norm.split()
                    correcto = any(palabra in extraido_norm for palabra in palabras_esperadas if len(palabra) > 2)
                
                if correcto:
                    campos_correctos += 1
                    print(f"  ✅ {campo}: '{valor_extraido}' (esperado: '{valor_esperado}')")
                else:
                    print(f"  ❌ {campo}: '{valor_extraido}' (esperado: '{valor_esperado}')")
            else:
                print(f"  ❌ {campo}: NO EXTRAÍDO (esperado: '{valor_esperado}')")
        
        # Evaluar caso
        porcentaje = (campos_correctos / campos_totales) * 100
        if porcentaje >= 70:  # 70% o más considerado éxito
            exitos += 1
            print(f"  ✅ CASO EXITOSO: {campos_correctos}/{campos_totales} campos ({porcentaje:.0f}%)")
        else:
            print(f"  ❌ CASO FALLIDO: {campos_correctos}/{campos_totales} campos ({porcentaje:.0f}%)")
    
    print(f"\n📊 RESUMEN FINAL:")
    print(f"   Casos exitosos: {exitos}/{total} ({(exitos/total)*100:.0f}%)")
    
    if exitos >= total * 0.8:  # 80% o más casos exitosos
        print("   🎉 ¡SISTEMA LISTO PARA PRODUCCIÓN!")
    elif exitos >= total * 0.6:  # 60% o más casos exitosos
        print("   ⚠️ Sistema funciona pero necesita ajustes menores")
    else:
        print("   ❌ Sistema necesita mejoras importantes")
    
    return exitos, total


if __name__ == "__main__":
    probar_conversaciones_whatsapp()
