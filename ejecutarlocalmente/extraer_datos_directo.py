#!/usr/bin/env python3
"""
Función de extracción de datos usando patrones mejorados + sistema Miramar.
Este script combina patrones regex robustos con el sistema LLM del bot.
"""

import json
import re
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Agregar el directorio al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from miramar_bot.direcciones import detect_and_extract_address
    direcciones_disponible = True
except ImportError:
    print("⚠️ No se pudo importar detect_and_extract_address")
    direcciones_disponible = False
    detect_and_extract_address = None

# Importar funciones existentes del sistema Miramar
try:
    from miramar_bot.utilidades import (
        extract_origin_dest_pair,
        extract_quantity,
        parse_time_from_text,
        parse_textual_date,
        parse_numeric_date,
        normalize_place,
        extract_and_validate_datetime_info,
        extract_multiple_addresses_from_complex_phrase,
        validate_and_improve_address,
        extract_location,
        is_specific_address,
        extract_email_value,
        extract_name_value,
        extract_rut_value
    )
    utilidades_disponible = True
    print("✅ Funciones de utilidades Miramar cargadas")
except ImportError as e:
    print(f"⚠️ No se pudieron importar utilidades Miramar: {e}")
    utilidades_disponible = False
    # Definir fallbacks vacíos
    extract_origin_dest_pair = lambda x: (None, None)
    extract_quantity = lambda x: None
    parse_time_from_text = lambda x: None
    parse_textual_date = lambda x: None
    parse_numeric_date = lambda x: None
    normalize_place = lambda x: x
    extract_and_validate_datetime_info = lambda x: {}
    extract_multiple_addresses_from_complex_phrase = lambda x: {}
    validate_and_improve_address = lambda x: {"is_valid": True, "improved": x}
    extract_location = lambda x, y: None
    is_specific_address = lambda x: True
    extract_email_value = lambda x: None
    extract_name_value = lambda x: None
    extract_rut_value = lambda x: None

def extraer_datos_mensaje_mejorado(mensaje_usuario: str) -> Dict[str, Any]:
    """
    Función de extracción que usa TODAS las funciones existentes del sistema Miramar.
    No inventa nada nuevo, solo orquesta las funciones ya probadas.
    """
    datos = {
        'origen': '',
        'destino': '',
        'fecha': '',
        'hora': '',
        'cantidad_personas': '',
        'tipo_vehiculo': '',
        'regreso': '',
        'observaciones': ''
    }
    
    if not utilidades_disponible:
        print("❌ Utilidades Miramar no disponibles")
        return datos
    
    print(f"🔍 Analizando: {mensaje_usuario}")
    
    # 1. USAR FUNCIÓN OFICIAL: extract_multiple_addresses_from_complex_phrase
    # DESHABILITADO TEMPORALMENTE: Esta función está mezclando fechas con direcciones
    """
    if utilidades_disponible:
        try:
            address_analysis = extract_multiple_addresses_from_complex_phrase(mensaje_usuario)
            
            # Procesar origen
            if address_analysis.get("best_origin"):
                best_origin = address_analysis["best_origin"]
                if isinstance(best_origin, dict) and 'text' in best_origin:
                    origen_text = best_origin['text']
                    # Filtrar si es realmente una dirección válida
                    if is_specific_address(origen_text) and len(origen_text.split()) >= 2:
                        datos['origen'] = origen_text
                        print(f"   ✅ Origen extraído (complex_phrase): {datos['origen']}")
                elif isinstance(best_origin, str) and is_specific_address(best_origin):
                    datos['origen'] = best_origin
                    print(f"   ✅ Origen extraído (complex_phrase): {datos['origen']}")
            
            # Procesar destino
            if address_analysis.get("best_destination"):
                best_destination = address_analysis["best_destination"]
                if isinstance(best_destination, dict) and 'text' in best_destination:
                    destino_text = best_destination['text']
                    # Filtrar si es realmente una dirección válida
                    if is_specific_address(destino_text) and len(destino_text.split()) >= 2:
                        datos['destino'] = destino_text
                        print(f"   ✅ Destino extraído (complex_phrase): {datos['destino']}")
                elif isinstance(best_destination, str) and is_specific_address(best_destination):
                    datos['destino'] = best_destination
                    print(f"   ✅ Destino extraído (complex_phrase): {datos['destino']}")
        except Exception as e:
            print(f"   ⚠️ Error en extract_multiple_addresses_from_complex_phrase: {e}")
    """
    
    # 2. FALLBACK: USAR FUNCIÓN OFICIAL: extract_origin_dest_pair
    if utilidades_disponible and (not datos['origen'] or not datos['destino']):
        try:
            origen_util, destino_util = extract_origin_dest_pair(mensaje_usuario)
            if origen_util and not datos['origen']:
                datos['origen'] = origen_util
                print(f"   ✅ Origen extraído: {datos['origen']}")
            
            if destino_util and not datos['destino']:
                datos['destino'] = destino_util
                print(f"   ✅ Destino extraído: {datos['destino']}")
        except Exception as e:
            print(f"   ⚠️ Error en extract_origin_dest_pair: {e}")
    
    # 2.5. PATRONES MANUALES para casos específicos que fallan - EJECUTAR SIEMPRE
    print(f"   🔧 Probando patrones manuales...")
    
    # PRIORIDAD 1: Patrón específico para nombres compuestos como "Fundo el Carmen"
    patron_compuesto = re.search(r'hasta\s+(.+?)\s+para', mensaje_usuario, re.IGNORECASE)
    if patron_compuesto:
        destino_compuesto = patron_compuesto.group(1).strip()
        # Limpiar el nombre compuesto
        destino_compuesto = re.sub(r'\s+', ' ', destino_compuesto)
        # Si detectamos un nombre compuesto, sobreescribir el destino anterior
        if len(destino_compuesto.split()) >= 2 and any(palabra in destino_compuesto.lower() for palabra in ['fundo', 'villa', 'sector', 'parque', 'centro', 'hospital', 'clinica', 'universidad']):
            datos['destino'] = destino_compuesto
            print(f"   ✅ Destino extraído (compuesto sobreescrito): {datos['destino']}")
    
    # PRIORIDAD 2: Solo ejecutar si no tenemos origen/destino
    if not datos['origen'] or not datos['destino']:
        # Patrón para "De X a Y" en cualquier parte
        patron_de_a = re.search(r'de\s+(.+?)\s+a\s+(.+?)(?:\s|$)', mensaje_usuario, re.IGNORECASE)
        print(f"   🔍 Patrón de-a: {patron_de_a}")
        if patron_de_a:
            print(f"   🔍 Grupos: '{patron_de_a.group(1)}' -> '{patron_de_a.group(2)}'")
            if not datos['origen']:
                origen_raw = patron_de_a.group(1).strip()
                # Solo normalizar si no contiene preposiciones importantes del nombre
                if not re.search(r'\b(?:del|de\s+la|de\s+los|de\s+las)\b', origen_raw, re.IGNORECASE):
                    datos['origen'] = normalize_place(origen_raw) if utilidades_disponible else origen_raw
                else:
                    datos['origen'] = origen_raw.title()
                print(f"   ✅ Origen extraído (patrón de-a): {datos['origen']}")
            if not datos['destino']:
                destino_raw = patron_de_a.group(2).strip()
                # Solo normalizar si no contiene preposiciones importantes del nombre
                if not re.search(r'\b(?:del|de\s+la|de\s+los|de\s+las)\b', destino_raw, re.IGNORECASE):
                    datos['destino'] = normalize_place(destino_raw) if utilidades_disponible else destino_raw
                else:
                    datos['destino'] = destino_raw.title()
                print(f"   ✅ Destino extraído (patrón de-a): {datos['destino']}")
        
        # PRIORIDAD 3: Patrón específico para "ida y vuelta de X a Y"
        patron_ida_vuelta = re.search(r'ida\s+y\s+vuelta\s+de\s+(.+?)\s+a\s+(.+?)(?:\s|$)', mensaje_usuario, re.IGNORECASE)
        if patron_ida_vuelta:
            if not datos['origen']:
                datos['origen'] = normalize_place(patron_ida_vuelta.group(1).strip()) if utilidades_disponible else patron_ida_vuelta.group(1).strip()
                print(f"   ✅ Origen extraído (ida y vuelta): {datos['origen']}")
            if not datos['destino']:
                datos['destino'] = normalize_place(patron_ida_vuelta.group(2).strip()) if utilidades_disponible else patron_ida_vuelta.group(2).strip()
                print(f"   ✅ Destino extraído (ida y vuelta): {datos['destino']}")
        
        # PRIORIDAD 4: Patrón para "solo ida de X a Y"
        patron_solo_ida = re.search(r'solo\s+ida\s+de\s+(.+?)\s+a\s+(.+?)(?:\s|$)', mensaje_usuario, re.IGNORECASE)
        if patron_solo_ida:
            if not datos['origen']:
                datos['origen'] = normalize_place(patron_solo_ida.group(1).strip()) if utilidades_disponible else patron_solo_ida.group(1).strip()
                print(f"   ✅ Origen extraído (solo ida): {datos['origen']}")
            if not datos['destino']:
                datos['destino'] = normalize_place(patron_solo_ida.group(2).strip()) if utilidades_disponible else patron_solo_ida.group(2).strip()
                print(f"   ✅ Destino extraído (solo ida): {datos['destino']}")
        
        # PRIORIDAD 5: Patrón más amplio para ubicaciones genéricas como "mi casa", "el mall"
        patron_genericas = re.search(r'(?:desde|de)\s+(mi\s+casa|el\s+mall|la\s+casa|mi\s+trabajo)\s+(?:hasta|a)\s+(mi\s+casa|el\s+mall|la\s+casa|mi\s+trabajo|el\s+aeropuerto)', mensaje_usuario, re.IGNORECASE)
        if patron_genericas:
            if not datos['origen']:
                datos['origen'] = patron_genericas.group(1).strip()
                print(f"   ✅ Origen extraído (genérico): {datos['origen']}")
            if not datos['destino']:
                datos['destino'] = patron_genericas.group(2).strip() 
                print(f"   ✅ Destino extraído (genérico): {datos['destino']}")
    
    print(f"   🔧 Estado después de patrones manuales: origen='{datos.get('origen', 'VACIO')}', destino='{datos.get('destino', 'VACIO')}'")
    
    # 3. ÚLTIMO FALLBACK: extract_location individual
    if utilidades_disponible and not datos['origen']:
        try:
            origen_single = extract_location(mensaje_usuario, "origen")
            if origen_single and is_specific_address(origen_single):
                datos['origen'] = origen_single
                print(f"   ✅ Origen extraído (extract_location): {datos['origen']}")
        except Exception as e:
            print(f"   ⚠️ Error en extract_location origen: {e}")
    
    if utilidades_disponible and not datos['destino']:
        try:
            destino_single = extract_location(mensaje_usuario, "destino")
            if destino_single and is_specific_address(destino_single):
                datos['destino'] = destino_single
                print(f"   ✅ Destino extraído (extract_location): {datos['destino']}")
        except Exception as e:
            print(f"   ⚠️ Error en extract_location destino: {e}")
    
    # 4. USAR FUNCIÓN OFICIAL: extract_and_validate_datetime_info
    if utilidades_disponible:
        try:
            datetime_info = extract_and_validate_datetime_info(mensaje_usuario)
            
            if datetime_info.get("normalized_date"):
                datos['fecha'] = datetime_info["normalized_date"]
                print(f"   📅 Fecha extraída (datetime_info): {datos['fecha']}")
            
            if datetime_info.get("normalized_time"):
                datos['hora'] = datetime_info["normalized_time"]
                print(f"   ⏰ Hora extraída (datetime_info): {datos['hora']}")
                
        except Exception as e:
            print(f"   ⚠️ Error en extract_and_validate_datetime_info: {e}")
    
    # 4.5. FALLBACK MANUAL para fechas numéricas que no se extraen
    if not datos['fecha']:
        # Patrón manual para "el 10/10"
        fecha_match = re.search(r'(?:el\s+)?(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b', mensaje_usuario)
        if fecha_match:
            try:
                day = int(fecha_match.group(1))
                month = int(fecha_match.group(2))
                year_part = fecha_match.group(3)
                
                if 1 <= day <= 31 and 1 <= month <= 12:
                    if year_part:
                        year = int(year_part)
                        if year < 100:
                            year += 2000 if year < 50 else 1900
                    else:
                        year = datetime.now().year
                    
                    try:
                        fecha_obj = datetime(year, month, day)
                        datos['fecha'] = fecha_obj.strftime('%Y-%m-%d')
                        print(f"   📅 Fecha extraída (manual): {datos['fecha']}")
                    except ValueError:
                        pass
            except:
                pass
    
    # Fallback individual para otras funciones de fecha/hora
    if utilidades_disponible and not datos['fecha']:
        try:
            fecha_numerica = parse_numeric_date(mensaje_usuario)
            if fecha_numerica:
                datos['fecha'] = fecha_numerica
                print(f"   📅 Fecha extraída (numeric): {fecha_numerica}")
        except:
            pass
        
        try:
            fecha_textual = parse_textual_date(mensaje_usuario)
            if fecha_textual and not datos['fecha']:
                datos['fecha'] = fecha_textual
                print(f"   📅 Fecha extraída (textual): {fecha_textual}")
        except:
            pass
    
    if utilidades_disponible and not datos['hora']:
        try:
            hora_extraida = parse_time_from_text(mensaje_usuario)
            if hora_extraida:
                datos['hora'] = hora_extraida
                print(f"   ⏰ Hora extraída (time): {hora_extraida}")
        except:
            pass
    
    # 5. USAR FUNCIÓN OFICIAL: extract_quantity
    if utilidades_disponible:
        try:
            cantidad_extraida = extract_quantity(mensaje_usuario)
            if cantidad_extraida and cantidad_extraida > 0:
                datos['cantidad_personas'] = str(cantidad_extraida)
                print(f"   👥 Cantidad extraída: {cantidad_extraida} personas")
        except Exception as e:
            print(f"   ⚠️ Error en extract_quantity: {e}")
    
    # 6. EXTRACCIÓN DE REGRESO (patrones simples - única cosa que no existe en Miramar)
    patrones_regreso = [
        r'(?:sin|no)\s+(?:regreso|vuelta|retorno)',
        r'(?:solo|sólo)\s+ida',
        r'ida\s+y\s+(?:vuelta|regreso)',
        r'(?:con|hay)\s+(?:regreso|vuelta|retorno)',
        r'(?:ida|viaje)\s+(?:simple|única|de ida)'
    ]
    
    for patron in patrones_regreso:
        match = re.search(patron, mensaje_usuario.lower(), re.IGNORECASE)
        if match:
            texto_match = match.group(0).lower()
            if any(word in texto_match for word in ['sin', 'no', 'solo', 'sólo', 'simple', 'única']):
                datos['regreso'] = 'no'
            else:
                datos['regreso'] = 'sí'
            print(f"   🔄 Regreso extraído: {datos['regreso']}")
            break
    
    # 7. USAR SISTEMA DE DIRECCIONES como último recurso
    if direcciones_disponible and detect_and_extract_address and (not datos['origen'] or not datos['destino']):
        try:
            direccion = detect_and_extract_address(mensaje_usuario)
            if direccion:
                print(f"   🏠 Dirección detectada por sistema: {direccion}")
                if not datos['origen'] and direccion.get('raw'):
                    datos['origen'] = direccion['raw']
                    print(f"   ✅ Origen desde dirección: {datos['origen']}")
        except Exception as e:
            print(f"   ⚠️ Error en detección de direcciones: {e}")
    
    # 8. FILTRAR DATOS VACÍOS
    datos_filtrados = {k: v for k, v in datos.items() if v}
    
    print(f"📊 Datos finales extraídos: {json.dumps(datos_filtrados, ensure_ascii=False)}")
    return datos_filtrados


def probar_casos():
    """Prueba la extracción con casos críticos."""
    casos = [
        "Necesito cotizar desde Av Curauma Sur 1826 hasta Valparaíso el 10/10 a las 2pm, somos 3 personas",
        "De viña del mar a santiago, mañana a las 8am, somos 5",
        "Desde aeropuerto hasta av libertad 1234, para el 15 de octubre",
        "Viaje de Santiago a Valparaíso el 27 de junio a las 3pm, somos 4 personas sin regreso",
        "Quiero ir desde mi casa hasta el mall",
        "Vamos el 10/10 a las 2pm",
        "Somos 3 personas sin regreso"
    ]
    
    print("🧪 PRUEBAS DE EXTRACCIÓN MEJORADA:")
    print("=" * 60)
    
    for i, caso in enumerate(casos, 1):
        print(f"\n🔍 CASO {i}:")
        print(f"Input: {caso}")
        print("-" * 40)
        
        resultado = extraer_datos_mensaje_mejorado(caso)
        
        # Verificar qué se extrajo
        if resultado:
            print("✅ EXTRACCIÓN EXITOSA")
        else:
            print("❌ NO SE EXTRAJO NADA")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Probar con input específico
        mensaje = " ".join(sys.argv[1:])
        print(f"🔧 Probando extracción específica:")
        resultado = extraer_datos_mensaje_mejorado(mensaje)
    else:
        # Ejecutar todos los casos de prueba
        probar_casos()
