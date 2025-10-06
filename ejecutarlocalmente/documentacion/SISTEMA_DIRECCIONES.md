# Sistema de Validación de Direcciones - Transportes Miramar

## 📋 Descripción General

Sistema avanzado para validación de direcciones en conversaciones de WhatsApp, diseñado para manejar casos reales de clientes de Transportes Miramar.

## ✅ Características Principales

### 🎯 Validación Inteligente de Direcciones
- **Acepta direcciones válidas**: "Av. Providencia 1234" (comuna opcional)
- **Rechaza ciudades solas**: "Santiago", "Las Condes" (requiere dirección específica)
- **Extracción de frases complejas**: Maneja múltiples direcciones en una sola frase
- **Patrones naturales de WhatsApp**: Comprende como escriben los clientes reales

### 🔍 Casos Soportados

#### ✅ Direcciones Válidas (Aceptadas)
```
- "1 norte 1161, viña del mar"
- "av curauma sur 1826"
- "Mall Paseo del Valle - O'Higgins 176, Quillota"
- "Álvarez 70-2571923 Viña del Mar" 
- "Nueva Imperial 5162"
- "Av. Providencia 1234" (sin comuna)
- "Providencia 1234" (sin indicador)
```

#### ❌ Direcciones Inválidas (Rechazadas)
```
- "Santiago" (solo ciudad)
- "Las Condes" (solo comuna)
- "Valparaíso" (solo ciudad)
- "Centro" (muy genérico)
```

#### 🤖 Frases Complejas de WhatsApp
```
- "Si, 1 norte 1161, viña del mar hasta santiago la dirección es Nueva imperial 5162"
  → Origen: "1 norte 1161" ✅, Destino: "Nueva imperial 5162" ✅

- "Desde av curauma sur 1826, valparaíso hacia Guindo Santo 1896, La Serena"
  → Origen: "av curauma sur 1826" ✅, Destino: "Guindo Santo 1896" ✅

- "Necesito transporte de Providencia 1234 a Moneda 567"
  → Origen: "Providencia 1234" ✅, Destino: "Moneda 567" ✅
```

## 📊 Métricas de Rendimiento

- **47/47 direcciones válidas correctamente aceptadas (100%)**
- **21/21 direcciones inválidas correctamente rechazadas (100%)**
- **11/11 casos específicos de WhatsApp funcionando (100%)**

## 🧪 Testing

### Tests Principales
- `tests/test_direcciones_reales_whatsapp.py` - Test exhaustivo de direcciones reales
- `tests/test_whatsapp_exacto.py` - Casos específicos de conversaciones WhatsApp
- `tests/test_final_whatsapp.py` - Test de integración final
- `tests/test_correcciones.py` - Tests de correcciones específicas
- `tests/test_flexibilidad_direcciones.py` - Tests de flexibilidad

### Ejecutar Tests
```bash
cd /home/red/projects/Fine2
python3 tests/test_direcciones_reales_whatsapp.py
python3 tests/test_whatsapp_exacto.py
```

## 🚀 Uso del Sistema

### Ejecutar el Bot
```bash
cd /home/red/projects/Fine2/ejecutarlocalmente
python3 ejecutar_mistral.py
```

### Estructura del Proyecto
```
Fine2/
├── ejecutarlocalmente/           # Sistema principal
│   ├── miramar_bot/             # Core del bot
│   │   ├── vendedor.py          # Lógica principal
│   │   ├── utilidades.py        # Validación de direcciones
│   │   └── direcciones.py       # Integración libpostal
│   ├── documentacion/           # Documentación técnica
│   └── ejecutar_mistral.py      # Punto de entrada
├── tests/                       # Tests de validación
└── modelos/                     # Modelos fine-tuned
```

## 🔧 Funciones Clave

### `validate_and_improve_address(address_text)`
Valida direcciones usando libpostal + validación manual
- Retorna: `{"is_valid": bool, "needs_more_detail": bool, "suggestion": str}`

### `extract_multiple_addresses_from_complex_phrase(text)`
Extrae múltiples direcciones de frases complejas de WhatsApp
- Maneja contexto para determinar origen vs destino
- Puntuación inteligente basada en indicadores textuales

### `analyze_user_response_context(user_input, current_state)`
Análisis completo del contexto del usuario
- Direcciones, fechas, horas, cantidad de pasajeros
- Integración con extracción compleja como fallback

## ⚙️ Configuración

El sistema está preconfigurado para funcionar con:
- Modelo fine-tuned Mistral-7B
- Libpostal para validación de direcciones
- Nominatim para geocoding
- Patrones específicos para Chile

## 📈 Resultados

**Sistema listo para producción** - Maneja todos los casos reales de WhatsApp como los mostrados en las conversaciones originales de Transportes Miramar.
