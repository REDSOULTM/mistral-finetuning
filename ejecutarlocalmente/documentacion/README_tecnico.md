# README técnico – Asistente de cotizaciones Miramar

Este documento describe, módulo por módulo, las funciones y clases que componen el chatbot. Sirve como referencia para mantenimiento o extensiones futuras.

## Ejecutable

### `ejecutar_mistral.py`
- Importa `main` desde `miramar_bot.ejecucion` y lo ejecuta. Es el punto de entrada en consola.

## Paquete `miramar_bot`

### `configuracion.py`
- Define rutas (`BASE_MODEL_DIR`, `LORA_ADAPTER_DIR`, `COTIZACIONES_FILE`) y banderas (`ENABLE_PERSONAL_REGISTRATION`).
- Expone las constantes vía `__all__` para que otros módulos las importen sin repetir lógica.

### `cargar_modelo.py`
- `load_model_and_tokenizer()`: carga el modelo base Mistral y el adaptador LoRA usando Unsloth. Garantiza que el tokenizador tenga `pad_token` y devuelve `(model, tokenizer, device)` o `(None, None, device)` si falla.

### `constantes.py`
- Colección de listas, conjuntos y expresiones regulares reutilizadas:
  - `QUESTION_VARIATION_HINTS`, `FALLBACK_FIELD_PROMPTS`: textos alternativos para las preguntas del bot.
  - Conjuntos para identificar direcciones, palabras de regreso, reconocimientos, etc.
  - Patrones precompilados para reconocer estructuras como "desde X hasta Y" o frases con fechas.
  - `REGRESO_*_PATTERNS`: variantes de expresiones “solo ida” / “con regreso”.

### `utilidades.py`
Funciones puras que limpian o interpretan textos.
- `titlecase_clean`, `strip_accents`, `sanitize_print`: normalización básica.
- `clean_location_noise`, `normalize_place`, `is_specific_address`: procesan direcciones escritas por el usuario.
- `normalize_regreso`, `normalize_fecha`, `parse_time_from_text`: estandarizan respuestas de regreso, fechas y horas.
- `extract_email_value`, `extract_rut_value`, `extract_name_value`: extraen datos personales cuando está activado el registro.
- `extract_quantity`: obtiene la cantidad de pasajeros, incluyendo heurísticas para frases como “yo y mi pareja”.
- `extract_origin_dest_pair`, `extract_location`: identifican origen/destino en frases libres.
- `build_display_state`, `print_bot_turn`: formatean el estado y las respuestas en la consola.
- `build_fallback_question`, `select_question_fields`, `question_mentions_forbidden`: apoyo para elegir y validar preguntas.

### `direcciones.py`
- `detect_and_extract_address(text)`: usa libpostal + Nominatim para enriquecer direcciones completas (calle, número, comuna, coordenadas, enlace a mapa). Solo se invoca cuando ya hay datos precisos en el estado.

### `cotizacion.py`
- `guardar_cotizacion(state)`: escribe la cotización en `cotizaciones.json`, usando un archivo temporal para evitar corrupciones.

### `vendedor.py`
- Clase `MiramarSellerBot` encapsula todo el flujo conversacional.
  - `__init__`, `reset`: inicializan estado y banderas internas.
  - `initial_prompt()`: genera el saludo inicial usando `generate_greeting_llm()`.
  - `_generate_tagged_message(...)`: invoca al LLM y filtra los resultados según validadores.
  - `validate_input_llm(user_input)`: pide al LLM un JSON con los campos detectados; si falla, vuelve un diccionario vacío.
  - `_fallback_extract_fields(user_input, existing)`: heurísticas adicionales para rellenar campos faltantes (direcciones, fechas, horas, regreso, cantidad).
  - `update_state(user_input)`: núcleo del bot; combina datos nuevos, confirma cambios, solicita información pendiente y dispara el guardado final.
  - `get_next_question()`: decide qué preguntar según lo que falta; incluye un camino específico para preguntar “regreso + cantidad” en una frase clara.
  - `run_step(user_input)`: procesa un turno de usuario; coordina `update_state`, preguntas, confirmaciones y cierre.
  - Generadores de mensajes (`generate_*`): saludos, preguntas, confirmaciones, agradecimientos, pedidos de dirección, cierre, etc. Todos llaman al LLM y se respaldan en mensajes de emergencia si la salida no cumple los validadores.

### `ejecucion.py`
- `main()`: inicia el modelo, crea la instancia de `MiramarSellerBot` y maneja el loop de consola. Espera el primer mensaje del cliente, emite el saludo, y pasa cada entrada a `run_step`. Controla interrupciones (`EOF`, `Ctrl+C`) y despedidas.

### `nucleo.py`
- Reexporta las piezas clave (`MiramarSellerBot`, `load_model_and_tokenizer`, `guardar_cotizacion`, `extract_structured_pairs`, `sanitize_print`, `STATE_FIELDS`, `main`) para que otras partes del proyecto las importen con una sola declaración.

## Archivos adicionales
- `cotizaciones.json`: archivo JSON donde se van acumulando las cotizaciones guardadas.
- `ejecuta_mistral_chat_original_del_25_de_septiembre.py`: versión histórica previa a la modularización (solo referencia).

Con esta guía técnica puedes rastrear rápidamente dónde se implementa cada comportamiento del bot y qué función modificar si necesitas personalizarlo.
