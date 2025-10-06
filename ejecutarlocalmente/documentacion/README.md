# Asistente de Cotizaciones – Guía para usuarios

Este directorio contiene el chatbot que Transportes Miramar usa para preparar cotizaciones de traslados. Aquí encontrarás todo lo necesario para ejecutarlo desde la consola y entender, en palabras simples, qué hace cada archivo.

## ¿Qué hace el sistema?

- Conversa con el cliente (como en WhatsApp) para pedir origen, destino, fecha, hora, si el viaje es ida o ida/vuelta y cuántas personas viajarán.
- Revisa cada mensaje, valida los datos y, cuando está toda la información, guarda la solicitud en un archivo `cotizaciones.json`.
- Funciona incluso si el modelo de lenguaje no se puede cargar, aunque en ese caso las respuestas serán más básicas.

## Archivos que te interesan

- `ejecutar_mistral.py`: ¡El que ejecutas! Abre la consola, escribe los mensajes del cliente y muestra las respuestas del bot.
- Carpeta `miramar_bot/` (el cerebro del asistente):
  - `configuracion.py`: define rutas y opciones (por ejemplo, dónde están el modelo y el adaptador LoRA).
  - `cargar_modelo.py`: enciende el modelo de lenguaje y el adaptador LoRA usando Unsloth.
  - `constantes.py`: lista de palabras clave y expresiones regulares que ayudan a reconocer direcciones, fechas, números, etc.
  - `utilidades.py`: funciones de apoyo para limpiar textos, extraer direcciones, interpretar frases como “vamos yo y mi pareja” y formatear la conversación.
  - `direcciones.py`: conecta con libpostal y Nominatim (OpenStreetMap) para comprobar direcciones reales cuando ya se tienen calle y número.
  - `cotizacion.py`: guarda cada cotización en el archivo `cotizaciones.json` de forma segura.
  - `vendedor.py`: clase `MiramarSellerBot`; decide qué preguntar, cuándo confirmar datos y cuándo cerrar la conversación.
  - `ejecucion.py`: gestiona el bucle de consola; espera el primer mensaje del cliente, envía el saludo y pasa cada respuesta al bot.
  - `nucleo.py`: reexporta funciones y clases para que `ejecutar_mistral.py` las importe con una sola línea.

## Cómo ejecutar el bot (paso a paso)

1. Activa tu entorno virtual (si corresponde):
   ```bash
   source .venv/bin/activate
   ```
2. Entra a la carpeta del proyecto y ejecuta:
   ```bash
   python ejecutarlocalmente/ejecutar_mistral.py
   ```
3. La consola mostrará “Cliente:”. Escribe los mensajes como si fueras la persona que solicita la cotización. El bot responde después de cada línea.
4. Cuando el bot agradece y dice que revisará la solicitud, la cotización queda guardada.

## ¿Dónde se guarda la información?

- Todas las cotizaciones se añaden al archivo `ejecutarlocalmente/miramar_bot/cotizaciones.json`.
- Si se interrumpe la conversación, basta con ejecutar de nuevo `ejecutar_mistral.py`; el historial anterior se mantiene en el JSON.

## Ajustes rápidos

- **Rutas del modelo:** edita `miramar_bot/configuracion.py` si mueves el modelo base o el adaptador LoRA.
- **Tono de las preguntas:** modifica los textos en `miramar_bot/constantes.py` o los prompts en `miramar_bot/vendedor.py`.
- **Extras y validaciones:** revisa `miramar_bot/utilidades.py` para cambiar las reglas que interpretan fechas, horas o cantidades.

## Preguntas frecuentes

- **¿No tengo conexión a internet?** El bot sigue funcionando, pero no podrá validar direcciones con Nominatim.
- **¿Puedo borrar las cotizaciones?** Sí, elimina o vacía el archivo `cotizaciones.json` (haz una copia antes si necesitas conservarlas).
- **¿Se puede integrar con WhatsApp directamente?** Este script sólo cubre la lógica del bot. Para WhatsApp necesitarías un puente o API que envíe/reciba mensajes y llame a `ejecutar_mistral.py`.

Con esto tienes una visión clara de qué hace cada archivo y cómo ponerlo en marcha sin conocer detalles técnicos.
