"""Constantes y patrones regex que reutiliza el bot Miramar."""

from __future__ import annotations

import re
from typing import Dict, List, Pattern, Set

from .configuracion import COMMENT_FIELD, PERSONAL_FIELDS, TRIP_FIELDS

NEGATIVE_COMMENT_VALUES: Set[str] = {
    "no",
    "ninguno",
    "ninguna",
    "no tengo",
    "no tenemos",
    "sin comentarios",
    "sin comentario",
    "ningún comentario",
    "ningun comentario",
    "sin requerimiento",
    "sin requerimientos",
    "sin requerimiento especial",
    "sin requerimientos especiales",
}

QUESTION_VARIATION_HINTS: List[str] = [
    "reformula usando sinónimos y menciona primero el origen y luego el destino",
    "usa un estilo directo empezando por 'Necesito saber...'",
    "incluye un guiño a que es para preparar la cotización",
    "emplea la estructura '¿Podrías confirmarme ...?' con terminología distinta",
    "haz la pregunta en dos cláusulas separadas por 'y', evitando repetir palabras previas",
]

FALLBACK_FIELD_PROMPTS: Dict[str, List[str]] = {
    "nombre": [
        "¿Podrías indicarme tu nombre completo?",
        "¿Con qué nombre te registramos?",
        "¿Cuál es tu nombre?",
    ],
    "rut": [
        "¿Me compartes tu RUT?",
        "¿Cuál es tu RUT?",
        "¿Podrías confirmarme tu RUT?",
    ],
    "correo": [
        "¿Cuál es tu correo electrónico?",
        "¿Me facilitas tu correo electrónico?",
        "¿Podrías indicarme un correo de contacto?",
    ],
    "origen": [
        "¿Con qué dirección partimos el viaje?",
        "¿Cuál sería el punto exacto de origen?",
        "¿Desde qué lugar deberíamos pasar a buscarte?",
    ],
    "destino": [
        "¿Hacia dónde debemos llevarte?",
        "¿Cuál será el destino final del traslado?",
        "¿A qué dirección debemos llegar?",
    ],
    "origen_destino": [
        "¿Desde qué dirección inicia el viaje y cuál es el destino?",
        "¿Me confirmas punto de partida y destino final?",
        "¿Desde dónde salimos y hacia qué dirección vamos?",
    ],
    "fecha": [
        "¿Para qué fecha deseas programar el viaje?",
        "¿En qué día necesitas el traslado?",
        "¿Qué fecha te acomoda para la salida?",
    ],
    "hora": [
        "¿A qué hora deberíamos comenzar el servicio?",
        "¿Cuál es la hora de salida que prefieres?",
        "¿Qué horario te acomoda para partir?",
    ],
    "regreso": [
        "¿El viaje es solo ida o también regreso?",
        "¿Necesitas que coordinemos un regreso?",
        "¿Será un servicio de ida solamente o ida y vuelta?",
    ],
    "cantidad": [
        "¿Cuántas personas viajarán?",
        "¿Cuántos pasajeros debemos considerar?",
        "¿Para cuántos ocupantes organizamos el traslado?",
    ],
    COMMENT_FIELD: [
        "¿Hay algún comentario o requerimiento adicional que debamos saber?",
        "¿Quieres agregar alguna indicación especial?",
        "¿Tienes algún detalle extra que debamos considerar?",
    ],
}

LOCATION_EXCLUDE: Set[str] = {
    "sin regreso",
    "solo ida",
    "ida",
    "ida y vuelta",
    "ida y regreso",
    "no",
    "ninguno",
    "ninguna",
    "sin vuelta",
    "sin retorno",
    "no regreso",
}

PLACE_STOP = r"(?:,|\s+a\s+las\b|\s+con\b|\s+somos\b|\s+vamos\b|$)"
PLACE_TIGHT = r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\.\-'\s]{2,}?"

NEGATIVE_COTIZ_RE: Pattern[str] = re.compile(
    r"\b(?:(?:prefiero|por\s+ahora|de\s+momento|por\s+el\s+momento)\s+)?"
    r"no(?:\s+(?:quiero|deseo|busco|necesito))?\s+"
    r"(?:cotiz(?:ar|aci[óo]n|aciones)?|presupuesto|viaje)\b",
    re.IGNORECASE,
)

NEGATIVE_TRAVEL_INTENT_RE: Pattern[str] = re.compile(
    r"\b(?:no|prefiero\s+no|por\s+ahora\s+no|por\s+el\s+momento\s+no|no\s+deseo|no\s+quiero|no\s+busco|no\s+necesito)\b.*"
    r"\b(?:viaj|traslad|transport|cotiz|presupuest)\w*",
    re.IGNORECASE,
)

ADDRESS_KEYWORDS: Set[str] = {
    "calle",
    "avenida",
    "av.",
    "av ",
    "pasaje",
    "psje",
    "ruta",
    "camino",
    "carretera",
    "km",
    "kilometro",
    "kilómetro",
    "kilometros",
    "terminal",
    "aeropuerto",
    "hotel",
    "hostal",
    "plaza",
    "feria",
    "paradero",
    "estacion",
    "estación",
    "condominio",
    "villa",
    "poblacion",
    "población",
    "sector",
    "barrio",
    "parcela",
    "lote",
    "edificio",
    "hospital",
    "colegio",
    "campus",
    "galpón",
    "galpon",
}

ADDRESS_DIRECTION_KEYWORDS: Set[str] = {
    "norte",
    "sur",
    "este",
    "oeste",
    "oriente",
    "poniente",
    "nororiente",
    "norponiente",
    "suroeste",
    "sureste",
    "centro",
}

ADDRESS_KEYWORD_SET: Set[str] = ADDRESS_KEYWORDS.union(ADDRESS_DIRECTION_KEYWORDS)

ORIGIN_CHANGE_TOKENS: Set[str] = {
    "origen",
    "desde",
    "salgo de",
    "salimos de",
    "parto de",
    "partimos de",
    "punto de partida",
    "nos recogen en",
    "recojo en",
}

DESTINATION_CHANGE_TOKENS: Set[str] = {
    "destino",
    "hasta",
    "hacia",
    "llegar a",
    "vamos a",
    "vamos hasta",
    "nos llevan a",
    "punto de destino",
}

COMMENT_FIELD_PATTERN = re.escape(COMMENT_FIELD).replace(r"\ ", r"\s+")
_STRUCTURED_KEYS = [
    re.escape(key).replace(r"\ ", r"\s+")
    for key in PERSONAL_FIELDS + TRIP_FIELDS + [COMMENT_FIELD]
]
STRUCTURED_KEY_PATTERN: Pattern[str] = re.compile(
    rf"\b({'|'.join(_STRUCTURED_KEYS)})\s*[:=]\s*([^,;\n]+)",
    re.IGNORECASE,
)

ORIG_DEST_PATTERNS = [
    re.compile(
        rf"(?:desde|parto desde|salgo de|partida de|salida de)\s+(?P<orig>{PLACE_TIGHT})(?={PLACE_STOP}).*?(?:hacia|hasta|a)\s+(?P<dest>{PLACE_TIGHT})(?={PLACE_STOP})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:vamos|voy|iremos|iré|queremos ir|quiero ir|queremos viajar|quiero viajar|viajamos)\s+desde\s+(?P<orig>{PLACE_TIGHT})(?={PLACE_STOP}).*?(?:a|hacia|hasta)\s+(?P<dest>{PLACE_TIGHT})(?={PLACE_STOP})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<orig>{PLACE_TIGHT})\s+(?:hasta|a|hacia)\s+(?P<dest>{PLACE_TIGHT})(?={PLACE_STOP})",
        re.IGNORECASE,
    ),
]

ORIGIN_EXPLICIT_RE = re.compile(
    rf"(?:el\s+)?origen\s+(?:es(?:\s+ahora)?|ser[aá](?:\s+ahora)?|seria|sera|queda(?:\s+en)?|quedara|quedará)?\s*(?P<orig>{PLACE_TIGHT})(?={PLACE_STOP})",
    re.IGNORECASE,
)

DESTINATION_EXPLICIT_RE = re.compile(
    rf"(?:el\s+)?destino\s+(?:es(?:\s+ahora)?|ser[aá](?:\s+ahora)?|seria|sera|queda(?:\s+en)?|quedara|quedará)?\s*(?P<dest>{PLACE_TIGHT})(?={PLACE_STOP})",
    re.IGNORECASE,
)

SPANISH_NUMBER_WORDS = {
    "un": 1,
    "una": 1,
    "uno": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
}

_NUM_WORD_PATTERN = "|".join(sorted(SPANISH_NUMBER_WORDS.keys(), key=len, reverse=True))

WORD_CONTEXT_PATTERNS = [
    re.compile(
        rf"(?:somos|vamos|seremos|seran|viajamos|viajan|van|iremos|iran)\s+({_NUM_WORD_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b({_NUM_WORD_PATTERN})\b\s*(?:personas?|pasajeros?|viajeros?|clientes|ocupantes)",
        re.IGNORECASE,
    ),
]

WORD_DIGIT_PATTERN = re.compile(
    r"(?:somos|vamos|seremos|seran|serán|viajamos|viajan|van|iremos|iran|irán)\s+(\d{1,3})\b",
    re.IGNORECASE,
)

QUANTITY_STANDALONE_DIGIT_RE = re.compile(r"^\s*(\d{1,3})\s*$")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RUT_RE = re.compile(r"\b\d{1,2}\.?(?:\d{3})\.?(?:\d{3})-?[\dkK]\b")

NAME_STOPWORDS = {
    "hola",
    "buenos",
    "buenas",
    "quiero",
    "quisiera",
    "necesito",
    "necesitamos",
    "traslado",
    "viaje",
    "viajar",
    "transporte",
    "cotizar",
    "cotizacion",
    "cotización",
    "presupuesto",
    "consulta",
    "desde",
    "hasta",
    "destino",
    "origen",
    "somos",
    "vamos",
    "ir",
    "iria",
    "iremos",
}

LOCATION_WORD = r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\.\-'\s]+"
ADDR_TOKEN = r"(calle|av(?:\.\s*|(?:enida))?|pasaje|psje|camino|ruta|carretera|km|kil[oó]metro|plaza|terminal|aeropuerto|condominio|villa|poblaci[oó]n|sector|parcela|lote|edificio)"
ADDR_CORE_RE = re.compile(
    rf"\b(?:{ADDR_TOKEN})\b.*?\b(\d{{1,5}}[A-Za-z]?)?\b",
    re.IGNORECASE,
)

MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
    "ene": 1,
    "feb": 2,
    "mar": 3,
    "abr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "sep": 9,
    "set": 9,
    "oct": 10,
    "nov": 11,
    "dic": 12,
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "aug": 8,
    "sept": 9,
    "dec": 12,
}

WEEKDAYS = {
    "lunes": 0,
    "martes": 1,
    "miércoles": 2,
    "miercoles": 2,
    "jueves": 3,
    "viernes": 4,
    "sábado": 5,
    "sabado": 5,
    "domingo": 6,
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

MONTH_PATTERN = "|".join(
    sorted({re.escape(k) for k in MONTHS.keys()}, key=len, reverse=True)
)

DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(rf"\b\d{{1,2}}\s+de\s+({MONTH_PATTERN})\b", re.IGNORECASE),
    re.compile(rf"\b({MONTH_PATTERN})\s+\d{{1,2}}\b", re.IGNORECASE),
]

DATE_KEYWORDS = {
    "hoy",
    "mañana",
    "pasado mañana",
    "pasado-manana",
    "proximo",
    "próximo",
    "este",
}

MONTH_TOKEN_RE = re.compile(rf"\b({MONTH_PATTERN})\b", re.IGNORECASE)
DATE_PHRASE_RE = re.compile(rf"\bel\s+\d{{1,2}}\s+de\s+({MONTH_PATTERN})\b", re.IGNORECASE)

TIME_PATTERNS = [
    re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm|hrs?\.?|horas?|h)?\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s*(?:am|pm)\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s*(?:hrs?\.?|horas?)\b", re.IGNORECASE),
    re.compile(r"\b(?:medianoche|mediod[ií]a|mediodia)\b", re.IGNORECASE),
]

TIME_HINT_WORDS = {
    "hora",
    "horas",
    "hr",
    "hrs",
    "am",
    "pm",
    "salir",
    "salgo",
    "sale",
    "salimos",
    "salida",
    "partida",
    "partir",
}

REGRESO_NO_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bsin\s+regreso\b",
        r"\bsin\s+vuelta\b",
        r"\bsin\s+retorno\b",
        r"\bsolo\s+ida\b",
        r"\bsolo\s+de\s+ida\b",
        r"\bida\s+solamente\b",
        r"\bno\s+regreso\b",
        r"\bno\b[^,.;\n]*\bregreso\b",
    ]
]

REGRESO_YES_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bcon\s+regreso\b",
        r"\bida\s+y\s+vuelta\b",
        r"\bida\s+y\s+regreso\b",
        r"\bcon\s+retorno\b",
        r"\bcon\s+vuelta\b",
        r"\bcon\s+regreso\s+incluido\b",
        r"\bincluye\s+regreso\b",
        r"\bincluye\s+retorno\b",
        r"\bincluye\s+ida\s+y\s+vuelta\b",
        r"\bincluye\s+ida\s+y\s+regreso\b",
    ]
]

__all__ = [
    "NEGATIVE_COMMENT_VALUES",
    "QUESTION_VARIATION_HINTS",
    "FALLBACK_FIELD_PROMPTS",
    "LOCATION_EXCLUDE",
    "NEGATIVE_COTIZ_RE",
    "NEGATIVE_TRAVEL_INTENT_RE",
    "ADDRESS_KEYWORDS",
    "ADDRESS_DIRECTION_KEYWORDS",
    "ADDRESS_KEYWORD_SET",
    "ORIGIN_CHANGE_TOKENS",
    "DESTINATION_CHANGE_TOKENS",
    "COMMENT_FIELD_PATTERN",
    "STRUCTURED_KEY_PATTERN",
    "SPANISH_NUMBER_WORDS",
    "WORD_CONTEXT_PATTERNS",
    "WORD_DIGIT_PATTERN",
    "QUANTITY_STANDALONE_DIGIT_RE",
    "EMAIL_RE",
    "RUT_RE",
    "NAME_STOPWORDS",
    "LOCATION_WORD",
    "ADDR_TOKEN",
    "ADDR_CORE_RE",
    "PLACE_STOP",
    "PLACE_TIGHT",
    "ORIG_DEST_PATTERNS",
    "ORIGIN_EXPLICIT_RE",
    "DESTINATION_EXPLICIT_RE",
    "MONTHS",
    "WEEKDAYS",
    "MONTH_PATTERN",
    "DATE_PATTERNS",
    "DATE_KEYWORDS",
    "MONTH_TOKEN_RE",
    "DATE_PHRASE_RE",
    "TIME_PATTERNS",
    "TIME_HINT_WORDS",
    "REGRESO_NO_PATTERNS",
    "REGRESO_YES_PATTERNS",
]
