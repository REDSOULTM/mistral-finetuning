"""Conjunto de utilidades reutilizables por el bot de Transportes Miramar."""

from __future__ import annotations

import datetime
import json
import random
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .configuracion import COMMENT_FIELD, PERSONAL_FIELDS, STATE_FIELDS, TRIP_FIELDS
from .direcciones import detect_and_extract_address
from .constantes import (
    ADDRESS_KEYWORD_SET,
    ADDR_CORE_RE,
    COMMENT_FIELD_PATTERN,
    DATE_KEYWORDS,
    DATE_PATTERNS,
    DATE_PHRASE_RE,
    DESTINATION_CHANGE_TOKENS,
    DESTINATION_EXPLICIT_RE,
    FALLBACK_FIELD_PROMPTS,
    LOCATION_EXCLUDE,
    MONTHS,
    MONTH_PATTERN,
    MONTH_TOKEN_RE,
    NEGATIVE_COMMENT_VALUES,
    ORIG_DEST_PATTERNS,
    ORIGIN_CHANGE_TOKENS,
    ORIGIN_EXPLICIT_RE,
    PLACE_STOP,
    PLACE_TIGHT,
    REGRESO_NO_PATTERNS,
    REGRESO_YES_PATTERNS,
    RUT_RE,
    NAME_STOPWORDS,
    SPANISH_NUMBER_WORDS,
    STRUCTURED_KEY_PATTERN,
    TIME_HINT_WORDS,
    TIME_PATTERNS,
    WORD_CONTEXT_PATTERNS,
    WORD_DIGIT_PATTERN,
    QUANTITY_STANDALONE_DIGIT_RE,
    EMAIL_RE,
)


def titlecase_clean(text: str) -> str:
    text = text.strip().strip(".,;:!¿?'()[]{}")
    return " ".join(word.capitalize() if len(word) > 2 else word.lower() for word in text.split())


def strip_accents(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", value) if not unicodedata.combining(char)
    )


def sanitize_print(text: str) -> str:
    text = text.replace("\r", " ").replace("\x00", "").strip()
    return unicodedata.normalize("NFKC", text)


def clean_location_noise(value: str) -> str:
    # Remover frases de viaje comunes
    value = re.sub(
        r"\b(de\s+mi\s+viaje|mi\s+viaje|del?\s+viaje)\b", "", value, flags=re.IGNORECASE
    ).strip()
    
    # Remover prefijos explícitos de origen/destino
    value = re.sub(r"^(?:el\s+)?(origen|destino)\s*[:=]\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"^(?:el\s+)?(origen|destino)\s+(?:es|ser[aá]|seria|será|sera)?\s*(?:ahora|actualmente)?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    
    # MEJORADO: Remover palabras de servicio y transporte al inicio
    value = re.sub(r"^(?:un\s+|el\s+)?(?:transporte|viaje|traslado|servicio)\s+(?:de\s+|desde\s+|urgente\s+)?", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"^(?:necesito|quiero|tengo que|voy|vamos)\s+(?:ir\s+)?(?:de\s+)?", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"^(?:parto|salgo|viajo)\s+(?:de\s+)?", "", value, flags=re.IGNORECASE).strip()
    
    lowered = strip_accents(value).lower()
    
    # Limpiar "hasta" al final
    if " hasta " in lowered:
        idx = lowered.rfind(" hasta ")
        value = value[idx + len(" hasta "):].strip()
        lowered = strip_accents(value).lower()
    
    # Limpiar "desde" al inicio
    if lowered.startswith("desde "):
        value = re.sub(r"^desde\s+", "", value, flags=re.IGNORECASE).strip()
        lowered = strip_accents(value).lower()
    
    # NUEVO: Limpiar "de" al inicio cuando no es parte del nombre del lugar
    if lowered.startswith("de ") and not any(lowered.startswith(prefix) for prefix in ["de la", "del ", "de los", "de las"]):
        value = re.sub(r"^de\s+", "", value, flags=re.IGNORECASE).strip()
    
    # MEJORADO: Dividir en palabras problemáticas más específicas
    split_pattern = re.compile(r"(,| que | quiero | es | sera | será | con | hasta | para | el \d+| a las | \d+\s*de\s+| sin | ida | vuelta | regreso | retorno | solo | sólo | cotización)", re.IGNORECASE)
    value = split_pattern.split(value)[0].strip()
    
    # NUEVO: Filtrar resultados que son claramente no-lugares
    lowered_result = strip_accents(value).lower()
    non_location_words = {
        'urgente', 'transporte', 'viaje', 'traslado', 'servicio', 'cotizacion', 'cotización',
        'necesito', 'quiero', 'tengo', 'voy', 'vamos', 'parto', 'salgo', 'viajo',
        'hola', 'buenos', 'buenas', 'dias', 'tardes', 'noches', 'solo', 'sólo',
        'ida', 'vuelta', 'regreso', 'retorno', 'sin', 'próximo', 'próxima', 'proximo', 'proxima',
        'las', 'los', 'el', 'la', 'de', 'del', 'para', 'por', 'en', 'con', 'hasta',
        'am', 'pm', 'hrs', 'horas', 'hora'
    }
    
    # Si es solo un número seguido de algo, probablemente no es lugar
    if re.match(r'^\d+\s*[a-z]*$', lowered_result):
        return ""
    
    # Si el resultado limpio es solo una palabra no-lugar, retornar vacío
    if len(value.split()) == 1 and lowered_result in non_location_words:
        return ""
    
    # Si contiene principalmente palabras no-lugar, retornar vacío
    words = value.split()
    if len(words) > 1:
        non_location_count = sum(1 for word in words if strip_accents(word).lower() in non_location_words)
        if non_location_count >= len(words) // 2:  # Si más de la mitad son palabras no-lugar
            return ""
    
    return titlecase_clean(value)


def normalize_place(value: str) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value).strip()
    place_prefix = re.compile(
        r"^(?:origen|destino|desde|hacia|hasta|rumbo a|rumbo|direcci[óo]n|punto de partida|punto de destino)\s*[:\-]?\s*",
        re.IGNORECASE,
    )
    text = place_prefix.sub("", text)
    return clean_location_noise(text)


def is_specific_address(value: str) -> bool:
    if not value:
        return False
    lowered = strip_accents(value).lower()
    has_keyword = any(keyword in lowered for keyword in ADDRESS_KEYWORD_SET)
    if DATE_PHRASE_RE.search(value) or contains_time_hint(lowered):
        if not has_keyword:
            return False
    elif MONTH_TOKEN_RE.search(lowered):
        if not has_keyword:
            return False
    if re.search(r"\b\d{1,2}\s*(?:am|pm|hrs?\.?|horas?)\b", lowered):
        return False
    if re.search(r"\b\d{1,2}:\d{2}\b", lowered):
        return False
    if ADDR_CORE_RE.search(lowered):
        return True
    if re.search(r"\b\d{1,7}[a-z]?\b", lowered) and re.search(r"[a-z]{3,}", lowered):
        tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
        if len(tokens) >= 2:
            return True
    return False


def normalize_regreso(value: str) -> Optional[str]:
    if not value:
        return None
    text = unicodedata.normalize("NFKC", value).strip().lower()
    text = re.sub(r"[^a-záéíóúñü\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    negative_matches = [
        "sin regreso",
        "sin vuelta",
        "sin retorno",
        "solo ida",
        "solo de ida",
        "ida solo",
        "ida solamente",
        "no regreso",
        "no llevamos regreso",
        "no tenemos regreso",
        "no quiero regreso",
        "no necesitamos regreso",
        "no requiero regreso",
        "no pedimos regreso",
    ]
    positive_matches = [
        "con regreso",
        "ida y regreso",
        "ida y vuelta",
        "con retorno",
        "con vuelta",
        "con regreso incluido",
        "incluye regreso",
        "incluye retorno",
        "incluye ida y vuelta",
        "incluye ida y regreso",
    ]
    if text in {"no", "n", "no gracias"} or any(phrase in text for phrase in negative_matches):
        return "no"
    if text in {"si", "sí", "s"} or any(phrase in text for phrase in positive_matches):
        return "sí"
    if re.search(r"\bno\b.*\bregreso\b", text):
        return "no"
    if re.search(r"\bcon\b.*\bregreso\b", text) or re.search(r"\bida\s+y\s+vuelta\b", text):
        return "sí"
    return None


def contains_time_hint(text: str) -> bool:
    for word in TIME_HINT_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text):
            return True
    return False


def mentions_field_change(text: str, field: str) -> bool:
    base = strip_accents(text or "").lower()
    tokens = ORIGIN_CHANGE_TOKENS if field == "origen" else DESTINATION_CHANGE_TOKENS
    if any(token in base for token in tokens):
        return True
    if field == "origen" and re.search(r"\borigen\s*[:=]", base):
        return True
    if field == "destino" and re.search(r"\bdestino\s*[:=]", base):
        return True
    return False


def normalize_rut_value(value: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^0-9kK]", "", unicodedata.normalize("NFKC", value))
    if len(cleaned) < 2:
        return ""
    body = cleaned[:-1]
    verifier = cleaned[-1].upper()
    if not body.isdigit():
        return ""
    body = str(int(body))
    return f"{body}-{verifier}"


def extract_rut_value(text: str) -> Optional[str]:
    if not text:
        return None
    match = RUT_RE.search(text)
    if not match:
        return None
    normalized = normalize_rut_value(match.group(0))
    return normalized or None


def extract_email_value(text: str) -> Optional[str]:
    if not text:
        return None
    match = EMAIL_RE.search(text)
    if not match:
        return None
    return match.group(0).lower()


def extract_name_value(text: str) -> Optional[str]:
    if not text:
        return None
    normalized = unicodedata.normalize("NFKC", text)
    normalized = EMAIL_RE.sub(" ", normalized)
    normalized = RUT_RE.sub(" ", normalized)
    normalized = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ'\s]", " ", normalized)
    normalized = re.sub(
        r"\b(correo|electr[oó]nico|mail|nombre|rut|registro|contacto)\b",
        " ",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None
    words = [word for word in normalized.split() if len(word) > 1]
    if not words:
        return None
    leading_connectors = {"me", "soy", "llamo", "llamó", "llame", "nombre", "es", "mi"}
    filtered: List[str] = []
    for word in words:
        base = strip_accents(word).lower()
        if not filtered and (base in leading_connectors or base in NAME_STOPWORDS):
            continue
        if base in NAME_STOPWORDS:
            return None
        filtered.append(word)
    if not filtered or len(filtered) > 4:
        return None
    return titlecase_clean(" ".join(filtered))


def extract_structured_pairs(text: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    if not text:
        return pairs
    for match in STRUCTURED_KEY_PATTERN.finditer(text):
        key = re.sub(r"\s+", " ", match.group(1).lower()).strip()
        value = match.group(2).strip().strip("\"' ")
        if value:
            pairs.setdefault(key, value)
    return pairs


def normalize_for_intent(text: str) -> str:
    base = strip_accents(text or "").lower()
    base = re.sub(r"(.)\\1{2,}", r"\\1\\1", base)
    base = re.sub(r"[^a-z0-9\s]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def build_display_state(state: Dict[str, Any]) -> Dict[str, Any]:
    display: Dict[str, Any] = {}
    for key in STATE_FIELDS:
        if key == "cantidad":
            display[key] = state.get(key, 0)
        else:
            display[key] = state.get(key, "")
    return display


def print_bot_turn(message: str, state: Dict[str, Any]) -> None:
    sanitized = sanitize_print(message or "")
    lines = sanitized.splitlines() if sanitized else []
    display_state = build_display_state(state)
    print()
    print(f"({json.dumps(display_state, ensure_ascii=False)})")
    if lines:
        print(f"Transporte Miramar: {lines[0]}")
        for line in lines[1:]:
            print(line)
    else:
        print("Transporte Miramar:")


def extract_quantity(user_input: str) -> Optional[int]:
    if not user_input:
        return None
    normalized = unicodedata.normalize("NFKC", user_input)
    simple = strip_accents(normalized).lower()

    match = QUANTITY_STANDALONE_DIGIT_RE.match(normalized)
    if match:
        try:
            value = int(match.group(1))
            if value > 0:
                return value
        except ValueError:
            pass

    match = WORD_DIGIT_PATTERN.search(normalized)
    if match:
        try:
            value = int(match.group(1))
            if value > 0:
                return value
        except ValueError:
            pass

    for pattern in WORD_CONTEXT_PATTERNS:
        match = pattern.search(simple)
        if match:
            number = SPANISH_NUMBER_WORDS.get(match.group(1))
            if number:
                return number

    context_match = re.search(
        r"\b(\d{1,3})\b\s*(?:personas?|pasajeros?|viajeros?|clientes|ocupantes)",
        normalized,
    )
    if context_match:
        try:
            value = int(context_match.group(1))
            if value > 0:
                return value
        except ValueError:
            pass

    # Heurística para frases del tipo "yo y mi pareja", "mis padres y yo", etc.
    relation_terms = [
        r"mi pareja",
        r"mi esposo",
        r"mi esposa",
        r"mi mujer",
        r"mi marido",
        r"mi novio",
        r"mi novia",
        r"mi pololo",
        r"mi polola",
        r"mi papa",
        r"mi papá",
        r"mi mama",
        r"mi mamá",
        r"mi padre",
        r"mi madre",
        r"mi hijo",
        r"mi hija",
        r"mi hermano",
        r"mi hermana",
        r"mi amigo",
        r"mi amiga",
        r"mi tio",
        r"mi tío",
        r"mi tia",
        r"mi tía",
        r"mi sobrino",
        r"mi sobrina",
    ]

    heuristic_count = 0
    if re.search(r"\byo\b", simple):
        heuristic_count += 1
    for term in relation_terms:
        heuristic_count += len(re.findall(term, simple))

    if heuristic_count == 0 and re.search(r"\bnosotr[oa]s\b", simple):
        heuristic_count = 2

    if heuristic_count > 0:
        return heuristic_count
    return None


def normalize_fecha(value: str, context: str) -> str:
    if not value:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return value.strip()
    try:
        dt = datetime.datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        return value
    if re.search(r"\b20\d{2}\b", context):
        return value.strip()
    today = datetime.date.today()
    candidate = datetime.date(today.year, dt.month, dt.day)
    if candidate < today:
        candidate = datetime.date(today.year + 1, dt.month, dt.day)
    return candidate.isoformat()


def user_mentions_date(user_input: str) -> bool:
    if not user_input:
        return False
    text = unicodedata.normalize("NFKC", user_input)
    lowered = text.lower()
    if any(pattern.search(text) for pattern in DATE_PATTERNS):
        return True
    return any(keyword in lowered for keyword in DATE_KEYWORDS)


def user_mentions_time(user_input: str) -> bool:
    if not user_input:
        return False
    text = unicodedata.normalize("NFKC", user_input)
    return any(pattern.search(text) for pattern in TIME_PATTERNS)


def user_mentions_regreso(user_input: str) -> bool:
    if not user_input:
        return False
    text = unicodedata.normalize("NFKC", user_input)
    return any(pattern.search(text) for pattern in REGRESO_NO_PATTERNS + REGRESO_YES_PATTERNS)


def extract_location(user_input: str, field: str = "origen") -> Optional[str]:
    text = user_input.strip()
    
    if field == "origen":
        # Patrones mejorados para origen
        origin_patterns = [
            # "Necesito ir desde X"
            r"(?:necesito|quiero)\s+ir\s+desde\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s',0-9]+?)(?:\s+a\s|\s+hacia\s|\s+hasta\s|$)",
            # "Desde X" (al inicio)
            r"^desde\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s',0-9]+?)(?:\s+a\s|\s+hacia\s|\s+hasta\s|$)",
            # "Salgo de X"
            r"(?:salgo|parto)\s+de\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s',0-9]+?)(?:\s+a\s|\s+hacia\s|\s+hasta\s|$)",
            # "Mi origen es X"
            r"(?:mi\s+)?origen\s+es\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s',0-9]+?)$",
        ]
        
        for pattern in origin_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = normalize_place(match.group(1))
                if value and len(value) >= 2 and value.lower() not in LOCATION_EXCLUDE:
                    return value
                    
    else:  # destino
        # Patrones mejorados para destino
        dest_patterns = [
            # "Quiero ir a X"
            r"(?:quiero|necesito)\s+ir\s+a\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s',0-9]+?)$",
            # "Hacia X"
            r"hacia\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s',0-9]+?)$",
            # "Hasta X"
            r"hasta\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s',0-9]+?)$",
            # "Mi destino es X"
            r"(?:mi\s+)?destino\s+es\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s',0-9]+?)$",
            # "Voy a X"
            r"voy\s+a\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s',0-9]+?)$",
        ]
        
        for pattern in dest_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = normalize_place(match.group(1))
                if value and len(value) >= 2 and value.lower() not in LOCATION_EXCLUDE:
                    return value
    
    return None


def extract_origin_dest_pair(user_input: str) -> Tuple[Optional[str], Optional[str]]:
    text = user_input.strip()
    
    # PATRONES ESPECÍFICOS MEJORADOS para frases comunes como las escriben en WhatsApp
    improved_patterns = [
        # "Necesito transporte [cualquier_cosa] X a Y" - NUEVO patrón más flexible
        re.compile(r"(?:necesito|quiero)\s+(?:transporte|ir)\s+(?:\w+\s+)*(.+?)\s+a\s+(.+?)(?:\s+para|\s+el|\s*$)", re.IGNORECASE),
        # "Desde X hacia Y" - muy común en WhatsApp
        re.compile(r"desde\s+(.+?)\s+hacia\s+(.+?)(?:\s+para|\s+el|\s*$)", re.IGNORECASE),
        # "Desde X hasta Y" 
        re.compile(r"desde\s+(.+?)\s+hasta\s+(.+?)(?:\s+para|\s+el|\s*$)", re.IGNORECASE),
        # "Desde X a Y"
        re.compile(r"desde\s+(.+?)\s+a\s+(.+?)(?:\s+para|\s+el|\s*$)", re.IGNORECASE),
        # "De X a Y" (inicio de frase)
        re.compile(r"^de\s+(.+?)\s+a\s+(.+?)(?:\s+para|\s+el|\s*$)", re.IGNORECASE),
        # "X hasta Y" (simple)
        re.compile(r"^(.+?)\s+hasta\s+(.+?)(?:\s+para|\s+el|\s*$)", re.IGNORECASE),
        # NUEVO: "transporte [cualquier_cosa] X a Y" - para casos como "transporte urgente Santiago a Rancagua"
        re.compile(r"transporte\s+(?:\w+\s+)*(.+?)\s+a\s+(.+?)(?:\s+para|\s+el|\s*$)", re.IGNORECASE),
        # NUEVO: "transporte para X personas de Y al Z" - para casos empresariales
        re.compile(r"transporte\s+para\s+\d+\s+personas\s+de\s+(.+?)\s+al?\s+(.+?)(?:\s+el|\s+para|\s+a\s+las|\s*$)", re.IGNORECASE),
        # NUEVO: "X a Y" simple al inicio
        re.compile(r"^(.+?)\s+a\s+(.+?)(?:\s+el|\s+para|\s+a\s+las|\s+sin|\s+ida|\s+vuelta|\s*$)", re.IGNORECASE),
    ]
    
    # Primero intentar con patrones mejorados
    for i, pattern in enumerate(improved_patterns):
        match = pattern.search(text)
        if match:
            origin_raw = match.group(1).strip()
            dest_raw = match.group(2).strip()
            
            # Limpiar y normalizar
            origin = normalize_place(origin_raw)
            destination = normalize_place(dest_raw)
            
            # Verificar que ambos sean válidos
            if (origin and destination and 
                len(origin) >= 2 and len(destination) >= 2 and
                origin.lower() not in LOCATION_EXCLUDE and 
                destination.lower() not in LOCATION_EXCLUDE):
                return origin, destination
    
    # Fallback a patrones originales
    for pattern in ORIG_DEST_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        origin_raw = match.groupdict().get("orig")
        dest_raw = match.groupdict().get("dest")
        if not origin_raw or not dest_raw:
            continue
        origin = normalize_place(origin_raw)
        destination = normalize_place(dest_raw)
        if origin and destination and origin.lower() not in LOCATION_EXCLUDE and destination.lower() not in LOCATION_EXCLUDE:
            return origin, destination
    
    # Fallback manual usando "hasta"
    split_parts = re.split(r"\bhasta\b", text, maxsplit=1, flags=re.IGNORECASE)
    if len(split_parts) == 2:
        origin_raw = split_parts[0].strip(" ,;:\n\t")
        dest_raw = split_parts[1].strip(" ,;:\n\t")
        if origin_raw and dest_raw:
            origin = normalize_place(origin_raw)
            destination = normalize_place(dest_raw)
            if origin and destination and origin.lower() not in LOCATION_EXCLUDE and destination.lower() not in LOCATION_EXCLUDE:
                return origin, destination
    return None, None


def parse_numeric_date(text: str) -> Optional[str]:
    normalized = unicodedata.normalize("NFKC", text)
    # Mejorado: incluir contexto como "el 10/10"
    match = re.search(r"(?:el\s+)?(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", normalized)
    if not match:
        return None
    day = int(match.group(1))
    month = int(match.group(2))
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    year_part = match.group(3)
    if year_part:
        year = int(year_part)
        if year < 100:
            year += 2000 if year < 50 else 1900
    else:
        year = datetime.date.today().year
    try:
        date_obj = datetime.date(year, month, day)
    except ValueError:
        return None
    return date_obj.isoformat()


def parse_textual_date(text: str) -> Optional[str]:
    normalized = unicodedata.normalize("NFKC", text)
    lowered = strip_accents(normalized).lower()
    if "pasado manana" in lowered or "pasado-manana" in lowered:
        target = datetime.date.today() + datetime.timedelta(days=2)
        return target.isoformat()
    if "mañana" in normalized.lower() or "manana" in lowered:
        target = datetime.date.today() + datetime.timedelta(days=1)
        return target.isoformat()
    match = re.search(rf"\b(\d{{1,2}})\s+de\s+({MONTH_PATTERN})\b", normalized, flags=re.IGNORECASE)
    if match:
        day = int(match.group(1))
        month_name = strip_accents(match.group(2)).lower()
        month = MONTHS.get(month_name)
        if month:
            year = datetime.date.today().year
            try:
                date_obj = datetime.date(year, month, day)
            except ValueError:
                return None
            return date_obj.isoformat()
    match = re.search(rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})\b", normalized, flags=re.IGNORECASE)
    if match:
        month_name = strip_accents(match.group(1)).lower()
        day = int(match.group(2))
        month = MONTHS.get(month_name)
        if month:
            year = datetime.date.today().year
            try:
                date_obj = datetime.date(year, month, day)
            except ValueError:
                return None
            return date_obj.isoformat()
    return None


def parse_time_from_text(text: str) -> Optional[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    match = re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", normalized)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        meridiem = match.group(3)
        if meridiem:
            if hour == 12:
                hour = 0
            if meridiem == "pm":
                hour += 12
        return f"{hour:02d}:{minute:02d}"
    match = re.search(r"\b(\d{1,2})\s*(am|pm)\b", normalized)
    if match:
        hour = int(match.group(1)) % 12
        meridiem = match.group(2)
        if meridiem == "pm":
            hour += 12
        return f"{hour:02d}:00"
    match = re.search(r"\b(\d{1,2})\s*(?:hrs?\.?|horas?)\b", normalized)
    if match:
        hour = int(match.group(1))
        if 0 <= hour <= 23:
            return f"{hour:02d}:00"
    if "mediod" in normalized:
        return "12:00"
    if "medianoche" in normalized:
        return "00:00"
    return None


def extract_structured_pairs_from_comment(value: str) -> Dict[str, str]:
    return extract_structured_pairs(value)


def build_fallback_question(missing: List[str]) -> str:
    ordered = list(STATE_FIELDS)
    missing_list = [item for item in ordered if item in missing]
    if not missing_list:
        return "¿Podrías confirmarme el dato que falta para continuar?"
    prompts: List[str] = []
    if "origen" in missing_list and "destino" in missing_list:
        prompts.append(random.choice(FALLBACK_FIELD_PROMPTS["origen_destino"]))
        missing_list = [item for item in missing_list if item not in {"origen", "destino"}]
    for field in missing_list:
        if field == "fecha":
            prompts.append(random.choice(FALLBACK_FIELD_PROMPTS["fecha"]))
        elif field == "hora":
            prompts.append(random.choice(FALLBACK_FIELD_PROMPTS["hora"]))
        elif field == "regreso":
            prompts.append(random.choice(FALLBACK_FIELD_PROMPTS["regreso"]))
        elif field == "cantidad":
            prompts.append(random.choice(FALLBACK_FIELD_PROMPTS["cantidad"]))
        elif field == COMMENT_FIELD:
            prompts.append(random.choice(FALLBACK_FIELD_PROMPTS[COMMENT_FIELD]))
        elif field == "origen":
            prompts.append(random.choice(FALLBACK_FIELD_PROMPTS["origen"]))
        elif field == "destino":
            prompts.append(random.choice(FALLBACK_FIELD_PROMPTS["destino"]))
        if len(prompts) == 2:
            break
    if not prompts:
        return "¿Podrías confirmarme el dato que falta para continuar?"
    if len(prompts) == 1:
        return prompts[0]
    return f"{prompts[0]} {prompts[1]}"


def select_question_fields(missing: List[str]) -> List[str]:
    ordered = ["origen", "destino", "fecha", "hora", "regreso", "cantidad"]
    missing_set = set(missing)
    if not missing_set:
        return []
    if {"origen", "destino"}.issubset(missing_set):
        return ["origen", "destino"]
    if {"fecha", "hora"}.issubset(missing_set):
        return ["fecha", "hora"]
    if {"regreso", "cantidad"}.issubset(missing_set):
        return ["regreso", "cantidad"]
    selected: List[str] = []
    for field in ordered:
        if field in missing_set:
            selected.append(field)
            if len(selected) == 2:
                break
    return selected


def question_mentions_forbidden(question: str, missing: List[str]) -> bool:
    lowered = question.lower()
    forbidden: List[str] = []
    if "origen" not in missing:
        forbidden.extend([
            "origen",
            "desde",
            "punto de partida",
            "de dónde",
            "donde parte",
            "inicio del viaje",
        ])
    if "destino" not in missing:
        forbidden.extend([
            "destino",
            "hasta",
            "hacia",
            "a dónde",
            "donde llega",
        ])
    if "regreso" not in missing:
        forbidden.extend([
            "regreso",
            "vuelta",
            "retorno",
        ])
    if "fecha" not in missing:
        forbidden.extend([
            "fecha",
            "cuándo viajas",
        ])
    if "hora" not in missing:
        forbidden.extend([
            "hora",
            "a qué hora",
        ])
    if "cantidad" not in missing:
        forbidden.extend([
            "cuántas personas",
            "personas viajarán",
            "pasajeros",
        ])
    return any(token in lowered for token in forbidden)


def validate_and_improve_address(address_text: str) -> Dict[str, Any]:
    """
    Valida y mejora una dirección usando libpostal y nominatim como prioridad,
    con validación manual como fallback.
    
    REGLA IMPORTANTE: Solo acepta direcciones completas con calle/avenida + número/comuna.
    Nombres de ciudades solas NO son válidos.
    
    Returns:
        Dict con información de la dirección validada y mejorada
    """
    if not address_text or len(address_text.strip()) < 3:
        return {
            "is_valid": False,
            "needs_more_detail": True,
            "suggestion": "Por favor, proporciona una dirección más específica con calle, número y comuna.",
            "original": address_text,
            "improved": None,
            "source": "validation"
        }
    
    # VALIDACIÓN ESTRICTA: Si es solo nombre de ciudad, rechazar inmediatamente
    if is_just_city_name(address_text):
        return {
            "is_valid": False,
            "needs_more_detail": True,
            "suggestion": f"CIUDAD_ONLY:{address_text}",  # Marcador para LLM
            "original": address_text,
            "improved": None,
            "source": "city_name_rejected"
        }
    
    # PRIORIDAD 1: Intentar extraer y validar con libpostal + nominatim
    extracted = detect_and_extract_address(address_text)
    
    if extracted:
        # Se procesó correctamente con libpostal
        street = extracted.get("street", "")
        number = extracted.get("number", "")
        comuna = extracted.get("comuna", "")
        
        # VALIDACIÓN MEJORADA: Una dirección es válida si tiene:
        # - Calle Y número (suficiente para cotización)
        # - O calle Y comuna (también válido)
        # - O los tres (ideal pero no obligatorio)
        has_street = bool(street and len(street) > 2)
        has_number = bool(number)
        has_comuna = bool(comuna)
        
        # DIRECCIÓN VÁLIDA: Calle + número ES suficiente (comuna opcional)
        if has_street and has_number:
            # Ya tenemos lo mínimo necesario para una cotización
            improved_parts = [street, number]
            if has_comuna:
                improved_parts.append(comuna)
            
            improved_address = ", ".join(improved_parts)
            
            # Solo sugerir comuna si no la tiene, pero NO es obligatorio
            suggestion = None
            if not has_comuna:
                suggestion = f"Perfecto. Si quieres mayor precisión, puedes agregar la comuna, pero con '{street} {number}' ya tenemos lo necesario."
            
            return {
                "is_valid": True,
                "needs_more_detail": False,  # NO necesita más detalles
                "suggestion": suggestion,
                "original": address_text,
                "improved": improved_address,
                "details": extracted,
                "source": "libpostal"
            }
        
        # DIRECCIÓN VÁLIDA: Calle + comuna (sin número específico)
        elif has_street and has_comuna:
            improved_address = f"{street}, {comuna}"
            
            suggestion = f"Perfecto. Si quieres mayor precisión, puedes agregar el número de la dirección, pero con '{street}, {comuna}' ya es suficiente."
            
            return {
                "is_valid": True,
                "needs_more_detail": False,  # Comuna compensa la falta de número
                "suggestion": suggestion,
                "original": address_text,
                "improved": improved_address,
                "details": extracted,
                "source": "libpostal"
            }
        
        # DIRECCIÓN INCOMPLETA: Solo calle sin número ni comuna
        else:
            return {
                "is_valid": False,
                "needs_more_detail": True,
                "suggestion": f"La dirección '{address_text}' necesita más detalles. Por favor agrega el número o la comuna.",
                "original": address_text,
                "improved": None,
                "source": "libpostal_incomplete"
            }
    
    # FALLBACK: Validación manual ESTRICTA
    # Solo acepta direcciones que claramente tienen calle + números/referencia
    if is_complete_address_manual(address_text):
        return {
            "is_valid": True,
            "needs_more_detail": False,
            "suggestion": None,
            "original": address_text,
            "improved": address_text.strip(),
            "source": "manual_validation"
        }
    else:
        return {
            "is_valid": False,
            "needs_more_detail": True,
            "suggestion": f"DIRECCION_INCOMPLETA:{address_text}",  # Marcador para LLM
            "original": address_text,
            "improved": None,
            "source": "manual_validation"
        }


def extract_and_validate_datetime_info(text: str) -> Dict[str, Any]:
    """
    Extrae y valida información de fecha y hora del texto del usuario.
    
    Returns:
        Dict con información de fecha/hora extraída y validada
    """
    result = {
        "has_date": False,
        "has_time": False,
        "extracted_date": None,
        "extracted_time": None,
        "normalized_date": None,
        "normalized_time": None,
        "needs_clarification": False,
        "suggestion": None
    }
    
    # Detectar si menciona fecha
    if user_mentions_date(text):
        result["has_date"] = True
        # Intentar extraer fecha específica
        date_extracted = parse_textual_date(text)
        if date_extracted:
            result["extracted_date"] = date_extracted
            result["normalized_date"] = normalize_fecha(date_extracted, text)
        else:
            result["needs_clarification"] = True
            result["suggestion"] = "¿Podrías especificar la fecha exacta? Por ejemplo: '15 de marzo' o '2024-03-15'"
    
    # Detectar si menciona hora
    if user_mentions_time(text):
        result["has_time"] = True
        # Intentar extraer hora específica
        time_extracted = parse_time_from_text(text)
        if time_extracted:
            result["extracted_time"] = time_extracted
            result["normalized_time"] = time_extracted
        else:
            result["needs_clarification"] = True
            result["suggestion"] = "¿Podrías especificar la hora? Por ejemplo: '14:30' o '2:30 PM'"
    
    return result


def analyze_user_response_context(user_input: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analiza la respuesta del usuario en el contexto actual para determinar
    qué información contiene y qué acciones tomar.
    """
    analysis = {
        "locations": {},
        "datetime": {},
        "quantity": None,
        "regreso": None,
        "personal_data": {},
        "needs_address_detail": False,
        "suggestions": []
    }
    
    # PASO 1: PRIORIDAD MÁXIMA - Extraer origen y destino usando patrones específicos
    # Intentar extraer par origen-destino primero
    origin, destination = extract_origin_dest_pair(user_input)
    
    if origin and not current_state.get("origen"):
        # Validar origen usando libpostal + validación estricta
        addr_analysis = validate_and_improve_address(origin)
        analysis["locations"]["origen"] = addr_analysis
        
        # Si no es válido, NO usar fallback heurístico
        # El sistema debe ser estricto y pedir dirección completa
        if not addr_analysis["is_valid"]:
            analysis["needs_address_detail"] = True
            if addr_analysis.get("suggestion"):
                analysis["suggestions"].append(addr_analysis["suggestion"])
        elif addr_analysis["needs_more_detail"]:
            analysis["needs_address_detail"] = True
            if addr_analysis.get("suggestion"):
                analysis["suggestions"].append(addr_analysis["suggestion"])
    
    if destination and not current_state.get("destino"):
        # Validar destino usando libpostal + validación estricta
        addr_analysis = validate_and_improve_address(destination)
        analysis["locations"]["destino"] = addr_analysis
        
        # Si no es válido, NO usar fallback heurístico
        # El sistema debe ser estricto y pedir dirección completa
        if not addr_analysis["is_valid"]:
            analysis["needs_address_detail"] = True
            if addr_analysis.get("suggestion"):
                analysis["suggestions"].append(addr_analysis["suggestion"])
        elif addr_analysis["needs_more_detail"]:
            analysis["needs_address_detail"] = True
            if addr_analysis.get("suggestion"):
                analysis["suggestions"].append(addr_analysis["suggestion"])
    
    # Si no se encontró par, intentar extraer individual
    if not origin and not current_state.get("origen"):
        origin_single = extract_location(user_input, "origen")
        if origin_single:
            addr_analysis = validate_and_improve_address(origin_single)
            analysis["locations"]["origen"] = addr_analysis
            if not addr_analysis["is_valid"] or addr_analysis["needs_more_detail"]:
                analysis["needs_address_detail"] = True
                if addr_analysis.get("suggestion"):
                    analysis["suggestions"].append(addr_analysis["suggestion"])
    
    if not destination and not current_state.get("destino"):
        dest_single = extract_location(user_input, "destino")
        if dest_single:
            addr_analysis = validate_and_improve_address(dest_single)
            analysis["locations"]["destino"] = addr_analysis
            if not addr_analysis["is_valid"] or addr_analysis["needs_more_detail"]:
                analysis["needs_address_detail"] = True
                if addr_analysis.get("suggestion"):
                    analysis["suggestions"].append(addr_analysis["suggestion"])
        else:
            # Intentar extraer destino con patrones específicos como fallback final
            dest_patterns = [
                r"(?:hacia|hasta|destino|llegar)\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s',0-9]+?)(?:\s*[,.]|$)",
                r"(?:a|para)\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s',0-9]+?)(?:\s*[,.]|$)",
            ]
            for pattern in dest_patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    dest_candidate = normalize_place(match.group(1).strip())
                    if dest_candidate and len(dest_candidate) > 2:
                        addr_analysis = validate_and_improve_address(dest_candidate)
                        analysis["locations"]["destino"] = addr_analysis
                        if not addr_analysis["is_valid"] or addr_analysis["needs_more_detail"]:
                            analysis["needs_address_detail"] = True
                            if addr_analysis.get("suggestion"):
                                analysis["suggestions"].append(addr_analysis["suggestion"])
                        break
    
    # PASO 1.5: Si no se encontraron direcciones VÁLIDAS con métodos simples, usar extracción compleja
    origen_valido = analysis["locations"].get("origen", {}).get("is_valid", False)
    destino_valido = analysis["locations"].get("destino", {}).get("is_valid", False)
    
    # Usar extracción compleja si no tenemos origen Y destino válidos
    if (not origen_valido and not current_state.get("origen")) or (not destino_valido and not current_state.get("destino")):
        complex_analysis = extract_multiple_addresses_from_complex_phrase(user_input)
        
        if complex_analysis["best_origin"] and not current_state.get("origen") and not origen_valido:
            origin_addr = complex_analysis["best_origin"]
            analysis["locations"]["origen"] = origin_addr["validation"]
            
            if not origin_addr["validation"]["is_valid"]:
                analysis["needs_address_detail"] = True
                if origin_addr["validation"].get("suggestion"):
                    analysis["suggestions"].append(origin_addr["validation"]["suggestion"])
        
        if complex_analysis["best_destination"] and not current_state.get("destino") and not destino_valido:
            dest_addr = complex_analysis["best_destination"]
            analysis["locations"]["destino"] = dest_addr["validation"]
            
            if not dest_addr["validation"]["is_valid"]:
                analysis["needs_address_detail"] = True
                if dest_addr["validation"].get("suggestion"):
                    analysis["suggestions"].append(dest_addr["validation"]["suggestion"])
    
    # PASO 2: Analizar fecha y hora
    analysis["datetime"] = extract_and_validate_datetime_info(user_input)
    if analysis["datetime"]["needs_clarification"]:
        analysis["suggestions"].append(analysis["datetime"]["suggestion"])
    
    # PASO 3: Analizar cantidad
    quantity = extract_quantity(user_input)
    if quantity:
        analysis["quantity"] = quantity
    
    # PASO 4: Analizar regreso
    regreso = normalize_regreso(user_input)
    if regreso:
        analysis["regreso"] = regreso
    
    # PASO 5: Analizar datos personales SOLO cuando el contexto actual lo requiere
    # NO extraer nombres automáticamente de frases sobre viajes
    
    # Solo extraer email si parece ser una dirección de correo explícita
    email = extract_email_value(user_input)
    if email:
        analysis["personal_data"]["correo"] = email
    
    # Solo extraer RUT si parece ser un RUT explícito
    rut = extract_rut_value(user_input)
    if rut:
        analysis["personal_data"]["rut"] = rut
    
    # IMPORTANTE: NO extraer nombres automáticamente de descripciones de viajes
    # Solo extraer si el usuario está explícitamente dando su nombre personal
    if is_explicit_personal_introduction(user_input):
        name = extract_name_value(user_input)
        if name and not is_likely_chilean_location(name):
            analysis["personal_data"]["nombre"] = name
    
    return analysis


def is_explicit_personal_introduction(text: str) -> bool:
    """
    Detecta si el usuario está explícitamente introduciendo su nombre personal,
    no describiendo un viaje o ubicaciones.
    """
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Patrones que indican introducción personal explícita
    personal_intro_patterns = [
        r"\bme llamo\b", r"\bsoy\b", r"\bmi nombre es\b",
        r"\bme llaman\b", r"\bnombre:\s*", r"\bmi nombre:\s*"
    ]
    
    # Si contiene patrones de introducción personal
    has_intro_pattern = any(re.search(pattern, text_lower) for pattern in personal_intro_patterns)
    
    # Si NO tiene patrones de viaje/transporte
    travel_patterns = [
        r"\bnecesito\s+ir\b", r"\bviaje\b", r"\btransporte\b", r"\btraslado\b",
        r"\bdesde\b", r"\bhasta\b", r"\bhacia\b", r"\ba\s+las?\b",
        r"\bmañana\b", r"\bhoy\b", r"\bfecha\b", r"\bhora\b",
        r"\bpersonas?\b", r"\bpasajeros?\b", r"\bregreso\b", r"\bida\b"
    ]
    has_travel_pattern = any(re.search(pattern, text_lower) for pattern in travel_patterns)
    
    # Solo considerar introducción personal si tiene patrón de intro Y NO tiene patrón de viaje
    return has_intro_pattern and not has_travel_pattern


def is_libpostal_available() -> bool:
    """Verifica si libpostal está disponible para uso."""
    try:
        from .direcciones import parse_address
        return parse_address is not None
    except ImportError:
        return False


def is_likely_chilean_location(location: str) -> bool:
    """
    Detecta si una ubicación parece ser una ciudad o lugar chileno válido.
    Funciona como fallback heurístico cuando libpostal no reconoce el lugar.
    """
    if not location or len(location.strip()) < 3:
        return False
    
    location_clean = strip_accents(location.strip()).lower()
    
    # Lista de ciudades principales de Chile (común en transporte)
    major_chilean_cities = {
        "santiago", "valparaiso", "valparaíso", "viña del mar", "vina del mar",
        "concepcion", "concepción", "antofagasta", "temuco", "rancagua",
        "talca", "arica", "iquique", "puerto montt", "chillan", "chillán",
        "valdivia", "osorno", "calama", "copiapó", "copiapó", "la serena",
        "curicó", "curico", "quillota", "san antonio", "melipilla",
        "los andes", "ovalle", "linares", "talagante", "buin",
        "maipú", "maipu", "las condes", "providencia", "ñuñoa", "nunoa",
        "puente alto", "san bernardo", "la florida", "maule", "bío bío", "bio bio"
    }
    
    # Verificar si es una ciudad conocida
    if location_clean in major_chilean_cities:
        return True
    
    # Verificar patrones típicos chilenos
    chilean_patterns = [
        r"\b(del?\s+)?mar\b",  # del Mar, Mar
        r"\bsur\b", r"\bnorte\b", r"\boriente\b", r"\bponiente\b",  # puntos cardinales
        r"\blos\s+\w+", r"\blas\s+\w+",  # Los/Las + nombre
        r"\bsan\s+\w+", r"\bsanta\s+\w+",  # San/Santa + nombre
        r"\bvilla\s+\w+", r"\bpuerto\s+\w+",  # Villa/Puerto + nombre
        r"\bla\s+\w+", r"\bel\s+\w+"  # La/El + nombre
    ]
    
    for pattern in chilean_patterns:
        if re.search(pattern, location_clean):
            return True
    
    # Verificar si tiene características de lugar chileno (al menos 2 palabras)
    words = location_clean.split()
    if len(words) >= 2:
        # Si tiene artículos en español típicos de nombres de lugares
        articles = {"de", "del", "de la", "los", "las", "la", "el"}
        if any(word in articles for word in words):
            return True
    
    # Si no cumple ningún patrón, probablemente no es una ubicación conocida
    return False


def is_just_city_name(location: str) -> bool:
    """
    Detecta si el texto es solo un nombre de ciudad SIN dirección específica.
    Esto debe ser rechazado ya que necesitamos direcciones completas.
    
    IMPORTANTE: NO rechazar direcciones que tengan números, incluso sin "Av." o "Calle".
    """
    if not location or len(location.strip()) < 3:
        return False
    
    location_clean = strip_accents(location.strip()).lower()
    
    # SI TIENE NÚMEROS, NO es solo ciudad - es dirección específica
    if re.search(r'\d+', location_clean):
        return False  # "Providencia 1234" NO es solo ciudad
    
    # Lista de ciudades/comunas chilenas comunes (estas DEBEN ser rechazadas si van solas)
    chilean_cities_only = {
        "santiago", "valparaiso", "valparaíso", "viña del mar", "vina del mar",
        "concepcion", "concepción", "antofagasta", "temuco", "rancagua",
        "talca", "arica", "iquique", "puerto montt", "chillan", "chillán",
        "valdivia", "osorno", "calama", "copiapó", "copiapó", "la serena",
        "curicó", "curico", "quillota", "san antonio", "melipilla",
        "los andes", "ovalle", "linares", "talagante", "buin",
        "maipú", "maipu", "las condes", "providencia", "ñuñoa", "nunoa",
        "puente alto", "san bernardo", "la florida", "maule", "bío bío", "bio bio",
        "san miguel", "la reina", "vitacura", "peñalolen", "peñalolén"
    }
    
    # Si es exactamente un nombre de ciudad conocida SIN números, rechazar
    if location_clean in chilean_cities_only:
        return True
    
    # Si son 1-2 palabras sin números ni indicadores de dirección, probablemente es solo ciudad
    words = location_clean.split()
    if len(words) <= 2:
        # No tiene indicadores de dirección específica
        has_address_indicators = any(indicator in location_clean for indicator in [
            "calle", "avenida", "av.", "pasaje", "psje", "camino", "numero", "número", "#",
            "block", "depto", "departamento", "casa", "parcela"
        ])
        
        # Si no tiene números NI indicadores de dirección, es solo ciudad
        if not has_address_indicators:
            return True
    
    return False


def is_complete_address_manual(address: str) -> bool:
    """
    Validación manual mejorada para determinar si una dirección está completa.
    Acepta direcciones con calle + número (comuna opcional).
    """
    if not address or len(address.strip()) < 5:
        return False
    
    address_clean = strip_accents(address.strip()).lower()
    
    # REQUISITO 1: Indicadores de dirección específica
    street_indicators = [
        "calle", "avenida", "av.", "avda.", "pasaje", "psje", "camino", "plaza",
        "block", "depto", "departamento", "casa", "parcela", "sitio"
    ]
    
    has_street_indicator = any(indicator in address_clean for indicator in street_indicators)
    
    # REQUISITO 2: Números en la dirección
    has_numbers = bool(re.search(r'\d+', address_clean))
    
    # VALIDACIÓN PRINCIPAL: Calle + número = dirección válida
    if has_street_indicator and has_numbers:
        # Verificar que sea una dirección real, no solo "calle 1" o algo muy básico
        words = address_clean.split()
        if len(words) >= 2:  # Al menos "avenida 1234" o "calle providencia"
            return True
    
    # CASO ESPECIAL: Direcciones bien conocidas sin indicador explícito
    # Ej: "Providencia 1234", "Las Condes 567"
    if has_numbers and not has_street_indicator:
        words = address_clean.split()
        if len(words) >= 2:  # Al menos 2 palabras con números
            # Verificar que no sea solo un número
            non_numeric_words = [w for w in words if not w.isdigit()]
            if len(non_numeric_words) >= 1:  # Al menos una palabra no numérica
                return True
    
    # CASO ESPECIAL: Lugares específicos reconocibles
    known_patterns = [
        r'plaza\s+\w+',  # "plaza de armas"
        r'\w+\s+\d+\s*,\s*\w+',  # "nombre 123, comuna"
        r'\w+\s+\d+.*\w+',  # "providencia 1234 santiago"
    ]
    
    for pattern in known_patterns:
        if re.search(pattern, address_clean):
            return True
    
    return False


def extract_multiple_addresses_from_complex_phrase(text: str) -> Dict[str, Any]:
    """
    Extrae múltiples direcciones de frases complejas como las escriben en WhatsApp.
    Maneja casos como: "Si, 1 norte 1161, viña del mar hasta santiago la dirección es Nueva imperial 5162"
    
    Returns:
        Dict con todas las direcciones encontradas y su análisis
    """
    result = {
        "addresses_found": [],
        "origin_candidates": [],
        "destination_candidates": [],
        "best_origin": None,
        "best_destination": None
    }
    
    # PASO 1: Buscar patrones de direcciones específicas (calle + número)
    # Patrones que indican direcciones válidas
    address_patterns = [
        # Calle + número (sin indicador explícito)
        r'\b([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]*\s*\d+[A-Za-z]*)\b',
        # Con indicadores explícitos
        r'\b(?:av|avenida|calle|pasaje|camino)\.?\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\s]+\d+[A-Za-z]*)\b',
        # Formato de regiones (1 norte 1234)
        r'\b(\d+\s+(?:norte|sur|oriente|poniente)\s+\d+)\b',
        # Direcciones con guiones y números de teléfono
        r'\b([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+\s+\d+(?:-\d+)*)\b'
    ]
    
    # Encontrar todas las direcciones potenciales
    found_addresses = []
    for pattern in address_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            addr_text = match.group(1).strip()
            
            # Validar si realmente es una dirección
            validation = validate_and_improve_address(addr_text)
            if validation["is_valid"]:
                found_addresses.append({
                    "text": addr_text,
                    "position": match.start(),
                    "validation": validation
                })
    
    # PASO 2: Eliminar duplicados y ordenar por posición
    unique_addresses = []
    seen_texts = set()
    for addr in found_addresses:
        addr_normalized = strip_accents(addr["text"]).lower()
        if addr_normalized not in seen_texts:
            seen_texts.add(addr_normalized)
            unique_addresses.append(addr)
    
    # Ordenar por posición en el texto
    unique_addresses.sort(key=lambda x: x["position"])
    result["addresses_found"] = unique_addresses
    
    # PASO 3: Determinar origen y destino basado en contexto
    if len(unique_addresses) >= 2:
        # Si hay múltiples direcciones, usar heurísticas de contexto
        
        # Buscar indicadores de origen/destino cerca de cada dirección
        for i, addr in enumerate(unique_addresses):
            addr_pos = addr["position"]
            addr_text = addr["text"]
            
            # Analizar contexto antes y después de la dirección
            context_before = text[max(0, addr_pos-50):addr_pos].lower()
            context_after = text[addr_pos:addr_pos+len(addr_text)+50].lower()
            
            # Puntuación para origen vs destino
            origin_score = 0
            destination_score = 0
            
            # Indicadores de origen
            if any(word in context_before for word in ["desde", "salgo", "parto", "origen"]):
                origin_score += 3
            if i == 0:  # Primera dirección mencionada suele ser origen
                origin_score += 1
            
            # Indicadores de destino
            if any(word in context_after for word in ["hasta", "hacia", "destino", "dirección es", "la dirección es"]):
                destination_score += 3
            if any(word in context_before for word in ["hasta", "hacia"]):
                destination_score += 2
            if "dirección es" in context_after or "la dirección es" in context_after:
                destination_score += 5  # "la dirección es" fuertemente indica destino final
            if i == len(unique_addresses) - 1:  # Última dirección suele ser destino
                destination_score += 1
            
            # NUEVA HEURÍSTICA: Si la dirección aparece después de "dirección es"
            direction_pattern = r"direcci[oó]n\s+es\s+.*?" + re.escape(addr_text)
            if re.search(direction_pattern, text, re.IGNORECASE):
                destination_score += 8  # Puntuación muy alta para direcciones después de "dirección es"
            
            # Asignar rol basado en puntuación
            addr["origin_score"] = origin_score
            addr["destination_score"] = destination_score
    
    # PASO 3.1: Asignar roles de manera inteligente
    if len(unique_addresses) >= 2:
        # Con múltiples direcciones, asegurar que haya al menos un origen y un destino
        
        # Ordenar por puntuación de origen (descendente)
        by_origin_score = sorted(unique_addresses, key=lambda x: x.get("origin_score", 0), reverse=True)
        # Ordenar por puntuación de destino (descendente)  
        by_dest_score = sorted(unique_addresses, key=lambda x: x.get("destination_score", 0), reverse=True)
        
        # Elegir el mejor candidato para origen
        best_origin_candidate = by_origin_score[0]
        # Elegir el mejor candidato para destino (que no sea el mismo que origen)
        best_dest_candidate = None
        for candidate in by_dest_score:
            if candidate["text"] != best_origin_candidate["text"]:
                best_dest_candidate = candidate
                break
        
        # Si aún no hay destino diferente, buscar direcciones diferentes por contenido
        if not best_dest_candidate:
            for candidate in unique_addresses:
                if candidate != best_origin_candidate and candidate["text"] not in best_origin_candidate["text"]:
                    best_dest_candidate = candidate
                    break
        
        # Preferir direcciones más completas/largas para destino si tienen puntuación similar
        if best_dest_candidate:
            for candidate in by_dest_score[1:]:  # Revisar otros candidatos
                if (candidate["text"] != best_origin_candidate["text"] and 
                    candidate["text"] not in best_origin_candidate["text"] and
                    len(candidate["text"]) > len(best_dest_candidate["text"]) and
                    abs(candidate.get("destination_score", 0) - best_dest_candidate.get("destination_score", 0)) <= 2):
                    best_dest_candidate = candidate
                    break
        
        # Si no hay un destino diferente, usar criterio de posición
        if not best_dest_candidate:
            # Usar primera y última dirección
            if len(unique_addresses) >= 2:
                result["origin_candidates"].append(unique_addresses[0])
                result["destination_candidates"].append(unique_addresses[-1])
        else:
            result["origin_candidates"].append(best_origin_candidate)
            result["destination_candidates"].append(best_dest_candidate)
    
    elif len(unique_addresses) == 1:
        # Una sola dirección - determinar si es origen o destino por contexto
        addr = unique_addresses[0]
        context = text.lower()
        
        if any(word in context for word in ["desde", "salgo", "parto", "origen"]):
            result["origin_candidates"].append(addr)
        elif any(word in context for word in ["hasta", "hacia", "destino", "voy"]):
            result["destination_candidates"].append(addr)
        else:
            # Por defecto, si no hay contexto claro, podría ser cualquiera
            result["origin_candidates"].append(addr)
    
    # PASO 4: Seleccionar mejores candidatos
    if result["origin_candidates"]:
        # Elegir el mejor candidato para origen (mayor puntuación)
        best_origin = max(result["origin_candidates"], key=lambda x: x.get("origin_score", 0))
        result["best_origin"] = best_origin
    
    if result["destination_candidates"]:
        # Elegir el mejor candidato para destino (mayor puntuación)
        best_dest = max(result["destination_candidates"], key=lambda x: x.get("destination_score", 0))
        result["best_destination"] = best_dest
    
    return result


__all__ = [
    "titlecase_clean",
    "strip_accents",
    "sanitize_print",
    "clean_location_noise",
    "normalize_place",
    "is_specific_address",
    "normalize_regreso",
    "contains_time_hint",
    "mentions_field_change",
    "normalize_rut_value",
    "extract_rut_value",
    "extract_email_value",
    "extract_name_value",
    "extract_structured_pairs",
    "normalize_for_intent",
    "extract_structured_pairs_from_comment",
    "build_display_state",
    "print_bot_turn",
    "extract_quantity",
    "normalize_fecha",
    "user_mentions_date",
    "user_mentions_time",
    "user_mentions_regreso",
    "extract_location",
    "extract_origin_dest_pair",
    "parse_numeric_date",
    "parse_textual_date",
    "parse_time_from_text",
    "build_fallback_question",
    "select_question_fields",
    "question_mentions_forbidden",
    "validate_and_improve_address",
    "extract_and_validate_datetime_info",
    "analyze_user_response_context",
    "is_libpostal_available",
    "is_likely_chilean_location",
    "is_explicit_personal_introduction",
    "is_just_city_name",
    "is_complete_address_manual",
    "extract_multiple_addresses_from_complex_phrase",
]
