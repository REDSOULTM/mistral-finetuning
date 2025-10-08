"""Implementación del asesor conversacional que guía la cotización."""

from __future__ import annotations

import json
import random
import re
import unicodedata
from typing import Callable, Iterable, Sequence

import torch

from .direcciones import detect_and_extract_address
from .configuracion import (
    COMMENT_FIELD,
    ENABLE_PERSONAL_REGISTRATION,
    LLM_SAMPLING_KWARGS,
    PERSONAL_FIELDS,
    STATE_FIELDS,
    TRIP_FIELDS,
    WELCOME_MESSAGE,
)
from .constantes import (
    FALLBACK_FIELD_PROMPTS,
    LOCATION_EXCLUDE,
    NEGATIVE_COMMENT_VALUES,
    NEGATIVE_COTIZ_RE,
    NEGATIVE_TRAVEL_INTENT_RE,
    ORIGIN_EXPLICIT_RE as _ORIGIN_EXPLICIT_RE,
    DESTINATION_EXPLICIT_RE as _DESTINATION_EXPLICIT_RE,
    QUESTION_VARIATION_HINTS,
)
from .cotizacion import guardar_cotizacion
from .utilidades import (
    build_display_state as _build_display_state,
    build_fallback_question as _build_fallback_question,
    contains_time_hint as _contains_time_hint,
    extract_email_value,
    extract_location,
    extract_name_value,
    extract_origin_dest_pair,
    extract_quantity,
    extract_rut_value,
    extract_structured_pairs as _extract_structured_pairs,
    is_specific_address,
    normalize_for_intent as _normalize_for_intent,
    normalize_fecha,
    normalize_place,
    normalize_regreso,
    normalize_rut_value,
    parse_numeric_date as _parse_numeric_date,
    parse_textual_date as _parse_textual_date,
    parse_time_from_text as _parse_time_from_text,
    sanitize_print as _sanitize_print,
    strip_accents as _strip_accents,
    titlecase_clean as _titlecase_clean,
    select_question_fields as _select_question_fields,
    question_mentions_forbidden as _question_mentions_forbidden,
    user_mentions_date,
    user_mentions_regreso,
    user_mentions_time,
)

BASE_STYLE = """Transportes Miramar: vendedor para cotizar viajes en Chile. Respuestas breves estilo WhatsApp."""


class MiramarSellerBot:
    def __init__(self, model, tokenizer, device="cpu", use_kv_cache=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_kv_cache = use_kv_cache
        self.reset()

    def reset(self):
        self.state = {
            "nombre": "",
            "rut": "",
            "correo": "",
            "origen": "",
            "destino": "",
            "fecha": "",
            "hora": "",
            "regreso": "",
            "cantidad": 0,
            COMMENT_FIELD: "",
        }
        self.step = 0
        self.saved = False
        self.pending_detail_fields: set[str] = set()
        self.last_question: str = ""

    @staticmethod
    def _clean_generated_message(raw: str, expect_question: bool = False) -> str:
        if not raw:
            return ""

        prefixes = (
            "tarea:",
            "reglas:",
            "instrucciones:",
            "mensaje del usuario",
            "respuesta del cliente:",
            "último mensaje del cliente",
            "ultimo mensaje del cliente",
            "mensaje reciente del cliente",
            "entrega:",
            "contexto",
            "guía de estilo:",
            "guia de estilo:",
            "mensaje:",
            "rol:",
        )

        filtered: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            for prefix in prefixes:
                if lowered.startswith(prefix):
                    _, _, remainder = stripped.partition(":")
                    stripped = remainder.strip()
                    lowered = stripped.lower()
                    break

            stripped = re.sub(
                r"^(?:pregunta|confirmaci[oó]n|respuesta|mensaje|saludo|texto|salida|nota|instrucci[oó]n|recordatorio|devuelve)[:\.]?\s*",
                "",
                stripped,
                flags=re.IGNORECASE,
            )
            stripped = re.sub(
                r"^(?:pregunta redactada|mensaje redactado|respuesta generada)[:\.]?\s*",
                "",
                stripped,
                flags=re.IGNORECASE,
            )
            stripped = re.sub(
                r"^(?:responde|responder|contesta|contestar|indica|proporciona)\s+[^.?!]*[.:]\s*",
                "",
                stripped,
                flags=re.IGNORECASE,
            )

            if expect_question and "¿" in stripped:
                stripped = stripped[stripped.find("¿") :]

            stripped = stripped.strip()
            if not stripped:
                continue

            filtered.append(stripped)

        cleaned = re.sub(r"\s+", " ", " ".join(filtered)).strip()

        if expect_question and cleaned:
            matches = re.findall(r"¿[^?]*\?", cleaned)
            if matches:
                cleaned = matches[-1].strip()

        lowered_clean = cleaned.lower()
        if "tus datos son" in lowered_clean and "cantidad" in lowered_clean:
            cleaned = ""

        return cleaned

    @staticmethod
    def _sanitize_llm_field_value(value: str) -> str:
        cleaned = unicodedata.normalize("NFKC", value or "")
        cleaned = cleaned.replace("\n", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\bjson\s*:?\b.*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(entrega|respuesta|salida|mensaje|tarea)\s*:?\b.*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip("-:;., ")
        return cleaned

    @staticmethod
    def _coerce_fields(data: dict) -> dict:
        output: dict[str, object] = {}
        if not isinstance(data, dict):
            return output
        nombre = data.get("nombre")
        if isinstance(nombre, str) and nombre.strip():
            output["nombre"] = _titlecase_clean(nombre)
        correo = data.get("correo")
        if isinstance(correo, str) and correo.strip():
            output["correo"] = correo.strip().lower()
        rut = data.get("rut")
        if isinstance(rut, str) and rut.strip():
            normalized_rut = normalize_rut_value(rut)
            if normalized_rut:
                output["rut"] = normalized_rut
        fecha = data.get("fecha")
        if isinstance(fecha, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", fecha.strip()):
            output["fecha"] = fecha.strip()
        hora = data.get("hora")
        if isinstance(hora, str) and re.fullmatch(r"[0-2]\d:[0-5]\d", hora.strip()):
            output["hora"] = hora.strip()
        cantidad = data.get("cantidad")
        if cantidad is not None:
            try:
                n = int(cantidad)
                if n > 0:
                    output["cantidad"] = n
            except Exception:
                pass
        for key in ("origen", "destino", COMMENT_FIELD):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                output[key] = val.strip()
        if "regreso" in data:
            reg = normalize_regreso(str(data["regreso"]))
            if reg:
                output["regreso"] = reg
        return output

    @staticmethod
    def _describe_address_fields(fields: set[str]) -> str:
        if {"origen", "destino"}.issubset(fields):
            return "del punto de partida y del destino"
        if "origen" in fields:
            return "del punto de partida"
        if "destino" in fields:
            return "del destino"
        return "de la dirección"

    def generate_registration_question(self, missing_fields: list[str]) -> str:
        label_map = {
            "nombre": "tu nombre completo",
            "rut": "tu RUT",
            "correo": "tu correo electrónico",
        }
        tokens_map = {
            "nombre": ["nombre"],
            "rut": ["rut"],
            "correo": ["correo", "email", "mail"],
        }
        personal_state = {field: self.state.get(field, "") for field in PERSONAL_FIELDS}
        labels = [label_map[field] for field in missing_fields if field in label_map]
        if len(labels) >= 2:
            requested = ", ".join(labels[:-1]) + f" y {labels[-1]}"
        elif labels:
            requested = labels[0]
        else:
            requested = "los datos"

        prompt = (
            BASE_STYLE
            + f"""
Contexto del registro: {personal_state}
Campos personales pendientes: {missing_fields}
Tarea: formula una única pregunta cordial para solicitar {requested} antes de continuar con la cotización.
Reglas:
- Usa una sola oración clara en tono profesional y cercano.
- No añadas saludos ni despedidas, empieza directamente con la pregunta.
- Menciona explícitamente cada dato pendiente.
- Termina con signo de interrogación.
Entrega: <q>…</q>

<q>
"""
        )

        def _covers(message: str) -> bool:
            lowered = _strip_accents(message).lower()
            for field in missing_fields:
                options = tokens_map.get(field, [field])
                if not any(token in lowered for token in options):
                    return False
            return message.strip().endswith("?")

        question = self._generate_tagged_message(
            prompt,
            tag="q",
            extra_suffixes=[
                "\nReformula la pregunta asegurando que se pidan explícitamente todos los datos pendientes en una sola oración.\n<q>",
            ],
            validators=[_covers, lambda m: len(m.split()) <= 40],
            max_new_tokens=100,
        )
        if question:
            return question

        simple_map = {
            "nombre": "tu nombre completo",
            "rut": "tu RUT",
            "correo": "tu correo electrónico",
        }
        pieces = [simple_map[field] for field in missing_fields if field in simple_map]
        if not pieces:
            pieces = ["los datos de registro"]
        if len(pieces) > 1:
            body = ", ".join(pieces[:-1]) + f" y {pieces[-1]}"
        else:
            body = pieces[0]
            
        # ELIMINADO: _force_llm_exact - ahora 100% LLM
        final_prompt = (
            BASE_STYLE + f"""
Tarea: pregunta cordial pidiendo {body}.
<q>
"""
        )
        final_question = self._generate_tagged_message(
            final_prompt,
            tag="q",
            max_new_tokens=100,
        )
        return final_question or f"¿Podrías confirmarme {body}?"

    def generate_address_detail_request(self, fields: set[str], user_input: str | None = None) -> str:
        descriptor = self._describe_address_fields(fields)
        latest_message = _sanitize_print(user_input or "")
        meta = self.state.get("_address_meta", {})
        field_hints = {}
        for field in fields:
            data = meta.get(field, {}) if isinstance(meta, dict) else {}
            hint = (data.get("value") or data.get("partial") or data.get("raw")) if isinstance(data, dict) else None
            if hint:
                field_hints[field] = hint
        hints_text = json.dumps(field_hints, ensure_ascii=False) if field_hints else "{}"
        def _avoid_repeat(message: str) -> bool:
            lowered = _strip_accents(message).lower()
            if lowered.count("por favor") > 1:
                return False
            sentences = [segment.strip() for segment in re.split(r"[.!?]+", lowered) if segment.strip()]
            return len(sentences) == 1
        prompt = (
            BASE_STYLE
            + f"""
Contexto del viaje registrado: {self.state}
Último mensaje del cliente: {latest_message}
Referencias previas de direcciones proporcionadas por el cliente: {hints_text}
Tarea: redacta un mensaje breve solicitando la dirección completa {descriptor}, retomando lo que comentó el cliente.
Reglas:
- Usa tono cordial tipo WhatsApp en una sola oración.
- Menciona literalmente la expresión "dirección completa".
- Haz referencia al mensaje del cliente sin repetirlo textualmente.
- Limítate a UNA sola frase de máximo 22 palabras y no repitas la misma idea.
- No añadas preguntas nuevas ni confirmaciones; solo pide el dato pendiente.
- Sin etiquetas ni instrucciones adicionales.
Entrega: <msg>…</msg>

<msg>
"""
        )
        message = self._generate_tagged_message(
            prompt,
            tag="msg",
            extra_suffixes=[
                "\nSi la frase se ve repetitiva, reformúlala usando sinónimos y menciona la dirección completa con tus propias palabras.\n<msg>",
            ],
            validators=[
                lambda m: "dirección completa" in m.lower(),
                _avoid_repeat,
            ],
            max_new_tokens=100,
        )
        if not message:
            fallback_prompt = (
                BASE_STYLE
                + f"""
Contexto del viaje registrado: {self.state}
Último mensaje del cliente: {latest_message}
Tarea: pide amablemente la dirección completa {descriptor} en una sola frase cordial, aludiendo al mensaje recibido.
Entrega: <msg>…</msg>

<msg>
"""
            )
            message = self._generate_tagged_message(
                fallback_prompt,
                tag="msg",
                validators=[
                    lambda m: "dirección completa" in m.lower(),
                    _avoid_repeat,
                ],
                max_new_tokens=100,
            )
        if not message:
            lugares: list[str] = []
            if "origen" in fields:
                origen_txt = self.state.get("origen") or "tu punto de partida"
                lugares.append(f"de {origen_txt}")
            if "destino" in fields:
                destino_txt = self.state.get("destino") or "tu destino"
                lugares.append(f"de {destino_txt}")
            if not lugares:
                lugares.append("del viaje")
            joined = " y ".join(lugares)
            fallback_text = f"¿Me compartes la dirección completa {joined} para continuar con la cotización?"
            return self._force_llm_exact(fallback_text, tag="msg")
        return message

    def generate_final_message(self) -> str:
        prompt = (
            BASE_STYLE
            + f"""
Contexto final del viaje: {self.state}
Tarea: redacta un único mensaje de cierre agradeciendo y avisando que revisaremos la solicitud pronto.
Reglas:
- Una sola oración cordial.
- Sin preguntas ni listas.
- Menciona explícitamente a Transportes Miramar.
- Si ya conoces algún dato clave (origen, destino o fecha), inclúyelo de forma natural en la frase.
- Si algún dato falta, omítelo; nunca uses marcadores como [origen] o [destino].
- No agregues instrucciones ni etiquetas.
Entrega: <msg>…</msg>

<msg>
"""
        )
        return self._generate_tagged_message(
            prompt,
            tag="msg",
            validators=[
                lambda m: "?" not in m and "¿" not in m,
                lambda m: "transportes miramar" in m.lower(),
                lambda m: "[" not in m and "]" not in m,
                lambda m, state=self.state: all(
                    (not value) or (value.lower() in _strip_accents(m).lower())
                    for value in [
                        str(state.get("origen", "")).strip() or "",
                        str(state.get("destino", "")).strip() or "",
                        str(state.get("fecha", "")).strip() or "",
                        str(state.get("hora", "")).strip() or "",
                    ]
                    if value
                ),
            ],
            max_new_tokens=80,
        )
        
        # ELIMINADO: fallback hardcodeado - ahora 100% LLM
        # Si el LLM no genera nada, reintentamos con prompt más simple
        if not result:
            simple_prompt = (
                BASE_STYLE + f"""
Estado final: {self.state}
Tarea: mensaje breve de cierre mencionando Transportes Miramar.
<msg>
"""
            )
            result = self._generate_tagged_message(
                simple_prompt,
                tag="msg",
                max_new_tokens=100,
            )
        
        return result or "Gracias por contactar Transportes Miramar."

    def _build_final_message_fallback(self) -> str:
        origen = self.state.get("origen") or "tu punto de partida"
        destino = self.state.get("destino") or "tu destino"
        fecha = self.state.get("fecha")
        hora = self.state.get("hora")
        regreso = self.state.get("regreso")
        cantidad = self.state.get("cantidad")
        nombre = self.state.get("nombre")
        if nombre:
            fragments = [f"{nombre}, registré tu traslado desde {origen} hasta {destino}"]
        else:
            fragments = [f"Registré tu traslado desde {origen} hasta {destino}"]
        if fecha:
            when = fecha
            if hora:
                when += f" a las {hora}"
            fragments.append(f"para el {when}")
        elif hora:
            fragments.append(f"con salida a las {hora}")
        if cantidad:
            fragments.append(f"para {cantidad} pasajeros")
        if regreso:
            fragments.append("con regreso" if regreso == "sí" else "solo ida")
        fragments.append("Transportes Miramar revisará la solicitud en breve.")
        fallback_text = " ".join(fragments)
        if self.model is None or self.tokenizer is None:
            return fallback_text
        return self._force_llm_exact(fallback_text, tag="msg")

    def generate_already_registered_message(self) -> str:
        prompt = (
            BASE_STYLE
            + f"""
Contexto del viaje guardado: {self.state}
Tarea: informa al cliente que la solicitud ya está registrada y ofrece ayuda futura.
Reglas:
- Una sola oración cordial.
- No repitas datos ni hagas preguntas nuevas.
- Menciona que la solicitud ya está registrada.
- Si procede, recuerda de forma breve un dato clave del viaje (por ejemplo origen o destino) para que la respuesta suene personalizada.
Entrega: <msg>…</msg>

<msg>
"""
        )
        return self._generate_tagged_message(
            prompt,
            tag="msg",
            validators=[
                lambda m: "registr" in _strip_accents(m).lower(),
                lambda m: "?" not in m and "¿" not in m,
                lambda m: "transportes miramar" in _strip_accents(m).lower(),
            ],
            max_new_tokens=80,
        )

    def generate_empty_input_prompt(self) -> str:
        prompt = (
            BASE_STYLE
            + """
Tarea: cuando el cliente no envía mensaje, pide amablemente que escriba la información para continuar.
Reglas: una sola oración cordial, sin etiquetas ni instrucciones adicionales.
Entrega: <msg>…</msg>

<msg>
"""
        )
        return self._generate_tagged_message(
            prompt,
            tag="msg",
            validators=[lambda m: True],
            max_new_tokens=80,
        )

    def generate_decline_ack(self, user_input: str) -> str:
        latest_message = _sanitize_print(user_input)
        prompt = (
            BASE_STYLE
            + f"""
Tarea: cuando el cliente indica que no quiere cotizar, responde con una frase cordial dejando la puerta abierta.
Último mensaje del cliente: {latest_message}
Reglas:
- Una sola oración sin preguntas.
- Usa trato cercano (tú).
- Reconoce brevemente la decisión del cliente usando tus propias palabras.
- Indica que quedarás disponible por si cambia de opinión.
- Menciona a Transportes Miramar para reforzar quién responde.
Entrega: <msg>…</msg>

<msg>
"""
        )
        return self._generate_tagged_message(
            prompt,
            tag="msg",
            validators=[
                lambda m: "?" not in m and "¿" not in m,
                lambda m: "transportes miramar" in _strip_accents(m).lower(),
            ],
            max_new_tokens=80,
        )

    def generate_generic_help_message(self, user_input: str) -> str:
        lower_reference = _strip_accents(user_input or "").lower().strip()

        def _avoid_copy(message: str, reference: str = lower_reference) -> bool:
            if not reference or len(reference) < 3:
                return True
            cleaned = _strip_accents(message).lower().strip()
            return not cleaned.endswith(reference)

        normalized = (user_input or "").strip().lower()
        if re.fullmatch(r"hola[!¡\.\s]*", normalized) or normalized in {
            "buenas",
            "buenas tardes",
            "buenos días",
            "buenas noches",
        }:
            greeting_reply = self._generate_greeting_followup(user_input)
            if greeting_reply:
                return greeting_reply

        if re.search(r"qu[ií][eé]n(?:es)?\s+son|qu[ií][eé]n\s+es\s+miramar", normalized):
            company_reply = self._generate_company_intro_response(user_input)
            if company_reply:
                return company_reply

        if re.search(r"gracias|muchas gracias", normalized):
            thanks_reply = self._generate_thanks_response(user_input)
            if thanks_reply:
                return thanks_reply

        if re.search(r"sin\s+extras|sin\s+requerim|sin\s+comentario", normalized):
            no_extras_reply = self._generate_no_extras_response()
            if no_extras_reply:
                return no_extras_reply

        if re.search(r"que\s+me\s+puedes\s+ofrecer|que\s+puedes\s+hacer|que\s+ofrecen|que\s+pueden\s+ofrecer", normalized):
            services_reply = self._generate_services_response(user_input)
            if services_reply:
                return services_reply

        prompt = (
            BASE_STYLE
            + f"""
Mensaje reciente del cliente: {user_input}
Tarea: responde como un asesor humano de Transportes Miramar cuando el mensaje no es todavía una cotización clara.
Reglas:
- Usa una o dos oraciones breves con tono cercano y profesional.
- Reconoce brevemente lo que comentó el cliente (por ejemplo, saludar, pedir información, agradecer, etc.).
- Recuerda que Transportes Miramar es una empresa de transporte de pasajeros en Chile; menciona la empresa en la respuesta.
- Incluye literalmente el nombre Transportes Miramar para reforzar la identidad.
- Explica en pocas palabras cómo podemos ayudar (gestión de traslados y cotizaciones) y ofrece continuar si el cliente lo desea.
- Si el cliente hizo una pregunta concreta, respóndela directamente antes de invitar a cotizar.
- No enumeres listas ni repitas siempre la misma estructura; usa sinónimos y un estilo natural.
- Usa trato cercano (tú) y evita duplicar literalmente el mensaje del cliente al final.
- Si el cliente solo saluda o pide información general, devuélvele un saludo y explícale de forma breve cómo podemos ayudar, sin solicitar direcciones ni horarios todavía.
- Si pregunta quiénes somos, indícale que Transportes Miramar ofrece traslados privados de pasajeros dentro de Chile, sin inventar servicios adicionales ni compartir enlaces externos.
- Evita mencionar transporte aéreo o marítimo; enfócate en traslados terrestres.
Entrega: <msg>…</msg>

<msg>
"""
        )
        response = self._generate_tagged_message(
            prompt,
            tag="msg",
            validators=[
                lambda m: len(m.split()) <= 40,
                _avoid_copy,
                lambda m: "transportes miramar" in _strip_accents(m).lower(),
            ],
            max_new_tokens=80,
        )

        if response:
            fragments = re.split(r"(?<=[\.!?])\s+", response.strip())
            response = " ".join(fragments[:2]).strip()
        if response:
            return response

        strict_prompt = (
            BASE_STYLE
            + """
Tarea: devuelve exactamente este mensaje sobre nuestros servicios: "En Transportes Miramar coordinamos traslados privados de pasajeros dentro de Chile. Cuando quieras cotizar, cuéntame origen, destino, fecha, hora y cantidad de personas para ayudarte.".
Entrega: <msg>…</msg>

<msg>
"""
        )
        return self._generate_tagged_message(
            strict_prompt,
            tag="msg",
            validators=[lambda m: True],
            max_new_tokens=80,
        )

    def _generate_greeting_followup(self, user_input: str) -> str:
        def _no_detail_keywords(message: str) -> bool:
            lowered = _strip_accents(message).lower()
            forbidden = ["direccion", "dirección", "fecha", "hora", "cantidad", "regreso"]
            return all(token not in lowered for token in forbidden)

        prompt = (
            BASE_STYLE
            + f"""
Saludo recibido del cliente: {user_input}
Tarea: responde el saludo con un tono cálido, menciona que representas a Transportes Miramar y ofrece ayuda para cotizar traslados en Chile sin solicitar datos precisos todavía.
Reglas:
- Usa una o dos oraciones cortas.
- Emplea trato cercano (tú).
- No pidas direcciones ni horarios en esta respuesta.
- No repitas literalmente el saludo del cliente.
Entrega: <msg>…</msg>

<msg>
"""
        )
        response = self._generate_tagged_message(
            prompt,
            tag="msg",
            validators=[
                lambda m: len(m.split()) <= 40,
                _no_detail_keywords,
                lambda m: "transportes miramar" in _strip_accents(m).lower(),
            ],
            max_new_tokens=100,
        )
        if response:
            fragments = re.split(r"(?<=[\.!?])\s+", response.strip())
            response = " ".join(fragments[:2]).strip()
        if response:
            return response

        fallback_prompt = (
            BASE_STYLE
            + """
Tarea: responde al saludo del cliente con un mensaje breve de bienvenida en nombre de Transportes Miramar, sin pedir datos específicos todavía.
Entrega: <msg>…</msg>

<msg>
"""
        )
        fallback = self._generate_tagged_message(
            fallback_prompt,
            tag="msg",
            validators=[
                lambda m: len(m.split()) <= 40,
                _no_detail_keywords,
                lambda m: "transportes miramar" in _strip_accents(m).lower(),
            ],
            max_new_tokens=80,
        )
        if fallback:
            return fallback

        strict_prompt = (
            BASE_STYLE
            + """
Tarea: devuelve exactamente el siguiente mensaje de bienvenida, sin añadir ni quitar palabras.
Mensaje: "Hola, soy Transportes Miramar. Estoy aquí para ayudarte con tu traslado en Chile."
Entrega: <msg>…</msg>

<msg>
"""
        )
        return self._generate_tagged_message(
            strict_prompt,
            tag="msg",
            validators=[lambda m: m == WELCOME_MESSAGE],
            max_new_tokens=100,
        )

    def _generate_company_intro_response(self, user_input: str) -> str:
        def _no_forbidden_terms(message: str) -> bool:
            lowered = _strip_accents(message).lower()
            forbidden = ["http", "www", "aereo", "aéreo", "maritim", "barco", "avion", "avión"]
            return all(token not in lowered for token in forbidden)

        prompt = (
            BASE_STYLE
            + f"""
Pregunta del cliente sobre la empresa: {user_input}
Tarea: explica quién es Transportes Miramar y qué servicios ofrece.
Reglas:
- Usa una o dos oraciones claras.
- Indica que Transportes Miramar ofrece traslados privados de pasajeros dentro de Chile.
- No menciones transporte aéreo ni marítimo, ni enlaces externos.
- Invita a continuar con la cotización de manera cordial.
- Usa trato cercano (tú).
Entrega: <msg>…</msg>

<msg>
"""
        )
        response = self._generate_tagged_message(
            prompt,
            tag="msg",
            validators=[
                lambda m: len(m.split()) <= 40,
                lambda m: _no_forbidden_terms(m) and "transportes miramar" in _strip_accents(m).lower() and "traslad" in _strip_accents(m).lower(),
            ],
            max_new_tokens=100,
        )
        if response:
            fragments = re.split(r"(?<=[\.!?])\s+", response.strip())
            response = " ".join(fragments[:2]).strip()
        if response:
            return response

        retry_prompt = (
            BASE_STYLE
            + """
Tarea: describe brevemente a Transportes Miramar como empresa de traslados privados de pasajeros en Chile. No menciones enlaces ni transporte aéreo o marítimo. Invita cordialmente a cotizar.
Entrega: <msg>…</msg>

<msg>
"""
        )
        response = self._generate_tagged_message(
            retry_prompt,
            tag="msg",
            validators=[
                lambda m: len(m.split()) <= 40,
                lambda m: _no_forbidden_terms(m) and "transportes miramar" in _strip_accents(m).lower() and "traslad" in _strip_accents(m).lower(),
            ],
            max_new_tokens=80,
        )
        if response:
            fragments = re.split(r"(?<=[\.!?])\s+", response.strip())
            response = " ".join(fragments[:2]).strip()
        if response:
            return response

        strict_prompt = (
            BASE_STYLE
            + """
Tarea: devuelve exactamente este mensaje sin modificaciones: "Somos Transportes Miramar, una empresa de traslados privados de pasajeros en Chile. Cuéntame qué viaje necesitas y preparamos tu cotización.".
Entrega: <msg>…</msg>

<msg>
"""
        )
        return self._generate_tagged_message(
            strict_prompt,
            tag="msg",
            validators=[lambda m: True],
            max_new_tokens=80,
        )

    def _generate_services_response(self, user_input: str) -> str:
        prompt = (
            BASE_STYLE
            + f"""
Consulta del cliente: {user_input}
Tarea: explica brevemente qué servicios ofrece Transportes Miramar.
Reglas:
- Usa una o dos oraciones con trato cercano (tú).
- Menciona que organizamos traslados privados de pasajeros dentro de Chile, tanto urbanos como interurbanos.
- Invita a compartir origen, destino, fecha, hora y cantidad de pasajeros cuando el cliente quiera cotizar.
- No menciones enlaces ni servicios que no sean transporte terrestre.
Entrega: <msg>…</msg>

<msg>
"""
        )
        response = self._generate_tagged_message(
            prompt,
            tag="msg",
            validators=[
                lambda m: len(m.split()) <= 40,
                lambda m: "traslad" in _strip_accents(m).lower() and "transportes miramar" in _strip_accents(m).lower(),
            ],
            max_new_tokens=80,
        )
        if response:
            fragments = re.split(r"(?<=[\.!?])\s+", response.strip())
            response = " ".join(fragments[:2]).strip()
        return response

    def _generate_thanks_response(self, user_input: str) -> str:
        prompt = (
            BASE_STYLE
            + f"""
Mensaje de agradecimiento del cliente: {user_input}
Tarea: responde el agradecimiento manteniendo un tono cordial y recordando que podemos ayudar con cotizaciones.
Reglas:
- Usa una sola oración.
- Emplea trato cercano (tú).
- Menciona a Transportes Miramar y ofrece continuar si necesita algo más.
Entrega: <msg>…</msg>

<msg>
"""
        )
        response = self._generate_tagged_message(
            prompt,
            tag="msg",
            validators=[
                lambda m: len(m.split()) <= 40,
                lambda m: "transportes miramar" in _strip_accents(m).lower(),
            ],
            max_new_tokens=80,
        )
        if response:
            fragments = re.split(r"(?<=[\.!?])\s+", response.strip())
            response = " ".join(fragments[:2]).strip()
        if response:
            return response

        fallback_prompt = (
            BASE_STYLE
            + """
Tarea: Agradece el mensaje del cliente en una oración breve y recuérdale que Transportes Miramar está disponible si desea cotizar.
Entrega: <msg>…</msg>

<msg>
"""
        )
        response = self._generate_tagged_message(
            fallback_prompt,
            tag="msg",
            validators=[
                lambda m: len(m.split()) <= 40,
                lambda m: "transportes miramar" in _strip_accents(m).lower(),
            ],
            max_new_tokens=100,
        )
        if response:
            fragments = re.split(r"(?<=[\.!?])\s+", response.strip())
            response = " ".join(fragments[:2]).strip()
        if response:
            return response

        strict_prompt = (
            BASE_STYLE
            + """
Tarea: devuelve exactamente este mensaje de agradecimiento sin modificaciones: "Gracias a ti. Desde Transportes Miramar quedo atento si deseas cotizar otro traslado.".
Entrega: <msg>…</msg>

<msg>
"""
        )
        return self._generate_tagged_message(
            strict_prompt,
            tag="msg",
            validators=[lambda m: m == WELCOME_MESSAGE],
            max_new_tokens=100,
        )

    def _generate_no_extras_response(self) -> str:
        resumen = {
            "origen": self.state.get("origen", ""),
            "destino": self.state.get("destino", ""),
            "fecha": self.state.get("fecha", ""),
            "hora": self.state.get("hora", ""),
            "regreso": self.state.get("regreso", ""),
            "cantidad": self.state.get("cantidad", 0),
        }
        prompt = (
            BASE_STYLE
            + f"""
Tarea: agradece que el cliente confirme que no hay necesidades especiales y resume brevemente los datos del viaje.
Contexto actual: {resumen}
Reglas:
- Usa una o dos oraciones cortas con trato cercano (tú).
- Indica que no registras extras y que mantienes la cotización lista.
- No pidas nuevas direcciones ni enlaces.
Entrega: <msg>…</msg>

<msg>
"""
        )
        response = self._generate_tagged_message(
            prompt,
            tag="msg",
            validators=[lambda m: len(m.split()) <= 40],
            max_new_tokens=100,
        )
        if response:
            fragments = re.split(r"(?<=[\.!?])\s+", response.strip())
            response = " ".join(fragments[:2]).strip()
        if response:
            return response

        origen = resumen.get("origen") or "tu punto de partida"
        destino = resumen.get("destino") or "tu destino"
        fecha = resumen.get("fecha")
        hora = resumen.get("hora")
        regreso = resumen.get("regreso")
        cantidad = resumen.get("cantidad")
        parts = [f"Registré que desde {origen} hasta {destino} no hay extras pendientes."]
        if fecha:
            fecha_txt = fecha
            if hora:
                parts.append(f"La salida sigue agendada para el {fecha_txt} a las {hora}.")
            else:
                parts.append(f"La fecha prevista es el {fecha_txt}.")
        elif hora:
            parts.append(f"La salida sigue para las {hora}.")
        if cantidad:
            parts.append(f"Son {cantidad} pasajeros confirmados.")
        if regreso:
            parts.append(f"Anoté que el servicio es con regreso." if regreso == "sí" else "Anoté que el servicio es solo ida.")
        parts.append("Si necesitas otro detalle, cuéntame." )
        fallback_text = " ".join(parts)
        if self.model is None or self.tokenizer is None:
            return fallback_text
        return self._force_llm_exact(fallback_text, tag="msg")

    def generate_fallback_question(self, question_fields: list[str]) -> str:
        prompt = (
            BASE_STYLE
            + f"""
Contexto del viaje: {self.state}
Campos que aún faltan: {question_fields}
Tarea: formula una sola pregunta cordial solicitando exclusivamente esos campos pendientes.
Reglas:
- Usa tono WhatsApp, sin saludos ni despedidas.
- Menciona literalmente los nombres de los campos faltantes.
- Termina con signo de interrogación.
Entrega: <q>…</q>

<q>
"""
        )
        question = self._generate_tagged_message(
            prompt,
            tag="q",
            validators=[lambda m: m.strip().endswith("?")],
            max_new_tokens=80,
        )
        if question and not self._question_covers_fields(question, question_fields):
            question = ""
        if question:
            return question
            
        # ELIMINADO: fallback hardcodeado - ahora 100% LLM
        # Reintentamos con prompt más simple si el LLM falló
        simple_prompt = (
            BASE_STYLE + f"""
Campos faltantes: {question_fields}
Tarea: pregunta simple sobre estos campos.
<q>
"""
        )
        simple_question = self._generate_tagged_message(
            simple_prompt,
            tag="q",
            max_new_tokens=100,
        )
        
        return simple_question or "¿Podrías confirmarme los datos pendientes?"

    def generate_missing_data_prompt(self, question_fields: list[str]) -> str:
        prompt = (
            BASE_STYLE
            + f"""
Contexto del viaje: {self.state}
Campos pendientes: {question_fields}
Tarea: agradece brevemente e invita al cliente a compartir los campos pendientes sin formular una pregunta directa.
Reglas: una sola oración cordial, sin signos de interrogación.
Entrega: <msg>…</msg>

<msg>
"""
        )
        return self._generate_tagged_message(
            prompt,
            tag="msg",
            validators=[lambda m: "?" not in m and "¿" not in m],
            max_new_tokens=100,
        )

    @staticmethod
    def _build_fallback_question(missing: list[str]) -> str:
        ordered = list(STATE_FIELDS)
        missing_list = [item for item in ordered if item in missing]
        if not missing_list:
            return "¿Podrías confirmarme el dato que falta para continuar?"
        prompts: list[str] = []
        if "origen" in missing_list and "destino" in missing_list:
            prompts.append(random.choice(FALLBACK_FIELD_PROMPTS["origen_destino"]))
            missing_list = [m for m in missing_list if m not in {"origen", "destino"}]
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

    def generate_confirmation_fallback(self, actualizados: dict) -> str:
        prompt = (
            BASE_STYLE
            + f"""
Contexto del viaje: {self.state}
Datos a confirmar: {actualizados}
Tarea: redacta una oración breve confirmando únicamente estos datos nuevos.
Reglas: una sola oración, sin preguntas ni datos adicionales.
Entrega: <conf>…</conf>

<conf>
"""
        )
        return self._generate_tagged_message(
            prompt,
            tag="conf",
            validators=[
                lambda m: "?" not in m and "¿" not in m,
                lambda m: "*" not in m,
                lambda m, values=tuple(str(v).strip() for v in actualizados.values() if isinstance(v, str) and v.strip()): all(val and val.lower() in _strip_accents(m).lower() for val in values) if values else True,
            ],
            max_new_tokens=100,
        )
        
        # ELIMINADO: fallback hardcodeado - ahora 100% LLM
        # Si el LLM no genera confirmación, reintentamos con prompt simple
        if not result:
            simple_prompt = (
                BASE_STYLE + f"""
Datos nuevos: {actualizados}
Tarea: confirmación breve sin preguntas.
<conf>
"""
            )
            result = self._generate_tagged_message(
                simple_prompt,
                tag="conf",
                max_new_tokens=100,
            )
        
        return result or "Datos registrados."

    def _build_confirmation_fallback(self, actualizados: dict) -> str:
        parts: list[str] = []
        if "nombre" in actualizados:
            parts.append(f"nombre {actualizados['nombre']}")
        if "rut" in actualizados:
            parts.append(f"RUT {actualizados['rut']}")
        if "correo" in actualizados:
            parts.append(f"correo {actualizados['correo']}")
        if "fecha" in actualizados:
            parts.append(f"fecha {actualizados['fecha']}")
        if "hora" in actualizados:
            parts.append(f"hora {actualizados['hora']}")
        if "regreso" in actualizados:
            regreso_text = "con regreso" if actualizados["regreso"] == "sí" else "solo ida"
            parts.append(regreso_text)
        if "cantidad" in actualizados:
            parts.append(f"{actualizados['cantidad']} pasajeros")
        message = ", ".join(parts)
        if not message and actualizados:
            message = ", ".join(f"{k} {v}" for k, v in actualizados.items())
        fallback_text = (
            f"Perfecto, anoté {message}."
            if message
            else "Dejé registro de la actualización."
        )
        if self.model is None or self.tokenizer is None:
            return fallback_text
        return self._force_llm_exact(fallback_text, tag="conf")

    def generate_goodbye_message(self, user_input: str) -> str:
        farewell_input = _sanitize_print(user_input)
        prompt = (
            BASE_STYLE
            + f"""
Tarea: despídete cordialmente cuando el cliente cierra la conversación.
Último mensaje del cliente: {farewell_input}
Reglas:
- Una sola oración cordial, sin preguntas y mencionando que quedas atento.
- Usa trato cercano (tú).
- Agradece brevemente y ofrece ayuda futura si la necesita.
- Menciona a Transportes Miramar para mantener la identidad de la empresa.
Entrega: <msg>…</msg>

<msg>
"""
        )
        message = self._generate_tagged_message(
            prompt,
            tag="msg",
            validators=[
                lambda m: "?" not in m and "¿" not in m,
                lambda m: "transportes miramar" in _strip_accents(m).lower(),
            ],
            max_new_tokens=80,
        )
        if not message:
            fallback_prompt = (
                BASE_STYLE
                + f"""
Último mensaje del cliente: {farewell_input}
Tarea: despedida final en una frase cordial, indicando que quedas disponible si necesita otra cotización.
Entrega: <msg>…</msg>

<msg>
"""
            )
            message = self._generate_tagged_message(
                fallback_prompt,
                tag="msg",
                validators=[
                    lambda m: "?" not in m and "¿" not in m,
                    lambda m: "transportes miramar" in _strip_accents(m).lower(),
                ],
                max_new_tokens=80,
            )
        return message

    @staticmethod
    def _question_covers_fields(question: str, fields: list[str]) -> bool:
        lowered = _strip_accents(question).lower()
        field_keywords = {
            "origen": ["origen", "desde", "punto de partida"],
            "destino": ["destino", "hasta", "hacia", "llegada"],
            "fecha": ["fecha", "día", "dia", "cuándo", "cuando"],
            "hora": ["hora", "horario"],
            "regreso": ["regreso", "ida", "vuelta", "retorno"],
            "cantidad": ["personas", "pasajeros", "cuántas", "cuantas", "cuántos", "cuantos"],
        }
        for field in fields:
            keywords = field_keywords.get(field, [])
            if keywords and not any(token in lowered for token in keywords):
                return False
        return True

    @staticmethod
    def _parse_numeric_date(text: str) -> str | None:
        normalized = unicodedata.normalize("NFKC", text)
    def _fallback_extract_fields(self, user_input: str, existing: dict[str, object]) -> dict[str, object]:
        result: dict[str, object] = {}
        lowered_input = _strip_accents(user_input or "").lower()
        if ENABLE_PERSONAL_REGISTRATION:
            if "correo" not in existing and "correo" not in result:
                email = extract_email_value(user_input)
                if email:
                    result["correo"] = email
            if "rut" not in existing and "rut" not in result:
                rut_value = extract_rut_value(user_input)
                if rut_value:
                    result["rut"] = rut_value
            if "nombre" not in existing and "nombre" not in result:
                name_value = extract_name_value(user_input)
                if name_value:
                    result["nombre"] = name_value
        orig, dest = extract_origin_dest_pair(user_input)
        if orig and "origen" not in existing and is_specific_address(orig):
            result["origen"] = orig
        if "origen" not in result and "origen" not in existing:
            if any(token in lowered_input for token in ["desde", "salgo", "parto", "origen"]):
                single = extract_location(user_input, "origen")
                if single and is_specific_address(single):
                    result["origen"] = single
        if "origen" not in result and "origen" not in existing:
            explicit = _ORIGIN_EXPLICIT_RE.search(user_input)
            if explicit:
                val = normalize_place(explicit.group("orig"))
                if val and is_specific_address(val):
                    result["origen"] = val
        if dest and "destino" not in existing and is_specific_address(dest):
            result["destino"] = dest
        if "destino" not in result and "destino" not in existing:
            if any(token in lowered_input for token in ["hasta", "hacia", "destino", "rumbo"]):
                single_dest = extract_location(user_input, "destino")
                if single_dest and is_specific_address(single_dest):
                    result["destino"] = single_dest
        if "destino" not in result and "destino" not in existing:
            explicit_dest = _DESTINATION_EXPLICIT_RE.search(user_input)
            if explicit_dest:
                val = normalize_place(explicit_dest.group("dest"))
                if val and is_specific_address(val):
                    result["destino"] = val

        if "fecha" not in existing:
            date_value = _parse_numeric_date(user_input) or _parse_textual_date(user_input)
            if date_value:
                result["fecha"] = date_value

        if "hora" not in existing:
            time_value = _parse_time_from_text(user_input)
            if time_value:
                result["hora"] = time_value

        if "regreso" not in existing:
            reg_value = normalize_regreso(user_input)
            if reg_value:
                result["regreso"] = reg_value

        if "cantidad" not in existing:
            qty = extract_quantity(user_input)
            if qty:
                result["cantidad"] = qty

        if COMMENT_FIELD not in existing:
            negative_phrases = [
                "sin extras",
                "sin extra",
                "sin comentario",
                "sin requerimiento",
                "sin requerimientos",
                "sin adicional",
                "sin requerimientos especiales",
            ]
            if any(phrase in lowered_input for phrase in negative_phrases):
                result[COMMENT_FIELD] = "no"
            else:
                interest_tokens = {
                    "necesito",
                    "necesitamos",
                    "requiere",
                    "requieren",
                    "requiero",
                    "comentario",
                    "detalle",
                    "indicacion",
                    "indicaciones",
                    "preferencia",
                    "especial",
                    "equipaje",
                    "silla",
                    "mascota",
                }
                if any(token in lowered_input for token in interest_tokens):
                    result[COMMENT_FIELD] = user_input.strip()

        return result

    def _generate_tagged_message(
        self,
        base_prompt: str,
        tag: str = "msg",
        extra_suffixes: Sequence[str] | None = None,
        validators: Sequence[Callable[[str], bool]] | None = None,
        max_new_tokens: int = 200,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            return ""
        attempts: list[str] = [base_prompt]
        if extra_suffixes:
            attempts.extend(base_prompt + suffix for suffix in extra_suffixes)

        tag_pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)

        for prompt in attempts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                generate_kwargs.update(LLM_SAMPLING_KWARGS)
                outputs = self.model.generate(
                    **inputs,
                    **generate_kwargs,
                )
            text = MiramarSellerBot.decode_new_only(self.tokenizer, outputs, inputs)
            match = tag_pattern.search(text)
            if match:
                message = match.group(1)
            else:
                message = text
            message = re.sub(r"</?\w+>", "", message, flags=re.IGNORECASE).strip()
            message = self._clean_generated_message(message, expect_question=(tag == "q"))
            if not message:
                continue
            if validators and not all(validator(message) for validator in validators):
                continue
            return message
        return ""

    @staticmethod
    def decode_new_only(tokenizer, outputs, inputs):
        seq = outputs[0]
        prompt_len = inputs["input_ids"].shape[-1]
        if seq.shape[0] <= prompt_len:
            return tokenizer.decode(seq, skip_special_tokens=True).strip()
        new_tokens = seq[prompt_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if decoded:
            return decoded
        return tokenizer.decode(seq, skip_special_tokens=True).strip()

    def _force_llm_exact(self, text: str, tag: str = "msg") -> str:
        if not text:
            return ""
        if self.model is None or self.tokenizer is None:
            return text
        target = _sanitize_print(text)
        strict_prompt = (
            BASE_STYLE
            + f"""
Tarea: responde exactamente con el mensaje objetivo sin añadir ni quitar palabras.
Mensaje objetivo: {json.dumps(target, ensure_ascii=False)}
Entrega: <{tag}>…</{tag}>

<{tag}>
"""
        )
        result = self._generate_tagged_message(
            strict_prompt,
            tag=tag,
            validators=[lambda m: _sanitize_print(m) == target],
            max_new_tokens=max(60, len(target.split()) * 5 + 20),
        )
        return result or text

    @staticmethod
    def _looks_like_quote_request(text: str) -> bool:
        if not text:
            return False
        normalized = _normalize_for_intent(text)
        if not normalized:
            return False
        if NEGATIVE_TRAVEL_INTENT_RE.search(normalized):
            return False

        # Si menciona directamente cotizaciones o presupuestos, asumimos intención válida.
        if any(token in normalized for token in ["cotiz", "presupuest"]):
            return True

        # Detecta expresiones comunes de intención de viaje o traslado.
        intent_pairs = [
            r"\b(quiero|quisiera|necesito|necesitamos|busco|buscamos|planeo|planeamos|pienso|pensamos|deseo|deseamos|tengo\s+ganas|tenemos\s+ganas|me\s+gustaria|me\s+gustaria|me\s+interesa)\b.*\b(viaj|traslad|transport|ir|mover|salir|llevar|moviliz|irnos)",
            r"\b(vamos|iremos|iremos|vamos\s+a|planeamos\s+ir|organizamos|organizaremos|planeamos\s+un)\b.*\b(viaj|traslad|transport|salir|moviliz)",
            r"\b(solici[tm]o|requiero|podrian\s+cotizar|ayuda\s+con)\b.*\b(traslad|transport|viaj|transfer)",
        ]
        for pattern in intent_pairs:
            if re.search(pattern, normalized):
                return True

        travel_keywords = [
            "traslado",
            "traslad",
            "viaje",
            "viajar",
            "transport",
            "transfer",
            "moviliz",
            "llevar",
            "ir",
        ]
        if any(kw in normalized for kw in travel_keywords):
            if re.search(r"\b(desde|hasta|entre|a|hacia|origen|destino)\b", normalized):
                return True
            if re.search(r"\b(salida|llegada|recogida|vuelo|hotel|terminal)\b", normalized):
                return True

        if re.search(r"\b(\d{1,2}[/\-]\d{1,2}|\d{1,2}:\d{2})\b", normalized):
            if any(kw in normalized for kw in travel_keywords):
                return True

        if re.search(r"desde\s+.+?\s+hasta\s+.+", normalized):
            return True
        if re.search(r"\borigen\b", normalized) and re.search(r"\bdestino\b", normalized):
            return True
        if re.search(r"\b(origen|destino|ida\s+y\s+vuelta|solo\s+ida)\b", normalized):
            if any(kw in normalized for kw in travel_keywords):
                return True
        key_tokens = {"origen", "destino", "fecha", "hora", "regreso", "ida", "vuelta", "cantidad", "personas", "pasajer"}
        hits = sum(1 for token in key_tokens if token in normalized)
        if hits >= 2:
            return True
        return False

    def next_missing(self):
        for key in PERSONAL_FIELDS:
            if not self.state.get(key):
                return key
        for key in TRIP_FIELDS:
            if key == "cantidad":
                if self.state.get(key, 0) == 0:
                    return key
            elif not self.state.get(key):
                return key
        return None

    def validate_input_llm(self, user_input):
        if self.model is None or self.tokenizer is None:
            return {}
        
        # PROMPT ULTRA-COMPACTO para máxima velocidad (reducido de ~1500 a ~50 tokens)
        prompt = f"""Extraer datos de: "{user_input}"
JSON campos: nombre, rut, correo, origen, destino, fecha (YYYY-MM-DD), hora (HH:MM), regreso (sí/no), cantidad, comentario adicional.
Estado: {json.dumps(self.state, ensure_ascii=False)}
JSON:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,  # AUMENTADO para frases completas
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Procesar salida del modelo
        generated = MiramarSellerBot.decode_new_only(self.tokenizer, outputs, inputs)
        try:
            # Buscar JSON en la respuesta
            import re
            json_match = re.search(r'\{[^}]*\}', generated)
            if json_match:
                result = json.loads(json_match.group())
                return self._coerce_fields(result)
        except:
            pass
        
        return {}
    def generate_greeting_llm(self) -> str:
        if self.model is None or self.tokenizer is None:
            # ELIMINADO: WELCOME_MESSAGE hardcodeado - generar siempre con LLM
            simple_prompt = BASE_STYLE + "Tarea: saludo inicial breve de Transportes Miramar.\n<msg>"
            return self._generate_tagged_message(simple_prompt, tag="msg", max_new_tokens=80) or "Hola."
            
        if not ENABLE_PERSONAL_REGISTRATION:
            prompt = (
                BASE_STYLE
                + """
Tarea: genera el mensaje de bienvenida inicial para iniciar la cotización.
Reglas:
- Usa una o dos oraciones completas y naturales.
- Menciona explícitamente a Transportes Miramar.
- Explica en tono cordial que estás listo para ayudar con la cotización y pide que cuenten origen, destino, fecha, hora y cantidad cuando estén listos.
- No utilices listas ni emojis.
- IMPORTANTE: Termina siempre las frases de forma completa y natural.
Entrega: <msg>…</msg>
<msg>
"""
            )
            forbidden_phrase = "como me puedes ayudar"

            greeting = self._generate_tagged_message(
                prompt,
                tag="msg",
                extra_suffixes=[
                    "\nReescribe el saludo asegurando que mencione a Transportes Miramar, evite pedir ayuda al cliente y mantenga el límite de palabras.",
                ],
                validators=[
                    lambda m: len(m.split()) <= 30,  # AUMENTADO para permitir frases completas
                    lambda m: "transportes miramar" in _strip_accents(m).lower(),
                    lambda m, fp=forbidden_phrase: fp not in _strip_accents(m).lower(),
                    lambda m: all(ord(ch) <= 255 for ch in m),
                    lambda m: not m.endswith((' cu', ' co', ' con', ' nos', ' te', ' para', ' que')),  # Detectar cortes
                ],
                max_new_tokens=80,  # AUMENTADO significativamente para evitar cortes
            )
            if greeting:
                return greeting

            # ELIMINADO: fallback hardcodeado - siempre generar con LLM
            simple_greeting = self._generate_tagged_message(
                BASE_STYLE + "Tarea: saludo inicial cordial de Transportes Miramar.\n<msg>",
                tag="msg",
                max_new_tokens=100,
            )
            return simple_greeting or "Hola, soy Transportes Miramar."
            
        # Modo con registro personal - también 100% LLM
        prompt = (BASE_STYLE + """
Tarea: Genera el primer mensaje para saludar cordialmente al usuario y solicitar sus datos de registro.
Objetivo: dar la bienvenida, mencionar que representas a Transportes Miramar y pedir nombre completo, RUT y correo electrónico antes de continuar con la cotización.
Guía de estilo:
- Usa una o dos oraciones cálidas y profesionales (frases completas y naturales).
- Menciona explícitamente a Transportes Miramar.
- Termina con una pregunta que solicite esos tres datos.
- No utilices listas ni apartados.
- Evita repeticiones innecesarias y asegúrate de que la redacción sea natural.
Entrega: <msg>…</msg>
<msg>
""")
        greeting = self._generate_tagged_message(
            prompt,
            tag="msg",
            extra_suffixes=[
                "\nRecuerda cerrar con una pregunta clara que pida nombre completo, RUT y correo electrónico.",
                "\nSi la primera respuesta no cumple, entrega otra versión en dos oraciones con ese mismo pedido explícito.",
            ],
            validators=[
                lambda m: len(m.split()) <= 30,
                lambda m: m.strip().endswith("?"),
                lambda m: "transportes miramar" in _strip_accents(m).lower(),
                lambda m: all(keyword in _strip_accents(m).lower() for keyword in ["nombre", "rut", "correo"]),
            ],
            max_new_tokens=80,
        )
        if greeting:
            return greeting
            
        # ELIMINADO: todos los fallbacks hardcodeados - solo LLM
        simple_registration = self._generate_tagged_message(
            BASE_STYLE + "Tarea: pedir registro con nombre, RUT y correo de Transportes Miramar.\n<msg>",
            tag="msg",
            max_new_tokens=80
        )
        return simple_registration or "Hola, soy Transportes Miramar. ¿Cómo puedo ayudarte?"

    def initial_prompt(self) -> str:
        return self.generate_greeting_llm()

    def get_next_question(self):
        missing_personal = [field for field in PERSONAL_FIELDS if not self.state.get(field)]
        if missing_personal:
            pregunta_personal = self.generate_registration_question(missing_personal)
            if pregunta_personal:
                self.last_question = pregunta_personal.strip()
                return pregunta_personal

        missing = []
        for k in TRIP_FIELDS:
            if k == "cantidad":
                if self.state.get(k, 0) == 0:
                    missing.append(k)
            elif not self.state.get(k):
                missing.append(k)
        if not missing:
            return None
        question_fields = _select_question_fields(missing)
        if not question_fields:
            question_fields = missing[:2]
        # ELIMINADO: hardcode de pregunta regreso+cantidad - ahora 100% LLM
        confirmed = {
            k: self.state[k]
            for k in TRIP_FIELDS
            if (k == "cantidad" and self.state[k] > 0)
            or (k != "cantidad" and self.state[k])
        }
        previous_question = getattr(self, "last_question", "")
        style_hint = random.choice(QUESTION_VARIATION_HINTS)
        prompt = (
            BASE_STYLE
            + f"""
Contexto de la cotización (estado): {self.state}
Datos confirmados: {confirmed}
Campos faltantes para esta pregunta: {question_fields}
Última pregunta enviada al cliente: {previous_question or "N/A"}
Sugerencia de estilo para variar la redacción: {style_hint}
Tarea: escribe UNA sola pregunta que avance la conversación y pida únicamente esos campos listados.
Reglas clave:
- Expresa la pregunta en tono cordial tipo WhatsApp y en una sola oración.
- No incluyas saludos ni despedidas; comienza directamente con la pregunta.
- Si faltan origen y destino, menciónalos juntos en la misma frase usando formulaciones como "¿Desde qué lugar inicia tu viaje y cuál es el destino?".
- IMPORTANTE: Si detectas que en el mensaje anterior del cliente se mencionó solo un nombre de ciudad (ej: "Valparaíso", "Santiago", "Viña del Mar") pero aún falta origen o destino en los Campos faltantes, significa que necesitas la dirección completa. Pide EXPLÍCITAMENTE la dirección completa con calle y número. Por ejemplo: "Perfecto, ¿me puedes dar la dirección completa con calle y número en Valparaíso?" o "Entendido, ¿cuál es la dirección exacta con calle y número?".
- No uses ni insinúes expresiones como "¿dónde estás?", "¿en qué lugar te encuentras?" o "ubicación actual".
- Si un campo ya está completo en el estado, prohíbete mencionarlo o confirmarlo; enfócate solo en los elementos listados en Campos faltantes.
- No reformules ni retomes datos listados en Datos confirmados; asume que ya quedaron claros.
- CRÍTICO PARA "regreso": Si falta "regreso", pregunta PRIMERO si el viaje es solo ida o incluye regreso usando EXACTAMENTE esta forma: "¿El viaje es solo ida o también incluye regreso?". NUNCA asumas que hay regreso, NUNCA menciones "día de regreso" ni "fecha de regreso", NUNCA preguntes "cuándo regresas". Primero debes confirmar SI hay regreso, solo después de que el cliente confirme que SÍ hay regreso, podrás preguntar por detalles del regreso.
- Si faltan "regreso" y "cantidad", puedes agruparlos así: pregunta PRIMERO sobre el regreso como se indicó arriba, y luego la cantidad. Ejemplo: "¿El viaje es solo ida o también incluye regreso? ¿Y cuántas personas viajarán?".
- Pide como máximo DOS datos en la misma pregunta; si faltan más, prioriza según el orden del listado en Campos faltantes.
- Puedes agrupar hasta dos campos en la misma pregunta (por ejemplo fecha+hora o regreso+cantidad) si faltan ambos.
- Termina siempre con signo de interrogación de cierre.
- Mantén la redacción fluida, evita copiar literalmente instrucciones de este prompt y no repitas siempre la misma estructura.
- Evita repetir literalmente la última pregunta enviada; usa sinónimos o una estructura distinta.
Entrega: <q>…</q>

<q>
"""
        )
        pregunta = self._generate_question_with_prompt(prompt, question_fields)
        if pregunta and previous_question and pregunta.strip().lower() == previous_question.strip().lower():
            alt_hint = random.choice([hint for hint in QUESTION_VARIATION_HINTS if hint != style_hint] or QUESTION_VARIATION_HINTS)
            alt_prompt = (
                BASE_STYLE
                + f"""
Contexto de la cotización (estado): {self.state}
Datos confirmados: {confirmed}
Campos faltantes para esta pregunta: {question_fields}
Última pregunta enviada al cliente: {previous_question or "N/A"}
Sugerencia de estilo para variar la redacción: {alt_hint}
Tarea: redacta nuevamente una pregunta distinta a la anterior, manteniendo las reglas.
Entrega: <q>…</q>

<q>
"""
            )
            alternativa = self._generate_question_with_prompt(alt_prompt, question_fields)
            if alternativa:
                pregunta = alternativa
        if pregunta and pregunta.strip():
            self.last_question = pregunta.strip()
            return pregunta
        fallback_question = self.generate_fallback_question(question_fields)
        if fallback_question:
            self.last_question = fallback_question.strip()
            return fallback_question
        message_prompt = self.generate_missing_data_prompt(question_fields)
        if message_prompt:
            return message_prompt
        fallback_text = MiramarSellerBot._build_fallback_question(question_fields)
        if self.model is None or self.tokenizer is None:
            fallback = fallback_text
        else:
            fallback = self._force_llm_exact(fallback_text, tag="q")
        self.last_question = fallback.strip()
        return fallback

    def _generate_question_with_prompt(self, prompt: str, question_fields: list[str]) -> str | None:
        validators = [
            lambda m: not _question_mentions_forbidden(m, question_fields),
            lambda m: m.strip().endswith("?"),
            lambda m: not m.lower().startswith("tarea:"),
            lambda m: "instruccion" not in m.lower(),
            lambda m: len(m.split()) <= 40,
        ]
        extra_suffixes = [
            "\nRecuerda generar solo una pregunta clara dentro de <q>...</q>.",
            "\nImportante: acabas de mencionar datos ya confirmados. Reescribe la pregunta mencionando únicamente los campos listados en Campos faltantes.\n<q>",
        ]
        question = self._generate_tagged_message(
            prompt,
            tag="q",
            extra_suffixes=extra_suffixes,
            validators=validators,
            max_new_tokens=100,
        )
        if question and not self._question_covers_fields(question, question_fields):
            question = ""
        return question or None

    def _generate_confirmation_llm(self, actualizados: dict) -> str:
        prompt = (
            BASE_STYLE
            + f"""
Contexto del viaje: {self.state}
Datos nuevos que debes confirmar: {actualizados}
Tarea: redacta UNA sola frase breve confirmando únicamente estos datos nuevos, sin pedir información adicional.
Reglas:
- Menciona solo los campos incluidos en "Datos nuevos".
- Usa un tono cordial tipo WhatsApp.
- No repitas datos ya confirmados previamente ni añadas preguntas.
- No inventes información ni añadas campos extra.
- Entrega el mensaje envuelto en <conf>…</conf>.

<conf>
"""
        )
        return self._generate_tagged_message(
            prompt,
            tag="conf",
            extra_suffixes=[
                "\nRecuerda: confirma únicamente los datos nuevos en una sola oración de máximo 20 palabras.",
                "\nGenera una nueva confirmación directa, sin agradecimientos adicionales, mencionando solo los campos de 'Datos nuevos'.",
            ],
            validators=[
                lambda m: len(m.split()) <= 40,
                lambda m: "?" not in m and "¿" not in m,
                lambda m: "*" not in m,
                lambda m, values=tuple(str(v).strip() for v in actualizados.values() if isinstance(v, str) and v.strip()): all(val and val.lower() in _strip_accents(m).lower() for val in values) if values else True,
            ],
        )

    def update_state(self, user_input: str):
        # Solo buscar dirección si origen y destino ya están llenos
        if not self.state["origen"] or not self.state["destino"]:
            addr = None
        else:
            addr = detect_and_extract_address(user_input)
        if addr:
            # Decide dónde guardarla: primero origen, luego destino
            target = "origen" if not self.state["origen"] else ("destino" if not self.state["destino"] else None)
            if target:
                # Etiqueta legible para el chat
                label = (addr.get("street") or "")
                if addr.get("number"):
                    label += f" #{addr['number']}"
                if addr.get("comuna"):
                    label += f", {addr['comuna']}"
                label = label.strip(", ").strip() or addr.get("raw") or "Dirección registrada"

                # Guarda en estado + metadatos
                self.state[target] = label
                meta = self.state.setdefault("_address_meta", {}).setdefault(target, {})
                meta.update(addr)
                meta["precision"] = "high"

                # Confirmación SIEMPRE con el LLM
                actualizados = {target: self.state[target]}
                llm_conf = self._generate_confirmation_llm(actualizados)
                confirm_msg = llm_conf or self.generate_confirmation_fallback(actualizados)

                # Ahora dejamos que el LLM pregunte lo siguiente
                siguiente = self.next_missing()
                if siguiente:
                    pregunta = self.get_next_question()  # también LLM
                    return f"{confirm_msg}\n{pregunta}", True

                # Si ya está todo, guardamos y cerramos
                if not self.saved:
                    guardar_cotizacion(self.state)
                    self.saved = True
                cierre = self.generate_final_message() or ""
                if cierre:
                    return f"{confirm_msg}\n{cierre}", True
                return confirm_msg, True
        # --- si no hubo dirección detectada, sigue tu flujo actual ---
        raw_fields = self.validate_input_llm(user_input)
        if not ENABLE_PERSONAL_REGISTRATION:
            raw_fields = {
                k: v
                for k, v in raw_fields.items()
                if k not in {"nombre", "rut", "correo"}
            }
        new_fields: dict[str, object] = {}
        for key, value in raw_fields.items():
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (int, float)) and value <= 0:
                continue
            new_fields[key] = value
        fallback_existing = dict(new_fields)
        for place_key in ("origen", "destino"):
            val = fallback_existing.get(place_key)
            if isinstance(val, str):
                has_digits = any(ch.isdigit() for ch in val)
                if not is_specific_address(val) or not has_digits:
                    fallback_existing.pop(place_key, None)
        fallback_fields = self._fallback_extract_fields(user_input, fallback_existing)
        for key, value in fallback_fields.items():
            if key not in new_fields:
                new_fields[key] = value
                continue
            if key in {"origen", "destino"}:
                current = str(new_fields.get(key, ""))
                candidate = str(value)
                current_has_digits = any(ch.isdigit() for ch in current)
                candidate_has_digits = any(ch.isdigit() for ch in candidate)
                if (
                    (not is_specific_address(current) and is_specific_address(candidate))
                    or (candidate_has_digits and not current_has_digits)
                ):
                    new_fields[key] = value
        actualizados: dict[str, str | int] = {}
        pending_address_fields: set[str] = set()
        for key, value in new_fields.items():
            if key in PERSONAL_FIELDS:
                value_str = str(value).strip()
                if not value_str:
                    continue
                if key == "rut":
                    value_str = normalize_rut_value(value_str)
                    if not value_str:
                        continue
                elif key == "correo":
                    value_str = value_str.lower()
                else:
                    value_str = _titlecase_clean(value_str)
                if self.state.get(key) != value_str:
                    self.state[key] = value_str
                    actualizados[key] = value_str
                continue
            if key == "cantidad":
                value_int = extract_quantity(user_input)
                if value_int is None or value_int <= 0:
                    continue
                if self.state[key] != value_int:
                    self.state[key] = value_int
                    actualizados[key] = value_int
            elif key == COMMENT_FIELD:
                value_str = str(value).strip()
                handled_address = False
                if value_str:
                    structured_pairs = _extract_structured_pairs(value_str)
                    for pair_key, pair_value in structured_pairs.items():
                        if pair_key not in {"origen", "destino"}:
                            continue
                        clean_value = pair_value.strip()
                        if not is_specific_address(clean_value):
                            continue
                        previous = self.state.get(pair_key, "")
                        if previous != clean_value:
                            self.state[pair_key] = clean_value
                            actualizados[pair_key] = clean_value
                        meta = self.state.setdefault("_address_meta", {}).setdefault(pair_key, {})
                        meta.pop("partial", None)
                        meta["value"] = clean_value
                        meta["precision"] = "high"
                        self.pending_detail_fields.discard(pair_key)
                        handled_address = True
                    if not handled_address:
                        sub_orig, sub_dest = extract_origin_dest_pair(value_str)
                        if sub_orig and is_specific_address(sub_orig):
                            previous = self.state.get("origen", "")
                            if previous != sub_orig:
                                self.state["origen"] = sub_orig
                                actualizados["origen"] = sub_orig
                            meta = self.state.setdefault("_address_meta", {}).setdefault("origen", {})
                            meta.pop("partial", None)
                            meta["value"] = sub_orig
                            meta["precision"] = "high"
                            self.pending_detail_fields.discard("origen")
                            handled_address = True
                        if sub_dest and is_specific_address(sub_dest):
                            previous = self.state.get("destino", "")
                            if previous != sub_dest:
                                self.state["destino"] = sub_dest
                                actualizados["destino"] = sub_dest
                            meta = self.state.setdefault("_address_meta", {}).setdefault("destino", {})
                            meta.pop("partial", None)
                            meta["value"] = sub_dest
                            meta["precision"] = "high"
                            self.pending_detail_fields.discard("destino")
                            handled_address = True
                if handled_address:
                    self.pending_detail_fields.discard("origen")
                    self.pending_detail_fields.discard("destino")
                    continue
                if not value_str:
                    continue
                normalized_comment = value_str.lower()
                if normalized_comment in NEGATIVE_COMMENT_VALUES:
                    if self.state.get(key) != "no":
                        self.state[key] = "no"
                        actualizados[key] = "no"
                    continue
                if self.state.get(key) != value_str:
                    self.state[key] = value_str
                    actualizados[key] = value_str
                continue
            else:
                value_str = str(value).strip()
                if key in {"origen", "destino"}:
                    value_str = normalize_place(value_str)
                    if not value_str or value_str.lower() in LOCATION_EXCLUDE:
                        continue
                    normalized_value = _strip_accents(value_str).lower()
                    if normalized_value and normalized_value not in _strip_accents(user_input).lower():
                        continue
                    has_existing_value = bool(self.state.get(key))
                    meta_container = self.state.setdefault("_address_meta", {})
                    meta = meta_container.setdefault(key, {})
                    precision = meta.get("precision")
                    allow_update = (
                        not has_existing_value
                        or key in self.pending_detail_fields
                        or precision == "partial"
                    )
                    if has_existing_value and not allow_update:
                        continue
                    if not is_specific_address(value_str):
                        meta["partial"] = value_str
                        meta["precision"] = "partial"
                        meta.pop("value", None)
                        if self.state.get(key):
                            self.state[key] = ""
                        pending_address_fields.add(key)
                        continue
                    self.pending_detail_fields.discard(key)
                    meta.pop("partial", None)
                    meta["value"] = value_str
                    meta["precision"] = "high"
                if key == "fecha":
                    if not user_mentions_date(user_input):
                        continue
                    value_str = normalize_fecha(value_str, user_input)
                if key == "regreso":
                    if not user_mentions_regreso(user_input):
                        continue
                    normalized_regreso = normalize_regreso(value_str)
                    if not normalized_regreso:
                        continue
                    value_str = normalized_regreso
                if key == "hora":
                    if not user_mentions_time(user_input):
                        continue
                if key == COMMENT_FIELD and value_str.lower() in NEGATIVE_COMMENT_VALUES:
                    if self.state.get(key) != "no":
                        self.state[key] = "no"
                        actualizados[key] = "no"
                elif self.state[key] != value_str and value_str:
                    self.state[key] = value_str
                    actualizados[key] = value_str
        detail_message = ""
        if pending_address_fields:
            detail_message = self.generate_address_detail_request(pending_address_fields, user_input) or ""
            self.pending_detail_fields = pending_address_fields.copy()
        elif self.pending_detail_fields:
            self.pending_detail_fields.clear()
        if actualizados:
            llm_conf = self._generate_confirmation_llm(actualizados)
            confirm_msg = llm_conf or self.generate_confirmation_fallback(actualizados)
            if detail_message:
                return f"{confirm_msg}\n{detail_message}", True
            return confirm_msg, True
        if detail_message:
            return detail_message, False
        return ("", False)

    def run_step(self, user_input: str):
        user_input = user_input.strip()
        if not user_input:
            return self.generate_empty_input_prompt() or ""
        self.step += 1
        mensaje, hubo_cambios = self.update_state(user_input)
        needs_detail = bool(getattr(self, "pending_detail_fields", set()))
        siguiente = self.next_missing()
        final_msg: str | None = None
        if hubo_cambios:
            if needs_detail and mensaje:
                return mensaje
            if siguiente:
                pregunta = self.get_next_question()
                return f"{mensaje}\n{pregunta}"
            if not self.saved:
                guardar_cotizacion(self.state)
                self.saved = True
            final_msg = final_msg or self.generate_final_message() or ""
            return f"{mensaje}\n{final_msg}" if final_msg else mensaje
        if siguiente:
            if mensaje:
                if needs_detail:
                    return mensaje or (self.generate_address_detail_request(self.pending_detail_fields, user_input) or "")
                pregunta = self.get_next_question()
                return f"{mensaje}\n{pregunta}"
            return self.get_next_question()
        if not self.saved:
            guardar_cotizacion(self.state)
            self.saved = True
            if mensaje:
                final_msg = final_msg or self.generate_final_message() or ""
                return f"{mensaje}\n{final_msg}" if final_msg else mensaje
            return self.generate_final_message() or ""
        if mensaje:
            return mensaje
        return self.generate_already_registered_message() or ""
