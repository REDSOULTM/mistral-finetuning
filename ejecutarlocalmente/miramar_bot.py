# miramar_bot.py
from __future__ import annotations

import os
import sys
import time
import types
import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Iterable, Optional

# ----------------------------
# Hooks de optimización (inyectados desde el lanzador)
# ----------------------------
OPT_HOOKS: Dict[str, Any] = {}

def register_opt_hooks(hooks: Dict[str, Any]) -> None:
    """Permite al lanzador inyectar hooks opcionales de optimización."""
    OPT_HOOKS.update({k: v for k, v in hooks.items() if v is not None})


# ----------------------------
# Config básica
# ----------------------------
@dataclass
class ModelConfig:
    model_id: str
    backend: str            # "hf" | "vllm"
    dtype: str = "auto"     # "auto" | "bfloat16" | "float16"
    device: str = "cuda"    # "cuda" | "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    streaming: bool = True

# Ajusta aquí tus rutas/modelos
DEFAULTS = {
    "lora_4bit": ModelConfig(model_id="mistralai/Mistral-7B-Instruct-v0.2", backend="hf", dtype="bfloat16"),
    "awq_4b":    ModelConfig(model_id="TheBloke/Mistral-7B-Instruct-v0.2-AWQ", backend="vllm", dtype="bfloat16"),
    # fallback
    "default":   ModelConfig(model_id="mistralai/Mistral-7B-Instruct-v0.2", backend="hf", dtype="bfloat16"),
}


# ----------------------------
# Utilidades seguras para hooks
# ----------------------------
def _hook(name: str, default: Any = None) -> Any:
    return OPT_HOOKS.get(name, default)

@contextlib.contextmanager
def _inference_ctx():
    ctx = _hook("inference_context")
    if ctx:
        with ctx():
            yield
    else:
        yield

def _diet_prompt(text: str) -> str:
    dp = _hook("diet_prompt")
    return dp(text) if dp else text

def _pretokenize_template(system: str, up: str, asst: str, sep: str = "\n"):
    fn = _hook("pretokenize_template")
    if fn:
        return fn(system, up, asst, sep)
    # fallback
    tpl = sep.join([system, up, asst]).strip()
    return system, up, asst, tpl

def _tune_tokenizer(tok: Any) -> None:
    tune = _hook("tune_tokenizer")
    if tune:
        try:
            tune(tok)
        except Exception:
            pass

def _transformers_defaults(model: Any) -> None:
    fn = _hook("transformers_generate_defaults")
    if fn:
        try:
            fn(model)
        except Exception:
            pass

def _compile_model(model: Any) -> Any:
    fn = _hook("compile_model")
    if fn:
        try:
            return fn(model)
        except Exception:
            return model
    return model

def _apply_vllm_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    fn = _hook("apply_vllm_kwargs")
    return fn(kwargs) if fn else kwargs


# ----------------------------
# Backend: HuggingFace Transformers
# ----------------------------
class HFEngine:
    def __init__(self, cfg: ModelConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        _tune_tokenizer(self.tok)

        load_kwargs: Dict[str, Any] = {"device_map": "auto"}
        if cfg.dtype == "bfloat16":
            load_kwargs["torch_dtype"] = "bfloat16"
        elif cfg.dtype == "float16":
            load_kwargs["torch_dtype"] = "float16"

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **load_kwargs)
        _transformers_defaults(self.model)
        self.model = _compile_model(self.model)  # no-op si no hay compile

        # Warm-up: compila/traza kernels y reduce TTFB
        self._warmup()

    def _warmup(self) -> None:
        try:
            with _inference_ctx():
                _ = self.generate("Ok.", max_new_tokens=1, temperature=0.0, stream=False)
        except Exception:
            pass

    def _build_inputs(self, system: str, user: str) -> Dict[str, Any]:
        # Limpia y pretokeniza
        system = _diet_prompt(system)
        user = _diet_prompt(user)
        sys_clean, up_clean, as_clean, tpl = _pretokenize_template(system, "Usuario:", "Asistente:")
        prompt = f"{sys_clean}\nUsuario: {user}\nAsistente:"
        inputs = self.tok(prompt, return_tensors="pt")
        return inputs

    def generate(
        self,
        user_text: str,
        system_text: str = "Eres Miramar, un asistente para Transportes Miramar.",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stream: Optional[bool] = None,
    ) -> Iterable[str]:
        cfg = self.cfg
        mx = max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens
        tmp = temperature if temperature is not None else cfg.temperature
        tp = top_p if top_p is not None else cfg.top_p
        rp = repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty
        streaming = stream if stream is not None else cfg.streaming

        inputs = self._build_inputs(system_text, user_text)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=mx,
            do_sample=(tmp > 0),
            temperature=max(0.0, tmp),
            top_p=min(1.0, max(0.0, tp)),
            repetition_penalty=max(1.0, rp),
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
        )

        with _inference_ctx():
            if not streaming:
                out = self.model.generate(**inputs, **gen_kwargs)
                text = self.tok.decode(out[0], skip_special_tokens=True)
                # devolver solo la parte del asistente tras "Asistente:"
                yield text.split("Asistente:", 1)[-1].strip()
                return

            # Streaming rudimentario: decodifica incremento de ids
            # (Transformers no stream nativo; hacemos chunking manual)
            past_ids = None
            generated = 0
            for _ in range(mx):
                out = self.model.generate(**inputs, **gen_kwargs, min_new_tokens=1, max_new_tokens=generated+1)
                ids = out[0]
                if past_ids is None:
                    delta_ids = ids
                else:
                    delta_ids = ids[len(past_ids):]
                past_ids = ids
                piece = self.tok.decode(delta_ids, skip_special_tokens=True)
                if piece:
                    yield piece
                generated += 1


# ----------------------------
# Backend: vLLM
# ----------------------------
class VLLMEngine:
    def __init__(self, cfg: ModelConfig):
        # vLLM lazy import
        from vllm import LLM, SamplingParams

        self.cfg = cfg

        # kwargs suaves + hooks (kv_cache_dtype, prefix caching, etc.)
        kwargs: Dict[str, Any] = dict(
            model=cfg.model_id,
            tensor_parallel_size=int(os.getenv("VLLM_TP_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_UTIL", "0.90")),
            enforce_eager=False,
        )
        kwargs = _apply_vllm_kwargs(kwargs)

        self.llm = LLM(**kwargs)
        self.SamplingParams = SamplingParams

        # Warm-up
        self._warmup()

    def _warmup(self) -> None:
        try:
            sp = self.SamplingParams(max_tokens=1, temperature=0.0)
            with _inference_ctx():
                _ = self.llm.generate(["Ok."], sp)
        except Exception:
            pass

    def _build_prompt(self, system: str, user: str) -> str:
        system = _diet_prompt(system)
        user = _diet_prompt(user)
        sys_clean, up_clean, as_clean, tpl = _pretokenize_template(system, "Usuario:", "Asistente:")
        return f"{sys_clean}\nUsuario: {user}\nAsistente:"

    def generate(
        self,
        user_text: str,
        system_text: str = "Eres Miramar, un asistente para Transportes Miramar.",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stream: Optional[bool] = None,
    ) -> Iterable[str]:
        cfg = self.cfg
        mx = max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens
        tmp = temperature if temperature is not None else cfg.temperature
        tp = top_p if top_p is not None else cfg.top_p
        rp = repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty
        streaming = stream if stream is not None else cfg.streaming

        prompt = self._build_prompt(system_text, user_text)
        sp = self.SamplingParams(
            max_tokens=mx,
            temperature=max(0.0, tmp),
            top_p=min(1.0, max(0.0, tp)),
            repetition_penalty=max(1.0, rp),
            stop=["</s>", "\nUsuario:"],  # stops razonables
        )

        with _inference_ctx():
            if streaming:
                # vLLM tiene streaming vía iterator
                for out in self.llm.generate(prompt, sp, use_tqdm=False, streaming=True):
                    # out.text_delta contiene partes; fallback a out.outputs[0].text si no está
                    piece = getattr(out, "text_delta", None)
                    if piece is None:
                        # paquete completo (no streaming)
                        piece = out.outputs[0].text
                    if piece:
                        yield piece
            else:
                outs = self.llm.generate([prompt], sp)
                text = outs[0].outputs[0].text
                yield text.strip()


# ----------------------------
# Selección de backend
# ----------------------------
def _pick_backend(variant: str) -> str:
    cfg = DEFAULTS.get(variant, DEFAULTS["default"])
    backend = cfg.backend
    # Permitir override por env
    backend = os.getenv("MIRAMAR_BACKEND", backend)
    return backend

def _engine_for_variant(variant: str):
    cfg = DEFAULTS.get(variant, DEFAULTS["default"])
    backend = _pick_backend(variant)

    if backend == "vllm":
        try:
            import vllm  # noqa: F401
            return VLLMEngine(cfg)
        except Exception as e:
            print(f"[miramar_bot] vLLM no disponible ({type(e).__name__}: {e}), usando HF.", file=sys.stderr)
            return HFEngine(cfg)
    else:
        # HF por defecto
        return HFEngine(cfg)


# ----------------------------
# API principal
# ----------------------------
def chat_once(engine, user_text: str) -> str:
    """Devuelve una respuesta completa (no streaming)"""
    chunks = engine.generate(user_text, stream=False)
    return "".join(chunks)

def chat_stream(engine, user_text: str) -> Generator[str, None, None]:
    """Genera piezas de texto (streaming)."""
    for piece in engine.generate(user_text, stream=True):
        yield piece

def main(model_variant: str = "lora_4bit") -> None:
    """
    Entry-point utilizado por el lanzador.
    """
    engine = _engine_for_variant(model_variant)

    # Demo CLI simple
    print(f"[Miramar] Backend: {engine.__class__.__name__} | Modelo: {DEFAULTS.get(model_variant, DEFAULTS['default']).model_id}")
    print("Escribe 'exit' para salir. Usa '/stream on|off' para alternar streaming.\n")

    streaming = True
    while True:
        try:
            user = input("Tú: ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        if user.startswith("/stream"):
            parts = user.split()
            if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
                streaming = (parts[1].lower() == "on")
                print(f"(streaming = {streaming})")
                continue

        if streaming:
            print("Miramar: ", end="", flush=True)
            for piece in engine.generate(user, stream=True):
                print(piece, end="", flush=True)
            print()
        else:
            full = "".join(engine.generate(user, stream=False))
            print(f"Miramar: {full}")


if __name__ == "__main__":
    main()
