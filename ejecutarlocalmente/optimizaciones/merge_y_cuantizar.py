"""Script para fusionar el LoRA con el modelo base y cuantizarlo a 3 bits con AWQ."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from awq import AutoAWQForCausalLM
from awq.quantize.quantizer import (
    AwqQuantizer,
    append_str_prefix,
    apply_clip,
    apply_scale,
    clear_memory,
    exclude_layers_to_not_quantize,
    get_best_device,
    get_named_linears,
    get_op_name,
)
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LORA_DIR = ROOT / "modelos" / "mistral_finetuned_miramar_combined_steps20000" / "lora_adapter"  # CORREGIDO: ruta al adaptador real
DEFAULT_BASE_DIR = ROOT / "modelos" / "mistral-7b-bnb-4bit"  # CORREGIDO: usar modelo base correcto (4bit como el LoRA)
MERGED_DIR = ROOT / "modelos" / "mistral_merged_fp16"
QUANT_DIR = ROOT / "modelos" / "mistral_merged_awq_3bit"


class LoggingAwqQuantizer(AwqQuantizer):
    """Quantizer con trazas por capa para diagnosticar bloqueos."""

    def quantize(self):  # type: ignore[override]
        total_layers = len(self.modules)
        start_all = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for i in tqdm(range(total_layers), desc="AWQ"):
            layer_start = time.time()
            print(f"[AWQ] Iniciando capa {i+1}/{total_layers}", flush=True)
            self._current_layer_index = i

            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                if torch.cuda.is_available():
                    best_device = "cuda:" + str(i % torch.cuda.device_count())
                else:
                    best_device = get_best_device()
                self.modules[i] = self.modules[i].to(best_device)
                common_device = next(self.modules[i].parameters()).device

            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs["position_ids"].to(common_device)

            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs["attention_mask"].to(common_device)

            self.inps = self.inps.to(common_device)
            self.awq_model.move_embed(self.model, common_device)

            if (
                transformers.__version__ >= "4.48.0"
                and self.module_kwargs.get("position_embeddings") is None
            ):
                self.module_kwargs["position_embeddings"] = self.model.model.rotary_emb(
                    self.inps, self.module_kwargs["position_ids"]
                )

            if (
                transformers.__version__ >= "4.48.0"
                and self.module_kwargs.get("attention_mask") is None
            ):
                self.module_kwargs["attention_mask"] = None

            for k, v in list(self.module_kwargs.items()):
                if isinstance(v, tuple):
                    self.module_kwargs[k] = tuple(
                        item.to(common_device) if isinstance(item, (torch.Tensor, nn.Module)) else item
                        for item in v
                    )

            named_linears = get_named_linears(self.modules[i])
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )
            print(f"[AWQ][{i+1}/{total_layers}] Paso 1/4: obteniendo características", flush=True)
            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            print(f"[AWQ][{i+1}/{total_layers}] Paso 2/4: buscando escalas", flush=True)
            module_config = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.modules[i]) + "."
            )

            if self.apply_clip:
                print(f"[AWQ][{i+1}/{total_layers}] Paso 3/4: buscando clipping", flush=True)
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )
                apply_clip(self.modules[i], clip_list)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.modules[i]) + "."
                )

            if not self.export_compatible:
                print(f"[AWQ][{i+1}/{total_layers}] Paso 4/4: aplicando cuantización", flush=True)
                self._apply_quant(self.modules[i], named_linears)

            clear_memory()

            elapsed = time.time() - layer_start
            total_elapsed = time.time() - start_all
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / 1024**3
                print(
                    f"[AWQ][{i+1}/{total_layers}] capa procesada en {elapsed:.1f}s"
                    f" (acumulado {total_elapsed/60:.1f} min, pico GPU {mem:.2f} GB)",
                    flush=True,
                )
            else:
                print(
                    f"[AWQ][{i+1}/{total_layers}] capa procesada en {elapsed:.1f}s"
                    f" (acumulado {total_elapsed/60:.1f} min)",
                    flush=True,
                )

    def _search_best_clip(self, layer, named_linears, input_feat):  # type: ignore[override]
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        names = list(named_linears.keys())
        total = len(names)
        layer_num = getattr(self, "_current_layer_index", 0) + 1

        for idx, name in enumerate(names, start=1):
            if any(token in name for token in avoid_clipping):
                print(
                    f"[AWQ][{layer_num}] Clip {idx}/{total}: saltando {name} (avoid list)",
                    flush=True,
                )
                continue

            print(
                f"[AWQ][{layer_num}] Clip {idx}/{total}: procesando {name}",
                flush=True,
            )
            named_linears[name].to(get_best_device())
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list


def resolve_device(requested: str) -> str:
    """Normaliza la elección de dispositivo."""

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[ADVERTENCIA] CUDA no disponible, usando CPU.")
        return "cpu"
    return requested
def merge_lora(
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    device: str,
    dtype: torch.dtype,
) -> None:
    """Fusiona el LoRA sobre el modelo base en el dispositivo indicado."""

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cargando modelo base desde {base_dir} en {device}…")
    target_device = "cuda:0" if device.startswith("cuda") else device
    device_map = {"": target_device} if device != "cpu" else None
    offload_dir = output_dir / "offload"
    if device != "cpu" and not offload_dir.exists():
        offload_dir.mkdir(parents=True, exist_ok=True)
    if device == "cpu":
        base_model = AutoModelForCausalLM.from_pretrained(
            base_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        config = AutoConfig.from_pretrained(base_dir, trust_remote_code=True)
        with init_empty_weights():
            base_model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        base_model.tie_weights()
        base_model = load_checkpoint_and_dispatch(
            base_model,
            base_dir,
            device_map=device_map or {"": target_device},
            offload_folder=str(offload_dir),
            offload_state_dict=True,
            dtype=dtype,
        )

    print("Cargando adaptador LoRA…")
    lora_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        device_map=device_map,
    )

    print("Fusionando pesos LoRA → modelo base…")
    with torch.no_grad():
        merged_model = lora_model.merge_and_unload()

    print(f"Guardando modelo fusionado en {output_dir}…")
    merged_model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=False)
    tokenizer.save_pretrained(output_dir)

    del merged_model, lora_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def quantize_awq(
    merged_dir: Path,
    output_dir: Path,
    bits: int = 4,
    device: str = "cpu",
    max_calib_samples: int = 64,
    group_size: int = 64,
    calib_data: list[str] | str = "pileval",
    calib_seq_len: int = 256,
) -> None:
    """Cuantiza el modelo fusionado usando AWQ."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir)
    print(f"Cargando modelo fusionado desde {merged_dir} para cuantizar en {device}…")
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), use_fast=False)

    device_map = {"": device} if device != "cpu" else None
    offload_dir = output_dir / "offload"
    if device != "cpu" and not offload_dir.exists():
        offload_dir.mkdir(parents=True, exist_ok=True)
    model = AutoAWQForCausalLM.from_pretrained(
        str(merged_dir),
        safetensors=True,
        trust_remote_code=True,
        device_map=device_map,
        offload_folder=str(offload_dir) if device != "cpu" else None,
    )
    if device != "cpu":
        model.to(device)

    effective_seq_len = max(1, calib_seq_len)
    calib_payload: list[list[int]] | str
    if isinstance(calib_data, list):
        token_blocks: list[list[int]] = []
        max_token_len = 0
        for sample in calib_data:
            tokens = tokenizer.encode(sample)
            if not tokens:
                continue
            max_token_len = max(max_token_len, len(tokens))
            for start in range(0, len(tokens), effective_seq_len):
                block = tokens[start : start + effective_seq_len]
                if block:
                    token_blocks.append(block)
                if len(token_blocks) >= max_calib_samples:
                    break
            if len(token_blocks) >= max_calib_samples:
                break
        if not token_blocks:
            raise ValueError("Las muestras de calibración no contienen tokens válidos tras el preprocesado.")
        effective_seq_len = max(1, min(effective_seq_len, max_token_len))
        calib_payload = token_blocks[:max_calib_samples]
    else:
        calib_payload = calib_data

    print(
        f"Cuantizando a {bits} bits con group_size {group_size}, {len(calib_payload) if isinstance(calib_payload, list) else max_calib_samples} muestras"
        f" y seq_len {effective_seq_len}…"
    )
    quant_config = {
        "w_bit": bits,
        "q_group_size": group_size,
        "zero_point": True,
        "version": "GEMM",
    }
    model.quantize(
        tokenizer=tokenizer,
        quant_config=quant_config,
        calib_data=calib_payload,
        max_calib_samples=max_calib_samples,
        max_calib_seq_len=effective_seq_len,
        quantizer_cls=LoggingAwqQuantizer,
    )

    print(f"Guardando modelo cuantizado en {output_dir_str}…")
    model.save_quantized(output_dir_str)
    tokenizer.save_pretrained(output_dir_str)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_calib_data(path: Path, limit: int) -> list[str]:
    """Carga ejemplos de calibración desde un archivo JSONL/JSON o texto."""

    if not path.exists():
        raise FileNotFoundError(f"Archivo de calibración no encontrado: {path}")

    samples: list[str] = []
    if path.suffix.lower() in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, str):
                    text = data
                elif isinstance(data, dict):
                    text = ""
                    if "messages" in data and isinstance(data["messages"], list):
                        turns = [msg.get("content", "") for msg in data["messages"] if isinstance(msg, dict)]
                        text = " \n".join(filter(None, turns))
                    if not text:
                        instruction = data.get("instruction") or data.get("prompt") or ""
                        user = data.get("input") or data.get("question") or ""
                        output = data.get("output") or data.get("answer") or ""
                        if instruction or user:
                            text = " \n".join(filter(None, [instruction, user, output]))
                        else:
                            text = output or ""
                text = text.strip()
                if text:
                    samples.append(text)
                if len(samples) >= limit:
                    break
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                clean = line.strip()
                if clean:
                    samples.append(clean)
                if len(samples) >= limit:
                    break

    if not samples:
        raise ValueError(f"No se extrajeron textos de calibración desde {path}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Dispositivo principal para fusionar y cuantizar.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Precisión para la fusión en GPU.",
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=64,
        help="Número de muestras para la calibración AWQ.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Group size para la cuantización AWQ.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Ruta al modelo base en FP16.",
    )
    parser.add_argument(
        "--adapter-dir",
        default=str(DEFAULT_LORA_DIR),  # CORREGIDO: ya apunta directamente al lora_adapter
        help="Ruta al directorio con el adaptador LoRA.",
    )
    parser.add_argument(
        "--merged-dir",
        default=str(MERGED_DIR),
        help="Ruta donde se guardará el modelo fusionado (fp16).",
    )
    parser.add_argument(
        "--quant-dir",
        default=str(QUANT_DIR),
        help="Ruta de salida para el modelo cuantizado AWQ.",
    )
    parser.add_argument("--skip-merge", action="store_true", help="Omitir la fusión LoRA → modelo base")
    parser.add_argument("--skip-quant", action="store_true", help="Omitir la cuantización AWQ")
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4],
        help="Precisión de pesos para AWQ (solo 4-bit soportado)",
    )
    parser.add_argument(
        "--calib-file",
        help="Archivo con textos de calibración (JSONL o TXT). Si no se pasa, se usa 'pileval'.",
    )
    parser.add_argument(
        "--calib-limit",
        type=int,
        default=64,
        help="Número máximo de líneas a cargar del archivo de calibración.",
    )
    parser.add_argument(
        "--calib-seq-len",
        type=int,
        default=256,
        help="Longitud máxima de secuencia para las muestras de calibración.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    base_dir = Path(args.base_dir).resolve()
    adapter_dir = Path(args.adapter_dir).resolve()
    merged_dir = Path(args.merged_dir).resolve()
    quant_dir = Path(args.quant_dir).resolve()

    if not args.skip_merge:
        merge_lora(base_dir, adapter_dir, merged_dir, device=device, dtype=dtype)
    else:
        print("[INFO] Omitiendo etapa de fusión.")

    if args.skip_quant:
        print("[INFO] Omitiendo cuantización.")
        return

    quant_device = device if torch.cuda.is_available() else "cpu"
    calib_payload: list[str] | str
    if args.calib_file:
        calib_payload = load_calib_data(Path(args.calib_file), args.calib_limit)
        max_calib_samples = min(args.max_calib_samples, len(calib_payload))
    else:
        calib_payload = "pileval"
        max_calib_samples = args.max_calib_samples

    quantize_awq(
        merged_dir,
        quant_dir,
        bits=args.bits,
        device=quant_device,
        max_calib_samples=max_calib_samples,
        group_size=args.group_size,
        calib_data=calib_payload,
        calib_seq_len=args.calib_seq_len,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - ejecución manual
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


"""README RÁPIDO

Comandos principales (copiar y pegar):

# Activar entorno y fijar configuración de memoria
source /home/red/projects/Fine2/.venv/bin/activate
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# Fusionar LoRA sobre el modelo FP16 (sin cuantizar todavía)
python ejecutarlocalmente/optimizaciones/merge_y_cuantizar.py \
  --device cuda \
  --dtype float16 \
  --base-dir /home/red/projects/Fine2/mistral-7b-instruct-fp16 \
  --adapter-dir /home/red/projects/Fine2/mistral_finetuned_miramar_combined_steps20000/lora_adapter \
  --merged-dir /home/red/projects/Fine2/mistral_merged_fp16 \
  --quant-dir /home/red/projects/Fine2/mistral_merged_awq_3bit \
  --skip-quant

# Cuantizar el modelo fusionado con AWQ (usar dataset local para calibración)
python ejecutarlocalmente/optimizaciones/merge_y_cuantizar.py \
  --device cuda \
  --dtype float16 \
  --base-dir /home/red/projects/Fine2/mistral-7b-instruct-fp16 \
  --adapter-dir /home/red/projects/Fine2/mistral_finetuned_miramar_combined_steps20000/lora_adapter \
  --merged-dir /home/red/projects/Fine2/mistral_merged_fp16 \
  --quant-dir /home/red/projects/Fine2/mistral_merged_awq_3bit \
  --max-calib-samples 8 \
  --group-size 128 \
  --bits 3 \
  --skip-merge \
  --calib-file /home/red/projects/Fine2/Dataset_de_Miramar/transportes_miramar_dataset_20k_20250909_044327.jsonl \
  --calib-limit 64 \
  --calib-seq-len 256

"""
