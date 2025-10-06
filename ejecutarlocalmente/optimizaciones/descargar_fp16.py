"""Descarga el modelo Mistral-7B-Instruct en FP16 usando huggingface_hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

# Puedes pegar tu token aquí, aunque es más seguro usar la variable de entorno HF_TOKEN.
# Ejemplo: HARDCODED_TOKEN = "hf_xxx"
HARDCODED_TOKEN = None  # Por seguridad, usa la variable de entorno HF_TOKEN


def descargar_modelo(destino: Path, token: str | None) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo FP16 en {destino}…")
    snapshot_download(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        revision="main",
        local_dir=str(destino),
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )
    print("✅ Descarga completada.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destino",
        default="mistral-7b-instruct-fp16",
        help="Carpeta donde se guardará el modelo (ruta relativa al proyecto).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or HARDCODED_TOKEN,
        help=(
            "Token de acceso a Hugging Face. Puedes usar --token, la variable de entorno HF_TOKEN "
            "o asignar HARDCODED_TOKEN en el script (menos recomendado)."
        ),
    )
    args = parser.parse_args()

    if not args.token:
        raise SystemExit(
            "❌ Debes proporcionar un token de Hugging Face (argumento --token o variable HF_TOKEN)."
        )

    destino = Path(__file__).resolve().parents[2] / args.destino
    descargar_modelo(destino, args.token)


if __name__ == "__main__":
    main()
