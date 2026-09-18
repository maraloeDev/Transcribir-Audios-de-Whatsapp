#!/usr/bin/env python3
"""
Transcriptor local de notas de voz (WhatsApp .ogg/.opus y otros formatos de audio).

Usa faster-whisper: corre 100% en local, no envía nada a ningún servidor
y no necesita ffmpeg instalado (decodifica el audio con PyAV internamente).

Instalación:
    pip install faster-whisper

Uso:
    python transcribir.py nota.ogg
    python transcribir.py /ruta/a/carpeta --modelo small --srt
    python transcribir.py nota.ogg --idioma es --modelo medium

Modelos (calidad / velocidad / RAM aprox.):
    tiny    ~  75 MB   muy rápido, calidad baja
    base    ~ 150 MB   rápido, aceptable
    small   ~ 500 MB   buen equilibrio  <- por defecto
    medium  ~ 1.5 GB   notablemente mejor, más lento
    large-v3 ~ 3 GB    máxima calidad, necesita GPU para ser cómodo

La primera ejecución descarga el modelo y lo cachea en ~/.cache/huggingface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXTENSIONES = {".ogg", ".opus", ".m4a", ".mp3", ".wav", ".flac", ".aac", ".mp4", ".webm"}


def formato_timestamp(segundos: float) -> str:
    """Convierte segundos a HH:MM:SS,mmm (formato SRT)."""
    milis = int(round(segundos * 1000))
    horas, milis = divmod(milis, 3_600_000)
    minutos, milis = divmod(milis, 60_000)
    segs, milis = divmod(milis, 1000)
    return f"{horas:02d}:{minutos:02d}:{segs:02d},{milis:03d}"


def recopilar_archivos(ruta: Path) -> list[Path]:
    if ruta.is_file():
        return [ruta]
    if ruta.is_dir():
        return sorted(p for p in ruta.rglob("*") if p.suffix.lower() in EXTENSIONES)
    raise FileNotFoundError(f"No existe la ruta: {ruta}")


def transcribir(modelo, archivo: Path, idioma: str | None, srt: bool) -> None:
    print(f"\n--- {archivo.name} ---", flush=True)

    segmentos, info = modelo.transcribe(
        str(archivo),
        language=idioma,          # None = detección automática
        vad_filter=True,          # recorta silencios: más rápido y menos alucinaciones
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
    )

    print(f"Idioma: {info.language} ({info.language_probability:.0%}) | "
          f"Duración: {info.duration:.1f}s", flush=True)

    lineas_texto: list[str] = []
    lineas_srt: list[str] = []

    for i, seg in enumerate(segmentos, start=1):
        texto = seg.text.strip()
        print(f"[{seg.start:7.2f}s] {texto}", flush=True)
        lineas_texto.append(texto)
        if srt:
            lineas_srt.append(
                f"{i}\n{formato_timestamp(seg.start)} --> {formato_timestamp(seg.end)}\n{texto}\n"
            )

    salida_txt = archivo.with_suffix(".txt")
    salida_txt.write_text(" ".join(lineas_texto) + "\n", encoding="utf-8")
    print(f"\nGuardado: {salida_txt}")

    if srt:
        salida_srt = archivo.with_suffix(".srt")
        salida_srt.write_text("\n".join(lineas_srt), encoding="utf-8")
        print(f"Guardado: {salida_srt}")


def cargar_modelo(WhisperModel, nombre: str, dispositivo: str):
    """Carga el modelo. ctranslate2 solo toca las DLLs de CUDA en la primera
    inferencia (no al cargar), así que 'auto' no puede detectar en la carga
    si la GPU va a fallar: por eso 'auto' usa CPU directamente, que siempre
    funciona. Pide --dispositivo cuda explícitamente si tienes CUDA instalado."""
    dev, tipo_computo = ("cuda", "float16") if dispositivo == "cuda" else ("cpu", "int8")
    print(f"Cargando modelo '{nombre}' ({dev}/{tipo_computo})...", flush=True)
    return WhisperModel(nombre, device=dev, compute_type=tipo_computo)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audios en local con faster-whisper."
    )
    parser.add_argument("ruta", type=Path, help="Archivo de audio o carpeta con audios")
    parser.add_argument("--modelo", default="small",
                        help="tiny | base | small | medium | large-v3 (por defecto: small)")
    parser.add_argument("--idioma", default="es",
                        help="Código ISO del idioma, p. ej. 'es'. Usa 'auto' para detectarlo.")
    parser.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Dónde ejecutar el modelo (por defecto: auto)")
    parser.add_argument("--srt", action="store_true",
                        help="Generar también un .srt con marcas de tiempo")
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Falta la dependencia. Instálala con:\n    pip install faster-whisper",
              file=sys.stderr)
        return 1

    try:
        archivos = recopilar_archivos(args.ruta)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if not archivos:
        print("No se encontraron audios con extensiones soportadas.", file=sys.stderr)
        return 1

    # int8 en CPU: mucho más rápido y con una pérdida de calidad mínima.
    modelo = cargar_modelo(WhisperModel, args.modelo, args.dispositivo)

    idioma = None if args.idioma == "auto" else args.idioma

    for archivo in archivos:
        try:
            transcribir(modelo, archivo, idioma, args.srt)
        except Exception as e:  # un archivo corrupto no debe tumbar el lote
            print(f"Error con {archivo.name}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())