# Transcribir Audios de Whatsapp

Script en Python para transcribir notas de voz de WhatsApp (y otros audios) a texto, **100% en local**, sin enviar nada a ningún servidor.

Usa [faster-whisper](https://github.com/SYSTRAN/faster-whisper) para decodificar y transcribir el audio (incluye soporte nativo para `.ogg`/`.opus`, el formato de WhatsApp, sin necesidad de tener `ffmpeg` instalado).

## Características

- Transcribe un solo archivo o una carpeta completa de audios.
- Detección automática de idioma (o forzado con `--idioma`).
- Genera un `.txt` con el texto y, opcionalmente, un `.srt` con marcas de tiempo.
- Soporta CPU y GPU (CUDA).
- Formatos admitidos: `.ogg`, `.opus`, `.m4a`, `.mp3`, `.wav`, `.flac`, `.aac`, `.mp4`, `.webm`.

## Instalación

Requiere Python 3.9+.

```bash
pip install faster-whisper
```

## Uso

```bash
# Transcribir un solo archivo
python transcribir.py nota.ogg

# Transcribir todos los audios de una carpeta
python transcribir.py /ruta/a/carpeta

# Elegir modelo y generar también subtítulos .srt
python transcribir.py /ruta/a/carpeta --modelo small --srt

# Forzar idioma
python transcribir.py nota.ogg --idioma es --modelo medium

# Usar GPU (si tienes CUDA instalado)
python transcribir.py nota.ogg --dispositivo cuda
```

Cada audio genera un archivo `.txt` con la transcripción (y un `.srt` si se usa `--srt`) en la misma carpeta que el audio original.

### Modelos disponibles

| Modelo    | Tamaño   | Velocidad     | Calidad          |
|-----------|----------|---------------|------------------|
| tiny      | ~75 MB   | muy rápido    | baja             |
| base      | ~150 MB  | rápido        | aceptable        |
| small     | ~500 MB  | equilibrado   | buena (default)  |
| medium    | ~1.5 GB  | más lento     | notablemente mejor |
| large-v3  | ~3 GB    | lento (GPU recomendada) | máxima |

La primera ejecución descarga el modelo elegido y lo cachea en `~/.cache/huggingface`.

### Opciones

| Flag           | Descripción                                              | Default  |
|----------------|-----------------------------------------------------------|----------|
| `ruta`         | Archivo de audio o carpeta con audios                    | —        |
| `--modelo`     | tiny \| base \| small \| medium \| large-v3               | small    |
| `--idioma`     | Código ISO del idioma (p. ej. `es`), o `auto`              | es       |
| `--dispositivo`| auto \| cpu \| cuda                                        | auto     |
| `--srt`        | Genera también un archivo `.srt` con marcas de tiempo      | false    |

## Privacidad

Todo el procesamiento ocurre en tu máquina. El audio nunca se sube a internet.
