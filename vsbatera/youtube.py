"""Baixa o áudio de um link (YouTube e outros sites suportados pelo yt-dlp) para a pasta de músicas."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from .audio_io import _ffmpeg


def nome_seguro(titulo: str) -> str:
    """Tira caracteres que o Windows não aceita em nome de arquivo."""
    nome = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", titulo).strip(" .")
    return nome[:150] or "musica"


def baixar(url: str, pasta: str | Path, progresso: Callable[[float], None] | None = None) -> Path:
    """Baixa só o áudio de `url` como MP3 320k em `pasta` e devolve o caminho do arquivo."""
    import yt_dlp

    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)

    def gancho(d):
        if progresso and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progresso(min(0.95, 0.95 * d.get("downloaded_bytes", 0) / total))

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    nome = nome_seguro(info.get("title") or info.get("id") or "musica")

    opcoes = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "outtmpl": str(pasta / f"{nome}.%(ext)s"),
        "ffmpeg_location": _ffmpeg(),
        "progress_hooks": [gancho],
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "320"}],
    }
    with yt_dlp.YoutubeDL(opcoes) as ydl:
        ydl.download([url])
    destino = pasta / f"{nome}.mp3"
    if not destino.exists():
        raise FileNotFoundError(f"o download terminou mas {destino.name} não apareceu")
    if progresso:
        progresso(1.0)
    return destino
