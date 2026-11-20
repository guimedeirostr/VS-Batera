"""Leitura/gravação de áudio. Usa soundfile e cai para o ffmpeg quando o formato não é suportado."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def carregar(caminho: str | Path, sr_alvo: int | None = None) -> tuple[np.ndarray, int]:
    """Retorna (audio, sr) com audio no formato (canais, amostras), float32, sempre estéreo."""
    caminho = Path(caminho)
    try:
        dados, sr = sf.read(str(caminho), dtype="float32", always_2d=True)
    except Exception:
        dados, sr = _carregar_ffmpeg(caminho)
    audio = dados.T  # (canais, amostras)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if sr_alvo and sr != sr_alvo:
        audio = reamostrar(audio, sr, sr_alvo)
        sr = sr_alvo
    return np.ascontiguousarray(audio, dtype=np.float32), sr


def _carregar_ffmpeg(caminho: Path) -> tuple[np.ndarray, int]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"Não consegui ler {caminho} e o ffmpeg não está instalado.")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "decod.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(caminho), "-ac", "2", "-ar", "44100", str(wav)],
            check=True,
        )
        return sf.read(str(wav), dtype="float32", always_2d=True)


def reamostrar(audio: np.ndarray, sr_de: int, sr_para: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_de, sr_para)
    return resample_poly(audio, sr_para // g, sr_de // g, axis=-1).astype(np.float32)


def salvar(caminho: str | Path, audio: np.ndarray, sr: int, bits: int = 24) -> Path:
    """Salva (canais, amostras). WAV/FLAC via soundfile; MP3 via ffmpeg."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if caminho.suffix.lower() == ".mp3":
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "x.wav"
            sf.write(str(wav), audio.T, sr, subtype="PCM_24")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "320k", str(caminho)], check=True
            )
        return caminho
    subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}[bits]
    sf.write(str(caminho), audio.T, sr, subtype=subtype)
    return caminho
