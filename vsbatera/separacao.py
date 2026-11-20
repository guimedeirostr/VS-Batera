"""Separação de fontes com IA.

Backends:
  - demucs           (padrão) Hybrid Transformer Demucs da Meta. Modelos:
        htdemucs_ft  -> 4 faixas (drums, bass, other, vocals), melhor qualidade, ~4x mais lento
        htdemucs     -> 4 faixas, mais rápido
        htdemucs_6s  -> 6 faixas (+ guitar, piano) — útil para limpar guitarra/teclado separadamente
  - audio-separator  modelos do UVR (BS-RoFormer, Mel-RoFormer, MDX23C...). Instale com
        `pip install audio-separator` e passe o nome do arquivo do modelo (veja `audio-separator --list_models`).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import carregar, salvar

NOMES_PT = {
    "drums": "bateria",
    "bass": "baixo",
    "other": "outros",
    "vocals": "voz",
    "guitar": "guitarra",
    "piano": "teclado",
    "instrumental": "instrumental",
}


def separar(
    entrada: str | Path,
    pasta_saida: str | Path,
    backend: str = "demucs",
    modelo: str = "htdemucs_ft",
    dispositivo: str = "auto",
    shifts: int = 2,
    overlap: float = 0.25,
    progresso: Callable[[float], None] | None = None,
) -> dict[str, Path]:
    """Separa `entrada` e grava as faixas em `pasta_saida/stems/`. Retorna {nome_pt: caminho}.

    Sempre grava também `sem_bateria_residual.wav` = mix original − bateria (a forma mais fiel
    de tirar só a bateria, porque não depende da reconstrução dos outros instrumentos).
    """
    pasta = Path(pasta_saida) / "stems"
    pasta.mkdir(parents=True, exist_ok=True)

    if backend == "demucs":
        stems, sr = _separar_demucs(entrada, modelo, dispositivo, shifts, overlap, progresso)
    elif backend in ("audio-separator", "roformer", "uvr"):
        stems, sr = _separar_audio_separator(entrada, modelo, pasta)
    else:
        raise ValueError(f"Backend desconhecido: {backend}")

    caminhos: dict[str, Path] = {}
    for nome, audio in stems.items():
        nome_pt = NOMES_PT.get(nome, nome)
        caminhos[nome_pt] = salvar(pasta / f"{nome_pt}.wav", audio, sr)

    if "bateria" in caminhos:
        original, _ = carregar(entrada, sr_alvo=sr)
        bateria = stems["drums"]
        n = min(original.shape[1], bateria.shape[1])
        residual = original[:, :n] - bateria[:, :n]
        caminhos["sem_bateria_residual"] = salvar(Path(pasta_saida) / "sem_bateria_residual.wav", residual, sr)
    return caminhos


def _dispositivo(dispositivo: str) -> str:
    import torch

    if dispositivo != "auto":
        return dispositivo
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _separar_demucs(entrada, modelo, dispositivo, shifts, overlap, progresso=None) -> tuple[dict[str, np.ndarray], int]:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    dev = _dispositivo(dispositivo)
    print(f"[separar] Demucs '{modelo}' em {dev} (shifts={shifts}, overlap={overlap})")
    model = get_model(modelo)
    model.eval()
    sr = model.samplerate

    audio, _ = carregar(entrada, sr_alvo=sr)
    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    media, desvio = ref.mean(), ref.std() + 1e-8
    wav = (wav - media) / desvio

    callback = _contador_progresso(model, wav.shape[-1], shifts, overlap, progresso) if progresso else None
    with torch.no_grad():
        fontes = apply_model(
            model, wav[None], device=dev, shifts=shifts, split=True, overlap=overlap, progress=True,
            callback=callback,
        )[0]
    if progresso:
        progresso(1.0)
    fontes = fontes * desvio + media
    return {nome: fontes[i].cpu().numpy().astype(np.float32) for i, nome in enumerate(model.sources)}, sr


def _contador_progresso(model, comprimento: int, shifts: int, overlap: float, progresso: Callable[[float], None]):
    """Callback do Demucs que conta os pedaços (segmentos) processados e reporta a fração concluída."""
    import math

    modelos = getattr(model, "models", [model])
    seg = float(getattr(modelos[0], "segment", 7.8))
    passo = max(1, int((1 - overlap) * model.samplerate * seg))
    folga = int(0.5 * model.samplerate) if shifts else 0  # o Demucs desloca o áudio em até 0,5 s
    por_passada = math.ceil((comprimento + folga) / passo)
    total = max(1, len(modelos) * max(1, shifts) * por_passada)
    feitos = [0]

    def cb(d: dict) -> None:
        if d.get("state") == "end":
            feitos[0] += 1
            progresso(min(0.99, feitos[0] / total))

    return cb


def _separar_audio_separator(entrada, modelo, pasta: Path) -> tuple[dict[str, np.ndarray], int]:
    try:
        from audio_separator.separator import Separator
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Instale o backend opcional: pip install audio-separator") from e

    tmp = pasta / "_bruto"
    tmp.mkdir(exist_ok=True)
    print(f"[separar] audio-separator modelo '{modelo}'")
    sep = Separator(output_dir=str(tmp), output_format="WAV")
    sep.load_model(model_filename=modelo)
    arquivos = sep.separate(str(entrada))

    stems: dict[str, np.ndarray] = {}
    sr = 44100
    for arq in arquivos:
        p = Path(arq)
        if not p.is_absolute():
            p = tmp / p
        m = re.search(r"\(([^)]+)\)", p.stem)
        nome = (m.group(1) if m else p.stem).lower()
        nome = {"no drums": "instrumental", "no_drums": "instrumental"}.get(nome, nome)
        stems[nome], sr = carregar(p)
    return stems, sr
