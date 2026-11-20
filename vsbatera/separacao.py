"""Separação de fontes com IA.

Backends:
  - demucs           (padrão) Hybrid Transformer Demucs da Meta. Modelos:
        htdemucs_ft  -> 4 faixas (drums, bass, other, vocals), melhor qualidade, ~4x mais lento
        htdemucs     -> 4 faixas, mais rápido
        htdemucs_6s  -> 6 faixas (+ guitar, piano) — útil para limpar guitarra/teclado separadamente
  - audio-separator  modelos do UVR (BS-RoFormer, Mel-RoFormer, MDX23C...). Instale com
        `pip install audio-separator` e passe o nome do arquivo do modelo (veja `audio-separator --list_models`).
  - duplo            Demucs + BS-RoFormer-SW (6 faixas) e a média dos dois ("combinado"), que é mais limpa
        que cada um sozinho. Grava as três versões para comparar. Com `segunda_passada`, a versão
        "original − bateria" é separada de novo para tirar o resto de bateria que sobrou.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import achar_faixas, arquivo_faixa, carregar, reamostrar, salvar

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
    modelo_roformer: str = "BS-Roformer-SW.ckpt",
    segunda_passada: bool = True,
    guardar_versoes: bool = True,
) -> dict[str, Path]:
    """Separa `entrada` e grava as faixas em `pasta_saida/stems/`. Retorna {nome_pt: caminho}.

    Sempre grava também `sem_bateria_residual.wav` = mix original − bateria (a forma mais fiel
    de tirar só a bateria, porque não depende da reconstrução dos outros instrumentos).
    """
    raiz = Path(pasta_saida)
    pasta = raiz / "stems"
    pasta.mkdir(parents=True, exist_ok=True)

    if backend == "duplo":
        return _separar_duplo(entrada, raiz, modelo, modelo_roformer, dispositivo, shifts, overlap,
                              segunda_passada, progresso, guardar_versoes)
    if backend == "demucs":
        stems, sr = _separar_demucs(entrada, modelo, dispositivo, shifts, overlap, progresso)
    elif backend in ("audio-separator", "roformer", "uvr"):
        stems, sr = _separar_audio_separator(entrada, modelo, pasta, progresso)
    else:
        raise ValueError(f"Backend desconhecido: {backend}")

    caminhos = _gravar_stems(stems, sr, pasta)
    if "drums" in stems:
        caminhos["sem_bateria_residual"] = _gravar_residual(entrada, stems["drums"], sr, raiz)
    return caminhos


def _gravar_stems(stems: dict[str, np.ndarray], sr: int, pasta: Path) -> dict[str, Path]:
    pasta.mkdir(parents=True, exist_ok=True)
    for velho in achar_faixas(pasta):  # não deixa faixas de uma separação anterior misturadas
        velho.unlink()
    return {NOMES_PT.get(n, n): salvar(arquivo_faixa(pasta, NOMES_PT.get(n, n)), a, sr) for n, a in stems.items()}


def _gravar_residual(entrada, bateria: np.ndarray, sr: int, raiz: Path, nome: str = "sem_bateria_residual") -> Path:
    original, _ = carregar(entrada, sr_alvo=sr)
    n = min(original.shape[1], bateria.shape[1])
    return salvar(arquivo_faixa(raiz, nome), original[:, :n] - bateria[:, :n], sr)


def _parcial(progresso: Callable[[float], None] | None, ini: float, fim: float):
    """Converte o progresso 0..1 de uma parte no trecho [ini, fim] do progresso total."""
    if not progresso:
        return None
    return lambda f: progresso(ini + (fim - ini) * min(1.0, max(0.0, f)))


def _separar_duplo(entrada, raiz: Path, modelo_demucs, modelo_rof, dispositivo, shifts, overlap,
                   segunda_passada: bool, progresso, guardar_versoes: bool = True) -> dict[str, Path]:
    fim_rof = 0.85 if segunda_passada else 1.0
    d, sr = _separar_demucs(entrada, modelo_demucs, dispositivo, shifts, overlap, _parcial(progresso, 0, 0.5))
    if guardar_versoes:
        _gravar_stems(d, sr, raiz / "stems_demucs")

    print(f"[separar] RoFormer '{modelo_rof}'")
    r, sr_r = _separar_audio_separator(entrada, modelo_rof, raiz / "stems", _parcial(progresso, 0.5, fim_rof))
    if sr_r != sr:
        r = {k: reamostrar(v, sr_r, sr) for k, v in r.items()}
    if guardar_versoes:
        _gravar_stems(r, sr, raiz / "stems_roformer")

    combinado = combinar(d, r)
    caminhos = _gravar_stems(combinado, sr, raiz / "stems")
    residual = _gravar_residual(entrada, combinado["drums"], sr, raiz)
    caminhos["sem_bateria_residual"] = residual

    if segunda_passada:
        # Separa de novo a versão sem bateria: o que o modelo ainda achar de bateria nela é vazamento.
        print("[separar] 2ª passada: procurando resto de bateria na versão sem bateria")
        r2, sr2 = _separar_audio_separator(residual, modelo_rof, raiz / "stems", _parcial(progresso, fim_rof, 1.0),
                                           so_bateria=True)
        resto = r2["drums"] if sr2 == sr else reamostrar(r2["drums"], sr2, sr)
        sem, _ = carregar(residual)
        n = min(sem.shape[1], resto.shape[1])
        caminhos["sem_bateria_2passadas"] = salvar(arquivo_faixa(raiz, "sem_bateria_2passadas"), sem[:, :n] - resto[:, :n], sr)
    if progresso:
        progresso(1.0)
    return caminhos


def combinar(d: dict[str, np.ndarray], r: dict[str, np.ndarray], peso_rof: float = 0.5) -> dict[str, np.ndarray]:
    """Média ("ensemble") das faixas do Demucs (4) com as do RoFormer (6).

    bateria/baixo/voz: média direta. guitarra/teclado: só o RoFormer tem. "outros": a média do que cada
    modelo chamou de "resto da banda", menos guitarra e teclado — assim a soma das faixas continua fechando.
    """
    n = min(min(a.shape[1] for a in d.values()), min(a.shape[1] for a in r.values()))
    d = {k: v[:, :n] for k, v in d.items()}
    r = {k: v[:, :n] for k, v in r.items()}
    w = float(peso_rof)
    out: dict[str, np.ndarray] = {}
    for k in ("drums", "bass", "vocals"):
        if k in d and k in r:
            out[k] = (1 - w) * d[k] + w * r[k]
        elif k in d or k in r:
            out[k] = d.get(k, r.get(k))
    extras = [k for k in r if k not in ("drums", "bass", "vocals", "other")]  # guitar, piano
    resto_rof = r.get("other", 0) + sum(r[k] for k in extras)
    resto = (1 - w) * d.get("other", resto_rof) + w * resto_rof
    for k in extras:
        out[k] = r[k]
        resto = resto - r[k]
    out["other"] = resto
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


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


def _separar_audio_separator(entrada, modelo, pasta: Path, progresso=None,
                             so_bateria: bool = False) -> tuple[dict[str, np.ndarray], int]:
    import logging
    import os
    import shutil
    import tempfile

    try:
        from audio_separator.separator import Separator
        from audio_separator.separator.architectures import mdxc_separator
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Instale o backend opcional: pip install audio-separator") from e

    from .audio_io import _ffmpeg

    # o audio-separator chama "ffmpeg" pelo PATH: usa o mesmo que o resto do programa
    pasta_ff = str(Path(_ffmpeg()).parent)
    if pasta_ff not in os.environ.get("PATH", ""):
        os.environ["PATH"] = pasta_ff + os.pathsep + os.environ.get("PATH", "")
    if not shutil.which("ffmpeg"):
        alvo = Path(tempfile.gettempdir()) / "vsbatera_ffmpeg"
        alvo.mkdir(exist_ok=True)
        if not (alvo / "ffmpeg.exe").exists():
            shutil.copy(_ffmpeg(), alvo / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg"))
        os.environ["PATH"] = str(alvo) + os.pathsep + os.environ["PATH"]

    modelos = Path(os.environ.get("VSBATERA_MODELOS", Path.home() / ".vsbatera" / "modelos"))
    tmp = Path(tempfile.mkdtemp(prefix="vsb_sep_", dir=pasta))
    print(f"[separar] audio-separator modelo '{modelo}'")
    sep = Separator(output_dir=str(tmp), output_format="WAV", model_file_dir=str(modelos),
                    log_level=logging.WARNING, output_single_stem="drums" if so_bateria else None)
    sep.load_model(model_filename=modelo)

    tqdm_original = mdxc_separator.tqdm
    if progresso:
        def tqdm_com_progresso(it, *a, **k):
            itens = list(it)
            for i, x in enumerate(itens):
                progresso(i / max(1, len(itens)))
                yield x
            progresso(1.0)

        mdxc_separator.tqdm = tqdm_com_progresso
    try:
        arquivos = sep.separate(str(entrada))
    finally:
        mdxc_separator.tqdm = tqdm_original

    stems: dict[str, np.ndarray] = {}
    sr = 44100
    try:
        for arq in arquivos:
            p = Path(arq)
            if not p.is_absolute():
                p = tmp / p
            m = re.search(r"_\(([^)]+)\)", p.stem)
            nome = (m.group(1) if m else p.stem).lower()
            nome = {"no drums": "instrumental", "no_drums": "instrumental"}.get(nome, nome)
            stems[nome], sr = carregar(p)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return stems, sr
