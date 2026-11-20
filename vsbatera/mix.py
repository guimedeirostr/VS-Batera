"""Mixagem final: soma as faixas (menos a bateria), normaliza volume e exporta a versão de estudo."""
from __future__ import annotations

from typing import Any

import numpy as np


def aplicar_pan(audio: np.ndarray, pan: float) -> np.ndarray:
    """pan -1 (esquerda) .. 0 (centro) .. +1 (direita), lei de potência constante normalizada no centro."""
    if not pan:
        return audio
    pan = float(np.clip(pan, -1, 1))
    ang = (pan + 1) * np.pi / 4
    g = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32) * np.sqrt(2)
    return audio * g[:, None]


def mixar(
    faixas: dict[str, np.ndarray], sr: int, cfg_faixas: dict[str, dict[str, Any]], cfg_mix: dict[str, Any]
) -> np.ndarray:
    excluir = set(cfg_mix.get("excluir", ["bateria"]))
    guia_db = cfg_mix.get("bateria_guia_db")
    n = min(a.shape[1] for a in faixas.values())
    soma = np.zeros((2, n), dtype=np.float32)
    for nome, audio in faixas.items():
        cfg = cfg_faixas.get(nome, {})
        if cfg.get("mudo"):
            continue
        if nome in excluir:
            if nome == "bateria" and guia_db is not None:
                soma += audio[:, :n] * (10 ** (float(guia_db) / 20))
            continue
        soma += aplicar_pan(audio[:, :n], float(cfg.get("pan", 0) or 0))

    soma = alterar_velocidade(soma, sr, float(cfg_mix.get("velocidade", 1.0)), float(cfg_mix.get("tom_semitons", 0)))
    return masterizar(soma, sr, cfg_mix.get("lufs_alvo", -14.0), cfg_mix.get("teto_db", -1.0))


def alterar_velocidade(audio: np.ndarray, sr: int, velocidade: float = 1.0, semitons: float = 0.0) -> np.ndarray:
    """Muda andamento sem mudar o tom (e/ou muda o tom) usando Rubber Band (via pedalboard)."""
    if abs(velocidade - 1.0) < 1e-3 and abs(semitons) < 1e-3:
        return audio
    from pedalboard import time_stretch

    return np.asarray(
        time_stretch(audio, sr, stretch_factor=velocidade, pitch_shift_in_semitones=semitons), dtype=np.float32
    )


def masterizar(audio: np.ndarray, sr: int, lufs_alvo: float | None = -14.0, teto_db: float = -1.0) -> np.ndarray:
    import pedalboard as pb
    import pyloudnorm as pyln

    x = audio
    if lufs_alvo is not None:
        try:
            atual = pyln.Meter(sr).integrated_loudness(x.T)
            if np.isfinite(atual):
                x = x * (10 ** ((float(lufs_alvo) - atual) / 20))
        except ValueError:
            pass  # áudio curto demais para medir
    x = pb.Pedalboard([pb.Limiter(threshold_db=float(teto_db), release_ms=80.0)])(x.astype(np.float32), sr)
    teto = 10 ** (float(teto_db) / 20)
    return np.clip(x, -teto, teto).astype(np.float32)
