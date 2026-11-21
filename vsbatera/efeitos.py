"""Efeitos por faixa do mixer: EQ de 3 bandas, reverb (envio) e limpeza com intensidade ajustável.

O navegador usa exatamente a mesma resposta ao impulso do reverb (servida em /api/ir.wav) e filtros
equivalentes (BiquadFilter), então o arquivo salvo soa como o que se ouve no mixer.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np

# bandas do EQ (iguais às do navegador)
EQ_GRAVE_HZ = 100.0
EQ_MEDIO_HZ = 1000.0
EQ_MEDIO_Q = 0.9
EQ_AGUDO_HZ = 8000.0

# intensidades de limpeza oferecidas no mixer (1.0 = a calibração automática feita no processamento).
# "suave" é um modo à parte (ver CONFIG_SUAVE): só tira o vazamento de bateria, com máscara suavizada.
INTENSIDADES = {"original": 0.0, "suave": -1.0, "leve": 0.4, "media": 0.7, "automatica": 1.0}

# Modo "suave": medido em 3 músicas, tira 30–75% do vazamento de bateria sem a redução de ruído que deixava a
# guitarra fina/robótica (a automática chegava a mexer +5 dB no balanço de agudos).
CONFIG_SUAVE = {"remover_vazamento_bateria": 0.5, "mascara_suave": True, "piso_mascara": 0.35,
                "reducao_ruido": 0.0, "gate_db": None, "passa_alta_hz": 80, "passa_baixa_hz": None,
                "eq": [], "compressor": None}


def eq(audio: np.ndarray, sr: int, grave_db: float = 0, medio_db: float = 0, agudo_db: float = 0) -> np.ndarray:
    if not (grave_db or medio_db or agudo_db):
        return audio
    import pedalboard as pb

    cadeia = []
    if grave_db:
        cadeia.append(pb.LowShelfFilter(cutoff_frequency_hz=EQ_GRAVE_HZ, gain_db=float(grave_db)))
    if medio_db:
        cadeia.append(pb.PeakFilter(cutoff_frequency_hz=EQ_MEDIO_HZ, gain_db=float(medio_db), q=EQ_MEDIO_Q))
    if agudo_db:
        cadeia.append(pb.HighShelfFilter(cutoff_frequency_hz=EQ_AGUDO_HZ, gain_db=float(agudo_db)))
    return np.asarray(pb.Pedalboard(cadeia)(audio.astype(np.float32), sr), dtype=np.float32)


def resposta_reverb(sr: int = 44100, duracao: float = 2.4, pre_delay: float = 0.015) -> np.ndarray:
    """Resposta ao impulso de uma sala média (estéreo, determinística): ruído com decaimento exponencial,
    agudos morrendo mais rápido. Normalizada para energia 1 por canal (o envio sai no mesmo nível)."""
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(1234)
    n = int(duracao * sr)
    t = np.arange(n) / sr
    ir = np.zeros((2, n + int(pre_delay * sr)), dtype=np.float64)
    for c in range(2):
        ruido = rng.standard_normal(n)
        brilho = sosfilt(butter(2, 9000, "low", fs=sr, output="sos"), ruido)   # cauda inicial mais brilhante
        escuro = sosfilt(butter(2, 2500, "low", fs=sr, output="sos"), ruido)   # cauda final mais escura
        mistura = np.exp(-t / 0.35) * brilho + (1 - np.exp(-t / 0.35)) * escuro
        cauda = mistura * np.exp(-t / 0.42)                                     # RT60 ~ 2,9 s
        ir[c, -n:] = cauda / (np.sqrt(np.sum(cauda ** 2)) + 1e-12)
    return ir.astype(np.float32)


_IR_CACHE: dict[int, np.ndarray] = {}


def _ir(sr: int) -> np.ndarray:
    if sr not in _IR_CACHE:
        _IR_CACHE[sr] = resposta_reverb(sr)
    return _IR_CACHE[sr]


def reverb(envio: np.ndarray, sr: int) -> np.ndarray:
    """Aplica o reverb (100% molhado) ao barramento de envio (canais, amostras)."""
    from scipy.signal import oaconvolve

    ir = _ir(sr)
    x = envio if envio.shape[0] == 2 else np.repeat(envio[:1], 2, axis=0)
    saida = np.stack([oaconvolve(x[c], ir[c])[: x.shape[1]] for c in range(2)])
    return saida.astype(np.float32)


def wav_ir(sr: int = 44100) -> bytes:
    """A mesma resposta ao impulso em WAV float (para o ConvolverNode do navegador)."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, _ir(sr).T, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def config_com_intensidade(cfg: dict[str, Any], k: float) -> dict[str, Any]:
    """Escala a limpeza calibrada (k=1) para mais leve (k<1). k=0 = sem limpeza; k<0 = modo "suave"."""
    if k < 0:
        return dict(cfg, **CONFIG_SUAVE)
    k = float(np.clip(k, 0, 1))
    c = dict(cfg)
    c["remover_vazamento_bateria"] = float(c.get("remover_vazamento_bateria", 0) or 0) * k
    c["reducao_ruido"] = float(c.get("reducao_ruido", 0) or 0) * k
    if c.get("gate_db") is not None:
        # gate mais permissivo quanto mais leve; abaixo de "leve" desliga
        c["gate_db"] = None if k < 0.35 else float(c["gate_db"]) - (1 - k) * 20
    return c
