"""Limpeza e calibração de cada faixa separada.

Cadeia aplicada a cada faixa (na ordem), todos os passos são opcionais e configuráveis no preset:
  1. remover_vazamento_bateria  máscara espectral que tira o "vazamento" de bateria que a IA deixou
  2. passa_alta_hz / passa_baixa_hz  corta graves/agudos inúteis (ronco, chiado)
  3. reducao_ruido              redução de ruído espectral (noisereduce)
  4. gate_db                    noise gate: silencia trechos onde o instrumento não está tocando
  5. eq                         equalizador paramétrico (lista de bandas)
  6. compressor                 controla dinâmica
  7. ganho_db                   volume final da faixa
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import istft, stft

N_FFT = 2048
HOP = 512


# --------------------------------------------------------------------------- vazamento de bateria
def remover_vazamento(stem: np.ndarray, bateria: np.ndarray, sr: int, forca: float = 0.7) -> np.ndarray:
    """Atenua bins tempo-frequência onde a bateria domina (filtro de Wiener suave).

    forca: 0 = desligado, 1 = máximo. Valores altos podem deixar o som "aguado" nos ataques.
    """
    if forca <= 0:
        return stem
    forca = float(np.clip(forca, 0.0, 1.0))
    n = min(stem.shape[1], bateria.shape[1])
    saida = np.empty_like(stem[:, :n])
    k = 0.5 + 4.0 * forca  # quanto pesa a bateria na máscara
    piso = 1.0 - 0.9 * forca  # atenuação máxima
    for c in range(stem.shape[0]):
        _, _, S = stft(stem[c, :n], fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
        _, _, D = stft(bateria[c, :n], fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
        ps, pd = np.abs(S) ** 2, np.abs(D) ** 2
        mascara = ps / (ps + k * pd + 1e-12)
        mascara = np.maximum(mascara, piso)
        _, x = istft(S * mascara, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
        saida[c] = x[:n]
    return saida.astype(np.float32)


# --------------------------------------------------------------------------- redução de ruído
def reduzir_ruido(audio: np.ndarray, sr: int, quantidade: float, estacionario: bool = False) -> np.ndarray:
    if quantidade <= 0:
        return audio
    import noisereduce as nr

    out = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=estacionario,
        prop_decrease=float(np.clip(quantidade, 0, 1)),
        n_fft=N_FFT,
    )
    return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- cadeia completa
def construir_cadeia(cfg: dict[str, Any]):
    import pedalboard as pb

    efeitos = []
    if cfg.get("passa_alta_hz"):
        efeitos.append(pb.HighpassFilter(cutoff_frequency_hz=float(cfg["passa_alta_hz"])))
    if cfg.get("passa_baixa_hz"):
        efeitos.append(pb.LowpassFilter(cutoff_frequency_hz=float(cfg["passa_baixa_hz"])))
    if cfg.get("gate_db") is not None:
        efeitos.append(
            pb.NoiseGate(threshold_db=float(cfg["gate_db"]), ratio=8.0, attack_ms=2.0, release_ms=150.0)
        )
    for banda in cfg.get("eq") or []:
        tipo = banda.get("tipo", "pico")
        f, g, q = float(banda["freq"]), float(banda.get("ganho_db", 0)), float(banda.get("q", 0.707))
        if tipo == "grave":
            efeitos.append(pb.LowShelfFilter(cutoff_frequency_hz=f, gain_db=g, q=q))
        elif tipo == "agudo":
            efeitos.append(pb.HighShelfFilter(cutoff_frequency_hz=f, gain_db=g, q=q))
        else:
            efeitos.append(pb.PeakFilter(cutoff_frequency_hz=f, gain_db=g, q=q))
    comp = cfg.get("compressor")
    if comp:
        efeitos.append(
            pb.Compressor(
                threshold_db=float(comp.get("limiar_db", -18)),
                ratio=float(comp.get("razao", 3)),
                attack_ms=float(comp.get("ataque_ms", 10)),
                release_ms=float(comp.get("release_ms", 120)),
            )
        )
    if cfg.get("ganho_db"):
        efeitos.append(pb.Gain(gain_db=float(cfg["ganho_db"])))
    return pb.Pedalboard(efeitos)


def limpar_faixa(
    audio: np.ndarray, sr: int, cfg: dict[str, Any], bateria: np.ndarray | None = None
) -> np.ndarray:
    """Aplica a cadeia de limpeza a uma faixa (canais, amostras)."""
    x = _sem_nan(audio)
    if bateria is not None and cfg.get("remover_vazamento_bateria", 0):
        x = _sem_nan(remover_vazamento(x, _sem_nan(bateria), sr, float(cfg["remover_vazamento_bateria"])))
    # faixa quase muda (ex.: música sem guitarra) faz a redução de ruído dividir 0 por 0 e gerar NaN
    x = _sem_nan(reduzir_ruido(x, sr, float(cfg.get("reducao_ruido", 0) or 0), bool(cfg.get("ruido_estacionario", False))))
    cadeia = construir_cadeia(cfg)
    if len(cadeia):
        x = cadeia(x, sr)
    return _sem_nan(np.asarray(x, dtype=np.float32))


def _sem_nan(x: np.ndarray) -> np.ndarray:
    """Troca NaN/infinito por silêncio (o FLAC se recusa a gravar NaN; o WAV grava lixo)."""
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


# --------------------------------------------------------------------------- análise / auto-calibração
def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-10)))


def analisar_faixa(audio: np.ndarray, sr: int, bateria: np.ndarray | None = None) -> dict[str, float]:
    """Métricas para orientar a calibração."""
    import pyloudnorm as pyln

    mono = audio.mean(0)
    pico = _db(float(np.max(np.abs(audio))))
    try:
        lufs = float(pyln.Meter(sr).integrated_loudness(audio.T))
    except Exception:
        lufs = float("-inf")
    janela = int(0.05 * sr)
    n = len(mono) // janela
    rms = np.sqrt(np.mean(mono[: n * janela].reshape(n, janela) ** 2, axis=1) + 1e-12) if n else np.array([1e-6])
    rms_db = 20 * np.log10(rms)
    ativos = rms_db[rms_db > -80]
    piso = float(np.percentile(ativos, 10)) if len(ativos) else -120.0
    corpo = float(np.percentile(ativos, 80)) if len(ativos) else -120.0
    res = {
        "pico_dbfs": round(pico, 1),
        "lufs": round(lufs, 1),
        "piso_ruido_db": round(piso, 1),
        "nivel_tipico_db": round(corpo, 1),
        "silencio_pct": round(100 * float(np.mean(rms_db < corpo - 30)), 1),
    }
    if bateria is not None:
        n = min(audio.shape[1], bateria.shape[1])
        _, _, S = stft(mono[:n], fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
        _, _, D = stft(bateria.mean(0)[:n], fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
        ps, pd = np.abs(S) ** 2, np.abs(D) ** 2
        dominado = pd > ps
        res["vazamento_bateria_pct"] = round(100 * float(ps[dominado].sum() / (ps.sum() + 1e-12)), 1)
    return res


def sugerir_config(nome: str, m: dict[str, float]) -> dict[str, Any]:
    """Gera uma configuração inicial a partir da análise. Ajuste de ouvido depois."""
    cfg: dict[str, Any] = {"ganho_db": 0.0, "pan": 0.0}
    cfg["passa_alta_hz"] = {"baixo": 30, "voz": 80, "guitarra": 80, "teclado": 40}.get(nome, 40)
    if nome != "baixo":
        cfg["passa_baixa_hz"] = None
    else:
        cfg["passa_baixa_hz"] = 8000
    vaz = m.get("vazamento_bateria_pct", 0)
    cfg["remover_vazamento_bateria"] = round(float(np.clip(vaz / 25, 0.2, 0.9)), 2)
    margem = m["nivel_tipico_db"] - m["piso_ruido_db"]
    cfg["reducao_ruido"] = 0.3 if margem > 40 else 0.5 if margem > 25 else 0.7
    cfg["ruido_estacionario"] = False
    cfg["gate_db"] = round(m["piso_ruido_db"] + 0.35 * margem, 0) if m["silencio_pct"] > 10 else None
    cfg["eq"] = []
    cfg["compressor"] = None
    return cfg
