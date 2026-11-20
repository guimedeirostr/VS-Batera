"""Orquestra as etapas: separar -> analisar -> limpar -> mixar.

Estrutura de uma pasta de projeto (uma por música):
    saida/<musica>/
        original.<ext>             cópia de referência (opcional)
        stems/<faixa>.wav          saída crua da IA
        sem_bateria_residual.wav   original − bateria
        analise.json               métricas por faixa
        preset_sugerido.yaml       ponto de partida da calibração
        limpas/<faixa>.wav         faixas depois da limpeza
        <musica> - sem bateria.wav VERSÃO PARA TOCAR
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import config as cfgmod
from .audio_io import carregar, salvar
from .limpeza import analisar_faixa, limpar_faixa, sugerir_config
from .mix import masterizar, mixar


def pasta_projeto(entrada: str | Path, raiz: str | Path = "saida") -> Path:
    return Path(raiz) / Path(entrada).stem


def _trecho(audio: np.ndarray, sr: int, inicio: float | None, duracao: float | None) -> np.ndarray:
    if inicio is None and duracao is None:
        return audio
    a = int((inicio or 0) * sr)
    b = a + int(duracao * sr) if duracao else audio.shape[1]
    return audio[:, a:b]


def carregar_stems(
    projeto: Path, subpasta: str = "stems", inicio: float | None = None, duracao: float | None = None
) -> tuple[dict[str, np.ndarray], int]:
    pasta = Path(projeto) / subpasta
    arquivos = sorted(pasta.glob("*.wav"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhuma faixa em {pasta}. Rode `vsbatera separar` antes.")
    faixas, sr = {}, 44100
    for arq in arquivos:
        audio, sr = carregar(arq)
        faixas[arq.stem] = _trecho(audio, sr, inicio, duracao)
    return faixas, sr


# --------------------------------------------------------------------------- etapas
def etapa_separar(
    entrada: str | Path, projeto: Path, preset: dict[str, Any], progresso: Callable[[float], None] | None = None
) -> dict[str, Path]:
    from .separacao import separar

    s = preset["separacao"]
    return separar(
        entrada,
        projeto,
        backend=s["backend"],
        modelo=s["modelo"],
        dispositivo=s.get("dispositivo", "auto"),
        shifts=int(s.get("shifts", 2)),
        overlap=float(s.get("overlap", 0.25)),
        progresso=progresso,
    )


def etapa_analisar(projeto: Path, preset: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    faixas, sr = carregar_stems(projeto)
    bateria = faixas.get("bateria")
    analise = {nome: analisar_faixa(a, sr, None if nome == "bateria" else bateria) for nome, a in faixas.items()}
    (projeto / "analise.json").write_text(json.dumps(analise, indent=2, ensure_ascii=False), encoding="utf-8")

    sugerido = cfgmod.carregar_preset(None) if preset is None else json.loads(json.dumps(preset))
    sugerido["faixas"] = {
        nome: sugerir_config(nome, m) for nome, m in analise.items() if nome != "bateria"
    }
    cfgmod.salvar_preset(sugerido, projeto / "preset_sugerido.yaml")
    return analise


def etapa_limpar(
    projeto: Path,
    preset: dict[str, Any],
    inicio: float | None = None,
    duracao: float | None = None,
    progresso: Callable[[float], None] | None = None,
) -> dict[str, Path]:
    faixas, sr = carregar_stems(projeto, "stems", inicio, duracao)
    bateria = faixas.get("bateria")
    destino = projeto / ("limpas" if inicio is None and duracao is None else "previa_limpas")
    saida = {}
    for i, (nome, audio) in enumerate(faixas.items()):
        if progresso:
            progresso(i / len(faixas))
        if nome == "bateria":
            limpo = audio  # a bateria não é limpa: só serve de guia/referência
        else:
            cfg = cfgmod.config_da_faixa(preset, nome)
            print(f"[limpar] {nome}: {_resumo(cfg)}")
            limpo = limpar_faixa(audio, sr, cfg, bateria)
        saida[nome] = salvar(destino / f"{nome}.wav", limpo, sr)
    return saida


def etapa_mixar(
    projeto: Path, preset: dict[str, Any], inicio: float | None = None, duracao: float | None = None
) -> Path:
    m = preset["mix"]
    previa = inicio is not None or duracao is not None
    if m.get("fonte") == "residual":
        audio, sr = carregar(projeto / "sem_bateria_residual.wav")
        audio = _trecho(audio, sr, inicio, duracao)
        from .mix import alterar_velocidade

        audio = alterar_velocidade(audio, sr, float(m.get("velocidade", 1)), float(m.get("tom_semitons", 0)))
        final = masterizar(audio, sr, m.get("lufs_alvo", -14.0), m.get("teto_db", -1.0))
    else:
        sub = "previa_limpas" if previa and (projeto / "previa_limpas").exists() else "limpas"
        if not (projeto / sub).exists():
            sub = "stems"
            print("[mixar] Aviso: faixas ainda não foram limpas, usando as cruas.")
        faixas, sr = carregar_stems(projeto, sub, None if sub == "previa_limpas" else inicio,
                                    None if sub == "previa_limpas" else duracao)
        cfg_faixas = {n: cfgmod.config_da_faixa(preset, n) for n in faixas}
        final = mixar(faixas, sr, cfg_faixas, m)

    sufixo = " - sem bateria"
    if abs(float(m.get("velocidade", 1)) - 1) > 1e-3:
        sufixo += f" ({int(round(float(m['velocidade']) * 100))}%)"
    if previa:
        sufixo += " - PREVIA"
    destino = projeto / f"{projeto.name}{sufixo}.{m.get('formato', 'wav')}"
    salvar(destino, final, sr)
    print(f"[mixar] Pronto: {destino}")
    return destino


def _resumo(cfg: dict[str, Any]) -> str:
    partes = []
    if cfg.get("remover_vazamento_bateria"):
        partes.append(f"vazamento={cfg['remover_vazamento_bateria']}")
    if cfg.get("reducao_ruido"):
        partes.append(f"ruído={cfg['reducao_ruido']}")
    if cfg.get("passa_alta_hz"):
        partes.append(f"HPF={cfg['passa_alta_hz']}Hz")
    if cfg.get("passa_baixa_hz"):
        partes.append(f"LPF={cfg['passa_baixa_hz']}Hz")
    if cfg.get("gate_db") is not None:
        partes.append(f"gate={cfg['gate_db']}dB")
    if cfg.get("eq"):
        partes.append(f"eq={len(cfg['eq'])} bandas")
    if cfg.get("compressor"):
        partes.append("comp")
    if cfg.get("ganho_db"):
        partes.append(f"ganho={cfg['ganho_db']}dB")
    return ", ".join(partes) or "sem processamento"
