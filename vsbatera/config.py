"""Presets (YAML). Um preset descreve como separar, limpar cada faixa e mixar."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PADRAO: dict[str, Any] = {
    # backend: demucs (só Demucs) | duplo (Demucs + RoFormer, média dos dois: mais limpo e ~2x mais lento)
    "separacao": {
        "backend": "demucs", "modelo": "htdemucs_ft", "dispositivo": "auto", "shifts": 2, "overlap": 0.25,
        "modelo_roformer": "BS-Roformer-SW.ckpt",
        "segunda_passada": True,  # no modo duplo: separa de novo a versão sem bateria para tirar o resto
        "guardar_versoes": True,  # no modo duplo: guarda também "só Demucs" e "só RoFormer" para comparar
    },
    # aplicado a toda faixa que não tiver configuração própria em "faixas"
    "padrao_faixa": {
        "ganho_db": 0.0,
        "pan": 0.0,
        "mudo": False,
        "remover_vazamento_bateria": 0.5,
        "passa_alta_hz": 35,
        "passa_baixa_hz": None,
        "reducao_ruido": 0.4,
        "ruido_estacionario": False,
        "gate_db": None,
        "eq": [],
        "compressor": None,
    },
    "faixas": {},
    "mix": {
        "excluir": ["bateria"],
        "bateria_guia_db": None,  # ex.: -18 deixa a bateria original bem baixinha como guia
        "velocidade": 1.0,  # 0.8 = 80% do andamento, sem mudar o tom
        "tom_semitons": 0.0,
        "lufs_alvo": -14.0,
        "teto_db": -1.0,
        "formato": "wav",  # wav | flac | mp3
        "fonte": "faixas",  # faixas (soma das faixas limpas) | residual (original − bateria)
    },
}


def _mesclar(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _mesclar(out[k], v)
        else:
            out[k] = v
    return out


def carregar_preset(caminho: str | Path | None) -> dict[str, Any]:
    if not caminho:
        return copy.deepcopy(PADRAO)
    with open(caminho, encoding="utf-8") as f:
        return _mesclar(PADRAO, yaml.safe_load(f) or {})


def config_da_faixa(preset: dict[str, Any], nome: str) -> dict[str, Any]:
    return _mesclar(preset["padrao_faixa"], preset.get("faixas", {}).get(nome, {}))


def salvar_preset(preset: dict[str, Any], caminho: str | Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        yaml.safe_dump(preset, f, allow_unicode=True, sort_keys=False)
    return caminho
