"""Reconstrução de faixas com o Apollo (restauração de áudio, Kai Li & Yi Luo, 2024).

A limpeza (redução de ruído + máscara anti-vazamento) tira junto pedaços do timbre e a faixa fica
"robótica". O Apollo é um modelo de restauração: recompõe brilho/harmônicos perdidos SEM mudar notas
nem tempo (não é uma "nova versão" gerada).

A rede está em apollo_rede.py (código oficial dos autores); os pesos oficiais (JusperLee/Apollo no
Hugging Face, 66 MB) são baixados uma vez para VSBATERA_MODELOS/apollo e lidos com um leitor restrito
que não executa código do arquivo (ver _ler_pesos). Licença CC BY-SA 4.0.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import arquivo_faixa, carregar, reamostrar, salvar

# pesos oficiais dos autores (restauração de música comprimida/degradada), 44.1 kHz
REPO = "JusperLee/Apollo"
ARQUIVO_PESOS = "pytorch_model.bin"
CONFIG = {"sr": 44100, "win": 20, "feature_dim": 256, "layer": 6}
SR = 44100
_MODELO = {}


def _pasta_modelos() -> Path:
    return Path(os.environ.get("VSBATERA_MODELOS", Path.home() / ".vsbatera" / "modelos")) / "apollo"


def _dispositivo(d: str = "auto") -> str:
    import torch

    if d != "auto":
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


def carregar_modelo(dispositivo: str = "auto"):
    """Baixa (na primeira vez) os pesos oficiais e monta a rede (código em apollo_rede.py)."""
    import torch
    from huggingface_hub import hf_hub_download

    from .apollo_rede import Apollo

    dispositivo = _dispositivo(dispositivo)
    if dispositivo in _MODELO:
        return _MODELO[dispositivo]
    pasta = _pasta_modelos()
    pesos = pasta / ARQUIVO_PESOS
    if not pesos.exists():
        print("[reconstruir] baixando os pesos do Apollo (66 MB, só na primeira vez)…", flush=True)
        hf_hub_download(REPO, ARQUIVO_PESOS, local_dir=str(pasta))
    estado = _ler_pesos(pesos)
    if isinstance(estado, dict) and "state_dict" in estado:
        estado = estado["state_dict"]
    estado = {k[len("audio_model."):] if k.startswith("audio_model.") else k: v for k, v in estado.items()}
    modelo = Apollo(**CONFIG)
    modelo.load_state_dict(estado)
    modelo = modelo.to(dispositivo).eval()
    _MODELO[dispositivo] = modelo
    return modelo


class _Vazio:
    """Substitui objetos de configuração do treino (omegaconf etc.): aceita qualquer estado e não faz nada."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, estado):
        pass


def _ler_pesos(caminho: Path):
    """Lê o checkpoint sem executar código: só reconstrói tensores e tipos simples.

    O checkpoint oficial traz junto a configuração do treino (omegaconf), que o modo weights_only do PyTorch
    recusa. Em vez de liberar tudo (weights_only=False puro, que pode executar código), usamos um Unpickler
    restrito: tensores/storages do torch, OrderedDict e tipos básicos são permitidos; qualquer coisa de
    omegaconf/typing vira um objeto vazio; o resto é recusado com erro.
    """
    import pickle
    import types

    import torch

    permitidos = {
        ("collections", "OrderedDict"), ("collections", "defaultdict"),
        ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_tensor_v3"), ("torch.nn.parameter", "Parameter"),
        ("builtins", "dict"), ("builtins", "list"), ("builtins", "set"), ("builtins", "tuple"),
        ("builtins", "int"), ("builtins", "float"), ("builtins", "str"), ("builtins", "bool"),
        ("builtins", "frozenset"), ("builtins", "bytes"),
    }

    class Restrito(pickle.Unpickler):
        def find_class(self, modulo, nome):
            modulo = {"__builtin__": "builtins"}.get(modulo, modulo)  # nomes do Python 2 em pickles antigos
            if (modulo, nome) == ("builtins", "long"):
                return int
            if modulo.startswith(("omegaconf", "typing", "pathlib")) or (modulo, nome) == ("torch.torch_version", "TorchVersion"):
                return _Vazio
            if (modulo, nome) in permitidos or (modulo == "torch" and (nome.endswith("Storage") or nome in (
                    "Size", "device", "float32", "float16", "bfloat16", "int64", "int32", "bool"))):
                return super().find_class(modulo, nome)
            raise pickle.UnpicklingError(f"checkpoint pede {modulo}.{nome}, que não é permitido")

    modulo_seguro = types.SimpleNamespace(Unpickler=Restrito, load=pickle.load, __name__="pickle_restrito")
    return torch.load(caminho, map_location="cpu", weights_only=False, pickle_module=modulo_seguro)


def restaurar(audio: np.ndarray, sr: int, dispositivo: str = "auto",
              progresso: Callable[[float], None] | None = None) -> tuple[np.ndarray, int]:
    """audio (canais, amostras) -> audio restaurado (canais, amostras) a 44.1 kHz.

    Processa em blocos de 10 s com 5 s de passo; o fim de cada bloco é descartado (como no app oficial),
    então não há emendas audíveis.
    """
    import torch

    if sr != SR:
        audio, sr = reamostrar(audio, sr, SR), SR
    dispositivo = _dispositivo(dispositivo)
    modelo = carregar_modelo(dispositivo)
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    if x.ndim == 1:
        x = x.unsqueeze(0)
    n_orig = x.shape[1]

    C = 10 * SR
    passo = C // 2
    borda = C - passo
    fade = 3 * SR
    pad = x.shape[1] > 2 * borda
    if pad:
        x = torch.nn.functional.pad(x.unsqueeze(0), (borda, borda), mode="reflect").squeeze(0)
    janela_base = torch.ones(C)
    janela_base[-fade:] = 0  # descarta o fim de cada bloco (região onde o modelo "erra")

    total = x.shape[1]
    resultado = torch.zeros_like(x)
    contador = torch.zeros_like(x)
    i = 0
    while i < total:
        parte = x[:, i:i + C]
        L = parte.shape[-1]
        if L < C:
            modo = "reflect" if L > C // 2 + 1 else "constant"
            parte = torch.nn.functional.pad(parte.unsqueeze(0), (0, C - L), mode=modo).squeeze(0)
        with torch.no_grad():
            saida = modelo(parte.unsqueeze(0).to(dispositivo)).squeeze(0).cpu()
        if saida.ndim == 1:
            saida = saida.unsqueeze(0)
        janela = janela_base.clone()
        if i == 0:
            janela[:fade] = 1
        if i + C >= total:
            janela[-fade:] = 1
        resultado[:, i:i + L] += saida[:, :L] * janela[:L]
        contador[:, i:i + L] += janela[:L]
        i += passo
        if progresso:
            progresso(min(1.0, i / total))

    y = (resultado / contador.clamp_min(1e-8)).numpy()
    if pad:
        y = y[:, borda:-borda]
    y = np.nan_to_num(y[:, :n_orig], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y, SR


def reconstruir_faixa(projeto: str | Path, faixa: str, dispositivo: str = "auto",
                      progresso: Callable[[float], None] | None = None) -> Path:
    """Reconstrói limpas/<faixa> (ou stems/<faixa> se não houver limpa) em reconstruidas/<faixa>."""
    from .audio_io import achar_arquivo

    projeto = Path(projeto)
    origem = achar_arquivo(projeto / "limpas", faixa) or achar_arquivo(projeto / "stems", faixa)
    if origem is None:
        raise FileNotFoundError(f"faixa {faixa!r} não encontrada em {projeto}")
    audio, sr = carregar(origem)
    y, sr = restaurar(audio, sr, dispositivo, progresso)
    # mantém o volume parecido com o da faixa de origem (o modelo pode mexer no nível)
    rms_o = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
    rms_y = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    y = (y * min(4.0, rms_o / rms_y)).astype(np.float32)
    for velho in (projeto / "reconstruidas").glob(f"{faixa}.*"):  # não deixa .wav e .flac da mesma faixa
        velho.unlink()
    destino = salvar(arquivo_faixa(projeto / "reconstruidas", faixa), y, sr)
    print(f"[reconstruir] {faixa}: pronta → {destino}", flush=True)
    return destino
