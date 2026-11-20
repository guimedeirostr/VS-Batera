"""Linha de comando do VS-Batera.

Exemplos:
    vsbatera tudo musicas/minha.mp3                 # separa, analisa, limpa e gera a versão sem bateria
    vsbatera tudo musicas/minha.mp3 --auto          # idem, usando o preset sugerido pela análise
    vsbatera analisar saida/minha                   # mostra métricas e gera preset_sugerido.yaml
    vsbatera limpar saida/minha -p meu.yaml --inicio 60 --duracao 20   # prévia rápida de 20 s
    vsbatera mixar  saida/minha -p meu.yaml --velocidade 0.8 --guia -20
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import config as cfgmod
from . import pipeline as pl


def _preset(args) -> dict:
    preset = cfgmod.carregar_preset(getattr(args, "preset", None))
    s, m = preset["separacao"], preset["mix"]
    for chave, destino, campo in [
        ("modelo", s, "modelo"),
        ("backend", s, "backend"),
        ("dispositivo", s, "dispositivo"),
        ("shifts", s, "shifts"),
        ("velocidade", m, "velocidade"),
        ("tom", m, "tom_semitons"),
        ("guia", m, "bateria_guia_db"),
        ("lufs", m, "lufs_alvo"),
        ("formato", m, "formato"),
        ("fonte", m, "fonte"),
    ]:
        v = getattr(args, chave, None)
        if v is not None:
            destino[campo] = v
    return preset


def _add_sep(p):
    p.add_argument("--backend", choices=["demucs", "audio-separator"], help="motor de separação")
    p.add_argument("--modelo", help="htdemucs_ft (padrão), htdemucs, htdemucs_6s ou arquivo de modelo UVR")
    p.add_argument("--dispositivo", help="auto | cpu | cuda | mps")
    p.add_argument("--shifts", type=int, help="passadas extras do Demucs (mais = melhor e mais lento)")


def _add_mix(p):
    p.add_argument("--velocidade", type=float, help="ex.: 0.8 = 80%% do andamento, mesmo tom")
    p.add_argument("--tom", type=float, help="transpor em semitons")
    p.add_argument("--guia", type=float, help="deixa a bateria original como guia neste volume (dB), ex. -20")
    p.add_argument("--lufs", type=float, help="volume final (LUFS), padrão -14")
    p.add_argument("--formato", choices=["wav", "flac", "mp3"])
    p.add_argument("--fonte", choices=["faixas", "residual"], help="soma das faixas limpas ou original−bateria")


def _add_trecho(p):
    p.add_argument("--inicio", type=float, help="prévia: início em segundos")
    p.add_argument("--duracao", type=float, help="prévia: duração em segundos")


def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr):  # console do Windows (cp1252) não quebra com acentos/símbolos
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(prog="vsbatera", description="Separa a música e gera a versão sem bateria.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("separar", help="separa a música em faixas com IA")
    p.add_argument("musica")
    p.add_argument("-o", "--saida", default="saida", help="pasta raiz (padrão: saida/)")
    p.add_argument("-p", "--preset")
    _add_sep(p)

    p = sub.add_parser("analisar", help="mede ruído/vazamento de cada faixa e sugere um preset")
    p.add_argument("projeto")
    p.add_argument("-p", "--preset")

    p = sub.add_parser("limpar", help="aplica a limpeza do preset em cada faixa")
    p.add_argument("projeto")
    p.add_argument("-p", "--preset")
    _add_trecho(p)

    p = sub.add_parser("mixar", help="gera a versão final sem bateria")
    p.add_argument("projeto")
    p.add_argument("-p", "--preset")
    _add_mix(p)
    _add_trecho(p)

    p = sub.add_parser("tudo", help="separar + analisar + limpar + mixar")
    p.add_argument("musica")
    p.add_argument("-o", "--saida", default="saida")
    p.add_argument("-p", "--preset")
    p.add_argument("--auto", action="store_true", help="usa o preset sugerido pela análise")
    _add_sep(p)
    _add_mix(p)

    p = sub.add_parser("web", help="abre o painel no navegador: progresso em %%, faixa por faixa e salvar")
    p.add_argument("--porta", type=int, default=int(os.environ.get("PORT", 8765)))
    p.add_argument("--musicas", default="musicas", help="pasta com as músicas (padrão: musicas/)")
    p.add_argument("-o", "--saida", default="saida")
    p.add_argument("--sem-abrir", action="store_true", help="não abre o navegador automaticamente")

    args = ap.parse_args(argv)
    if args.cmd == "web":
        from .web import rodar

        rodar(args.porta, args.musicas, args.saida, abrir=not args.sem_abrir)
        return 0
    preset = _preset(args)

    if args.cmd == "separar":
        projeto = pl.pasta_projeto(args.musica, args.saida)
        for nome, cam in pl.etapa_separar(args.musica, projeto, preset).items():
            print(f"  {nome:22s} {cam}")
    elif args.cmd == "analisar":
        _mostrar_analise(pl.etapa_analisar(Path(args.projeto), preset), Path(args.projeto))
    elif args.cmd == "limpar":
        pl.etapa_limpar(Path(args.projeto), preset, args.inicio, args.duracao)
    elif args.cmd == "mixar":
        pl.etapa_mixar(Path(args.projeto), preset, args.inicio, args.duracao)
    elif args.cmd == "tudo":
        projeto = pl.pasta_projeto(args.musica, args.saida)
        pl.etapa_separar(args.musica, projeto, preset)
        _mostrar_analise(pl.etapa_analisar(projeto, preset), projeto)
        if args.auto:
            sugerido = cfgmod.carregar_preset(projeto / "preset_sugerido.yaml")
            sugerido["mix"] = preset["mix"]
            preset = sugerido
        pl.etapa_limpar(projeto, preset)
        pl.etapa_mixar(projeto, preset)
    return 0


def _mostrar_analise(analise: dict, projeto: Path) -> None:
    print(f"\n{'faixa':12s} {'pico':>7s} {'LUFS':>7s} {'ruído':>7s} {'típico':>7s} {'silêncio':>9s} {'vaz.bat':>8s}")
    for nome, m in analise.items():
        vaz = f"{m['vazamento_bateria_pct']:.1f}%" if "vazamento_bateria_pct" in m else "-"
        print(
            f"{nome:12s} {m['pico_dbfs']:7.1f} {m['lufs']:7.1f} {m['piso_ruido_db']:7.1f} "
            f"{m['nivel_tipico_db']:7.1f} {m['silencio_pct']:8.1f}% {vaz:>8s}"
        )
    print(f"\nPreset sugerido salvo em {projeto / 'preset_sugerido.yaml'}\n")


if __name__ == "__main__":
    sys.exit(main())
