# VS-Batera 🥁

Sistema para **separar os instrumentos de uma música com IA, limpar cada faixa e gerar a versão
sem bateria** — pra você tocar junto.

```
música.mp3 ──► [1. separar] ──► bateria / baixo / voz / guitarra / teclado / outros
                                   │
                                   ▼
                [2. analisar] ──► ruído, vazamento de bateria, volume  ──► preset_sugerido.yaml
                                   │
                                   ▼
                [3. limpar]  ──► por faixa: tira vazamento de bateria, filtros, redução de ruído,
                                   gate, EQ, compressor, ganho
                                   │
                                   ▼
                [4. mixar]   ──► soma tudo MENOS a bateria, pan, velocidade/tom, loudness, limiter
                                   │
                                   ▼
                         "<música> - sem bateria.wav"   ← versão pra tocar
```

## Pesquisa: qual tecnologia usar

| Opção | O que é | Por que (não) |
|---|---|---|
| **Demucs v4 – `htdemucs_ft`** (Meta, open source) | Hybrid Transformer, 4 faixas | **Padrão do sistema.** Na bateria empata com o estado da arte e roda localmente, de graça. |
| **Demucs `htdemucs_6s`** | 6 faixas (+ guitarra, teclado) | Use quando quiser limpar/ajustar guitarra e teclado separados. |
| **BS-RoFormer / Mel-RoFormer** (via [`audio-separator`](https://github.com/nomadkaraoke/python-audio-separator), modelos do UVR) | Melhor arquitetura geral (venceu o SDX23) | Suportado como backend opcional (`--backend audio-separator`). Ganha mais no baixo/voz; na bateria a diferença é pequena. |
| **UVR5** (Ultimate Vocal Remover) | App desktop com os mesmos modelos | Bom pra testar modelos na mão; não é automatizável como aqui. |
| **Plugin AudioPod** (diretório de plugins do Claude) | Serviço online pago: stems + limpeza de ruído | Alternativa sem instalar nada, mas manda sua música pra nuvem e cobra por uso. |
| Spleeter | Modelo antigo (2019) | Qualidade bem inferior; descartado. |

Truque importante: o sistema também grava **`sem_bateria_residual.wav` = original − bateria**. Como não
reconstrói os outros instrumentos, ele soa mais "inteiro"; a desvantagem é que qualquer resto de bateria
que a IA não pegou fica lá. Você escolhe com `--fonte residual` ou `--fonte faixas` (padrão, com limpeza).

## Instalação (no seu computador)

Requer Python 3.10+ e [ffmpeg](https://ffmpeg.org/) (para ler MP3/M4A e exportar MP3).

```bash
git clone <este repositório> && cd VS-Batera
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .                     # instala o comando `vsbatera`
pip install -e ".[roformer]"         # (opcional) modelos BS-RoFormer/MDX do UVR
```

- **GPU NVIDIA**: instale o PyTorch com CUDA (https://pytorch.org) — fica ~10× mais rápido.
- **Mac M1/M2/M3**: usa `mps` automaticamente.
- Na primeira execução o Demucs baixa os pesos do modelo (~300 MB para o `htdemucs_ft`).

## Uso rápido

```bash
# Tudo de uma vez, já com a calibração automática sugerida pela análise:
vsbatera tudo musicas/minha_musica.mp3 --auto

# Versão de estudo a 80% da velocidade, com a bateria original bem baixinha de guia:
vsbatera mixar saida/minha_musica --velocidade 0.8 --guia -22
```

Resultado em `saida/minha_musica/minha_musica - sem bateria.wav`.

## Fluxo de calibração (pra chegar no "som perfeito")

1. **Separar** (só uma vez por música — é a parte demorada):
   ```bash
   vsbatera separar musicas/minha.mp3 --modelo htdemucs_6s
   ```
2. **Analisar** — mostra, por faixa, pico, LUFS, piso de ruído, % de silêncio e % de vazamento de bateria,
   e gera `saida/minha/preset_sugerido.yaml`:
   ```bash
   vsbatera analisar saida/minha
   ```
3. **Copiar o preset e ajustar de ouvido** (veja `presets/exemplo_rock.yaml`, todo comentado):
   ```bash
   cp saida/minha/preset_sugerido.yaml presets/minha.yaml
   ```
4. **Prévia rápida de um trecho** (ex.: 20 s a partir do 1:00) — ouça os arquivos em
   `saida/minha/previa_limpas/` e a mix `... - PREVIA.wav`:
   ```bash
   vsbatera limpar saida/minha -p presets/minha.yaml --inicio 60 --duracao 20
   vsbatera mixar  saida/minha -p presets/minha.yaml --inicio 60 --duracao 20
   ```
   Repita 3 ↔ 4 até gostar.
5. **Render final**:
   ```bash
   vsbatera limpar saida/minha -p presets/minha.yaml
   vsbatera mixar  saida/minha -p presets/minha.yaml --formato mp3
   ```

### Parâmetros por faixa (`faixas: <nome>:`)

| Parâmetro | Faixa típica | O que faz |
|---|---|---|
| `remover_vazamento_bateria` | 0 – 1 (0.5) | Máscara espectral que tira restos de bateria (chimbal na voz, bumbo no baixo). Alto demais deixa os ataques "aguados". |
| `passa_alta_hz` / `passa_baixa_hz` | 30–120 / 5000–16000 | Corta ronco grave / chiado agudo. No baixo, `passa_baixa_hz: 6000` some com o "tic" de caixa/chimbal. |
| `reducao_ruido` | 0 – 1 (0.3–0.6) | Redução de ruído espectral. `ruido_estacionario: true` para chiado constante (gravação antiga/ao vivo). |
| `gate_db` | -60 a -40 | Silencia a faixa quando o instrumento não está tocando (bom para voz). |
| `eq` | lista | Bandas `{freq, ganho_db, q, tipo: pico\|grave\|agudo}`. |
| `compressor` | | `{limiar_db, razao, ataque_ms, release_ms}`. |
| `ganho_db`, `pan`, `mudo` | | Volume, posição estéreo (-1 a 1), silenciar a faixa. |

### Opções da mix (`mix:`)

`excluir` (padrão `[bateria]` — dá pra tirar outras também, ex. `[bateria, voz]`), `bateria_guia_db`,
`velocidade`, `tom_semitons`, `lufs_alvo`, `teto_db`, `formato`, `fonte`.

## Dicas de batera

- **Estudar devagar**: `--velocidade 0.7` mantém o tom (Rubber Band).
- **Guia**: `--guia -25` deixa a bateria original quase inaudível — ótimo para conferir convenções/viradas.
- Se o **baixo** perder ataque, diminua `remover_vazamento_bateria` dele (o bumbo e o baixo se sobrepõem).
- Se ainda ouvir **pratos** no fundo, aumente `remover_vazamento_bateria` em `outros`/`voz`, ou teste
  `--fonte residual` com um modelo RoFormer.

## Testes

```bash
pip install -e ".[test]" && pytest -q
```
Os testes usam áudio sintético e um modelo Demucs de pesos aleatórios, sem baixar nada.

## Estrutura

```
vsbatera/
  separacao.py   Demucs / audio-separator (RoFormer, MDX)
  limpeza.py     remoção de vazamento, ruído, filtros, gate, EQ, compressor, análise e auto-calibração
  mix.py         soma sem bateria, pan, velocidade/tom, loudness, limiter
  pipeline.py    etapas e organização das pastas
  cli.py         comando `vsbatera`
presets/         presets de exemplo
tests/
```
