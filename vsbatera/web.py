"""Painel web local: processa músicas mostrando a % de cada etapa, toca faixa por faixa e salva a mix.

    vsbatera web                 # abre http://localhost:8765 no navegador

Só usa a biblioteca padrão do Python (http.server). Tudo roda no seu computador.
"""
from __future__ import annotations

import glob
import io
import json
import mimetypes
import re
import sys
import threading
import time
import traceback
import unicodedata
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from . import config as cfgmod
from . import pipeline as pl
from .audio_io import achar_faixas

EXT_AUDIO = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
STATIC = Path(__file__).parent / "static"

# Peso de cada etapa no progresso total (a separação com IA é a parte demorada).
ETAPAS = [("separar", 0.80), ("analisar", 0.05), ("limpar", 0.10), ("mixar", 0.05)]


class Tarefa:
    """Estado da tarefa em andamento (uma por vez), lido pela página a cada segundo."""

    def __init__(self) -> None:
        self.trava = threading.Lock()
        self.estado: dict[str, Any] = {"rodando": False}

    def iniciar(self, tipo: str, projeto: str, etapas: list[tuple[str, float]]) -> None:
        with self.trava:
            self.estado = {
                "rodando": True,
                "tipo": tipo,
                "projeto": projeto,
                "etapas": [{"nome": n, "peso": p, "pct": 0.0} for n, p in etapas],
                "etapa": etapas[0][0],
                "pct_total": 0.0,
                "inicio": time.time(),
                "fim": None,
                "erro": None,
                "resultado": None,
                "log": [],
            }

    def progresso(self, etapa: str, frac: float) -> None:
        with self.trava:
            e = self.estado
            total = 0.0
            for item in e.get("etapas", []):
                if item["nome"] == etapa:
                    item["pct"] = max(item["pct"], round(min(1.0, frac) * 100, 1))
                    e["etapa"] = etapa
                total += item["peso"] * item["pct"]
            peso = sum(i["peso"] for i in e.get("etapas", [])) or 1
            e["pct_total"] = round(total / peso, 1)

    def log(self, texto: str) -> None:
        with self.trava:
            if self.estado.get("rodando"):
                self.estado["log"] = (self.estado["log"] + [texto])[-200:]

    def terminar(self, resultado: Any = None, erro: str | None = None) -> None:
        with self.trava:
            self.estado.update(rodando=False, fim=time.time(), resultado=resultado, erro=erro)
            if not erro:
                for item in self.estado.get("etapas", []):
                    item["pct"] = 100.0
                self.estado["pct_total"] = 100.0

    def copia(self) -> dict[str, Any]:
        with self.trava:
            return json.loads(json.dumps(self.estado))


TAREFA = Tarefa()


class _Tee(io.TextIOBase):
    """Copia tudo que o pipeline imprime para o log da página (e continua mostrando no terminal)."""

    def __init__(self, original):
        self.original = original

    def write(self, s: str) -> int:
        try:
            self.original.write(s)
        except Exception:
            pass
        for linha in re.split(r"[\r\n]+", s):
            linha = linha.strip()
            if linha and "%|" not in linha:  # ignora as barras do tqdm (já temos a nossa %)
                TAREFA.log(linha)
        return len(s)

    def flush(self) -> None:
        try:
            self.original.flush()
        except Exception:
            pass


# --------------------------------------------------------------------------- trabalhos
def _processar(musica: Path, raiz: Path, auto: bool, modelo: str | None, modo: str = "simples") -> None:
    try:
        preset = cfgmod.carregar_preset(None)
        if modelo:
            preset["separacao"]["modelo"] = modelo
        if modo == "duplo":
            preset["separacao"]["backend"] = "duplo"
        projeto = pl.pasta_projeto(musica, raiz)
        pl.etapa_separar(musica, projeto, preset, progresso=lambda f: TAREFA.progresso("separar", f))
        TAREFA.progresso("separar", 1)
        pl.etapa_analisar(projeto, preset)
        TAREFA.progresso("analisar", 1)
        if auto:
            sugerido = cfgmod.carregar_preset(projeto / "preset_sugerido.yaml")
            sugerido["mix"] = preset["mix"]
            preset = sugerido
        pl.etapa_limpar(projeto, preset, progresso=lambda f: TAREFA.progresso("limpar", f))
        TAREFA.progresso("limpar", 1)
        final = pl.etapa_mixar(projeto, preset)
        TAREFA.terminar({"projeto": projeto.name, "arquivo": final.name})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        TAREFA.terminar(erro=f"{type(e).__name__}: {e}")


def _salvar_mix(projeto: Path, pedido: dict[str, Any]) -> Path:
    """Gera a versão final com os volumes/mudos escolhidos no painel e grava o preset usado."""
    import numpy as np

    from .audio_io import salvar
    from .mix import mixar

    base = projeto / "preset_web.yaml"
    if not base.exists():
        base = projeto / "preset_sugerido.yaml"
    preset = cfgmod.carregar_preset(base if base.exists() else None)
    m = preset["mix"]
    for chave in ("velocidade", "tom_semitons", "lufs_alvo", "formato", "fonte"):
        if pedido.get(chave) not in (None, ""):
            m[chave] = pedido[chave]

    conf = pedido.get("faixas", {})
    bat = conf.get("bateria", {})
    m["bateria_guia_db"] = None if bat.get("mudo", True) else float(bat.get("volume_db", -18))
    for nome, c in conf.items():
        if nome == "bateria":
            continue
        f = preset.setdefault("faixas", {}).setdefault(nome, {})
        f["mudo"] = bool(c.get("mudo"))
        f["volume_mix_db"] = float(c.get("volume_db", 0))
        if "pan" in c:
            f["pan"] = float(c["pan"])

    sub = pedido.get("conjunto") or ("limpas" if (projeto / "limpas").exists() else "stems")
    faixas, sr = pl.carregar_stems(projeto, sub)
    for nome in list(faixas):
        if nome == "bateria":  # a bateria só entra como guia, no volume de bateria_guia_db
            continue
        db = float(conf.get(nome, {}).get("volume_db", 0))
        if db:
            faixas[nome] = (faixas[nome] * (10 ** (db / 20))).astype(np.float32)
    cfg_faixas = {n: cfgmod.config_da_faixa(preset, n) for n in faixas}
    final = mixar(faixas, sr, cfg_faixas, m)

    nome = (pedido.get("nome") or f"{projeto.name} - minha mix").strip()
    nome = re.sub(r'[\\/:*?"<>|]+', "_", nome)
    destino = projeto / f"{nome}.{m.get('formato', 'wav')}"
    salvar(destino, final, sr)
    cfgmod.salvar_preset(preset, projeto / "preset_web.yaml")
    print(f"[salvar] Pronto: {destino}")
    return destino


# --------------------------------------------------------------------------- Colab
def ler_status_colab(saida: Path) -> dict[str, Any]:
    """Lê o arquivo que o notebook do Colab grava no Drive. "vivo" = atualizado nos últimos 3 minutos."""
    try:
        st = json.loads((saida / "_colab_status.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"vivo": False}
    st["vivo"] = time.time() - float(st.get("atualizado", 0)) < 180
    # o Colab (Linux) grava acentos decompostos (NFD: "a" + til); o Windows usa NFC ("ã"). Unifica.
    nfc = lambda t: unicodedata.normalize("NFC", t) if isinstance(t, str) else t  # noqa: E731
    st["processando"] = nfc(st.get("processando"))
    for k in ("fila", "erros"):
        st[k] = [nfc(x) for x in st.get(k) or []]
    return st


def colab_pronta(saida: Path, stem: str) -> bool:
    proj = saida / stem
    return bool(achar_faixas(proj / "limpas")) and any(proj.glob(f"{glob.escape(stem)} - sem bateria.*"))



# --------------------------------------------------------------------------- fila
class Fila:
    """Fila de músicas para processar uma depois da outra. Fica salva em saida/fila.json,
    então se o painel for reiniciado ela continua de onde parou."""

    def __init__(self) -> None:
        self.trava = threading.Lock()
        self.itens: list[dict[str, Any]] = []
        self.arquivo: Path | None = None
        self.raiz_musicas: Path | None = None
        self.raiz_saida: Path | None = None
        self.acordar = threading.Event()

    def configurar(self, musicas: Path, saida: Path) -> None:
        self.raiz_musicas, self.raiz_saida = musicas, saida
        self.arquivo = saida / "fila.json"
        if self.arquivo.exists():
            try:
                self.itens = json.loads(self.arquivo.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.itens = []
        for it in self.itens:  # o que estava rodando quando o painel caiu volta para a fila
            if it["status"] == "processando":
                it["status"] = "esperando"
        self._gravar()
        threading.Thread(target=self._trabalhar, daemon=True).start()

    def _gravar(self) -> None:
        if self.arquivo:
            self.arquivo.write_text(json.dumps(self.itens, ensure_ascii=False, indent=1), encoding="utf-8")

    def adicionar(self, nomes: list[str], auto: bool = True, modelo: str | None = None,
                  modo: str = "duplo") -> int:
        n = 0
        with self.trava:
            pendentes = {i["musica"] for i in self.itens if i["status"] in ("esperando", "processando")}
            for nome in nomes:
                if nome in pendentes:
                    continue
                self.itens = [i for i in self.itens if i["musica"] != nome]  # tira feitas/erros antigos
                self.itens.append({"musica": nome, "auto": auto, "modelo": modelo, "modo": modo, "status": "esperando",
                                   "erro": None, "inicio": None, "fim": None})
                n += 1
            self._gravar()
        self.acordar.set()
        return n

    def remover(self, nome: str) -> None:
        with self.trava:
            self.itens = [i for i in self.itens if not (i["musica"] == nome and i["status"] != "processando")]
            self._gravar()

    def limpar_concluidas(self) -> None:
        with self.trava:
            self.itens = [i for i in self.itens if i["status"] in ("esperando", "processando")]
            self._gravar()

    def copia(self) -> list[dict[str, Any]]:
        with self.trava:
            self._atualizar_colab()
            return json.loads(json.dumps(self.itens))

    def _atualizar_colab(self) -> None:
        """Itens com modo "colab" são processados pelo notebook do Colab (pasta compartilhada pelo Drive).
        O painel só acompanha: lê saida/_colab_status.json e vê quando o resultado aparece."""
        if not self.raiz_saida or not any(i.get("modo") == "colab" for i in self.itens):
            return
        st = ler_status_colab(self.raiz_saida)
        mudou = False
        for it in self.itens:
            if it.get("modo") != "colab" or it["status"] in ("feita", "erro"):
                continue
            stem = unicodedata.normalize("NFC", Path(it["musica"]).stem)
            if colab_pronta(self.raiz_saida, Path(it["musica"]).stem):
                it.update(status="feita", fim=time.time(), pct=100)
                mudou = True
            elif st.get("processando") == stem and st.get("vivo"):
                if it["status"] != "processando":
                    it.update(status="processando", inicio=it.get("inicio") or time.time())
                    mudou = True
                it["pct"], it["etapa"] = st.get("pct", 0), st.get("etapa")
            elif stem in (st.get("erros") or []):
                it.update(status="erro", erro="o Colab não conseguiu processar (veja o notebook)", fim=time.time())
                mudou = True
            elif it["status"] == "processando":
                it["status"] = "esperando"
                mudou = True
        if mudou:
            self._gravar()

    def _proximo(self) -> dict[str, Any] | None:
        with self.trava:
            for it in self.itens:
                if it["status"] == "esperando" and it.get("modo") != "colab":
                    it["status"], it["inicio"] = "processando", time.time()
                    self._gravar()
                    return it
        return None

    def _trabalhar(self) -> None:
        while True:
            it = self._proximo()
            if it is None:
                _manter_acordado(False)
                self.acordar.wait(5)
                self.acordar.clear()
                continue
            _manter_acordado(True)
            musica = self.raiz_musicas / it["musica"]
            if not musica.exists():
                erro = "arquivo não encontrado"
            else:
                TAREFA.iniciar("processar", musica.stem, ETAPAS)
                _processar(musica, self.raiz_saida, it.get("auto", True), it.get("modelo"), it.get("modo", "simples"))
                erro = TAREFA.copia().get("erro")
            with self.trava:
                it["status"], it["erro"], it["fim"] = ("erro" if erro else "feita"), erro, time.time()
                self._gravar()


FILA = Fila()


def _manter_acordado(sim: bool) -> None:
    """No Windows, impede o PC de entrar em suspensão enquanto a fila trabalha (a tela pode apagar)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if sim else 0))
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- servidor
class App:
    def __init__(self, raiz_musicas: Path, raiz_saida: Path) -> None:
        self.musicas = raiz_musicas.resolve()
        self.saida = raiz_saida.resolve()
        self.musicas.mkdir(parents=True, exist_ok=True)
        self.saida.mkdir(parents=True, exist_ok=True)

    def listar_musicas(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.musicas.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.suffix.lower() in EXT_AUDIO:
                proj = self.saida / p.stem
                out.append({
                    "nome": p.name,
                    "tamanho_mb": round(p.stat().st_size / 1e6, 1),
                    "processada": bool(achar_faixas(proj / "stems")),
                })
        return out

    def info_projeto(self, nome: str) -> dict[str, Any]:
        proj = self._projeto(nome)

        def url(p: Path) -> str:
            return "/arquivo/" + quote(nome) + "/" + quote(p.relative_to(proj).as_posix())

        conjuntos = {}
        for sub in ("limpas", "stems", "stems_roformer", "stems_demucs"):
            pasta = proj / sub
            if achar_faixas(pasta):
                conjuntos[sub] = [{"nome": p.stem, "url": url(p)} for p in achar_faixas(pasta)]
        duplo = "stems_roformer" in conjuntos
        finais = [
            {"nome": p.name, "url": url(p), "mb": round(p.stat().st_size / 1e6, 1)}
            for p in sorted(proj.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            if p.is_file() and p.suffix.lower() in EXT_AUDIO
        ]
        original = next((m for m in self.musicas.iterdir() if m.stem == nome and m.suffix.lower() in EXT_AUDIO), None)
        analise = {}
        if (proj / "analise.json").exists():
            analise = json.loads((proj / "analise.json").read_text(encoding="utf-8"))
        salvo = {}
        if (proj / "preset_web.yaml").exists():
            pr = cfgmod.carregar_preset(proj / "preset_web.yaml")
            salvo = {
                "faixas": {n: {"volume_db": f.get("volume_mix_db", 0), "mudo": f.get("mudo", False),
                               "pan": f.get("pan", 0)} for n, f in pr.get("faixas", {}).items()},
                "guia_db": pr["mix"].get("bateria_guia_db"),
                "velocidade": pr["mix"].get("velocidade", 1.0),
                "formato": pr["mix"].get("formato", "wav"),
            }
        return {
            "nome": nome,
            "conjuntos": conjuntos,
            "duplo": duplo,
            "finais": finais,
            "original": ("/musica/" + quote(original.name)) if original else None,
            "analise": analise,
            "salvo": salvo,
        }

    def _projeto(self, nome: str) -> Path:
        proj = (self.saida / nome).resolve()
        if proj.parent != self.saida or not proj.is_dir():
            raise FileNotFoundError(nome)
        return proj

    def caminho_seguro(self, base: Path, rel: str) -> Path:
        p = (base / rel).resolve()
        if base not in p.parents or not p.is_file():
            raise FileNotFoundError(rel)
        return p


def _handler(app: App):
    class H(BaseHTTPRequestHandler):
        server_version = "VSBatera"

        def log_message(self, fmt, *args):  # silencia o log de cada requisição
            pass

        # ---- respostas
        def _json(self, dados: Any, status: int = 200) -> None:
            corpo = json.dumps(dados, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corpo)

        def _arquivo(self, p: Path, baixar: bool = False) -> None:
            tamanho = p.stat().st_size
            tipo = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            ini, fim = 0, tamanho - 1
            faixa = self.headers.get("Range")
            m = re.match(r"bytes=(\d*)-(\d*)", faixa or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    ini = int(m.group(1))
                    fim = int(m.group(2)) if m.group(2) else tamanho - 1
                else:
                    ini = max(0, tamanho - int(m.group(2)))
                fim = min(fim, tamanho - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {ini}-{fim}/{tamanho}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(fim - ini + 1))
            if baixar:
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(p.name))
            self.end_headers()
            try:
                with open(p, "rb") as f:
                    f.seek(ini)
                    resta = fim - ini + 1
                    while resta > 0:
                        bloco = f.read(min(1 << 20, resta))
                        if not bloco:
                            break
                        self.wfile.write(bloco)
                        resta -= len(bloco)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

        def _corpo_json(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        # ---- rotas
        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            caminho = unquote(u.path)
            try:
                if caminho in ("/", "/index.html"):
                    return self._arquivo(STATIC / "index.html")
                if caminho == "/api/musicas":
                    return self._json(app.listar_musicas())
                if caminho == "/api/youtube":
                    return self._json(DOWNLOADS)
                if caminho == "/api/status":
                    e = TAREFA.copia()
                    e["fila"] = FILA.copia()
                    e["colab"] = ler_status_colab(app.saida)
                    return self._json(e)
                if caminho.startswith("/api/projeto/"):
                    return self._json(app.info_projeto(caminho[len("/api/projeto/"):]))
                if caminho.startswith("/arquivo/"):
                    nome, _, rel = caminho[len("/arquivo/"):].partition("/")
                    p = app.caminho_seguro(app._projeto(nome), rel)
                    return self._arquivo(p, baixar="baixar" in q)
                if caminho.startswith("/musica/"):
                    return self._arquivo(app.caminho_seguro(app.musicas, caminho[len("/musica/"):]))
                self._json({"erro": "não encontrado"}, 404)
            except FileNotFoundError:
                self._json({"erro": "não encontrado"}, 404)

        def do_POST(self):  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/api/upload":
                    nome = Path(unquote(q.get("nome", [""])[0])).name
                    if Path(nome).suffix.lower() not in EXT_AUDIO:
                        return self._json({"erro": "formato não suportado"}, 400)
                    n = int(self.headers.get("Content-Length") or 0)
                    destino = app.musicas / nome
                    with open(destino, "wb") as f:
                        resta = n
                        while resta > 0:
                            bloco = self.rfile.read(min(1 << 20, resta))
                            if not bloco:
                                break
                            f.write(bloco)
                            resta -= len(bloco)
                    return self._json({"ok": True, "nome": nome})

                if u.path == "/api/youtube":
                    d = self._corpo_json()
                    url = str(d.get("url", "")).strip()
                    if not re.match(r"^https?://", url):
                        return self._json({"erro": "cole um link começando com https://"}, 400)
                    item = {"id": int(time.time() * 1000), "url": url, "status": "começando", "pct": 0,
                            "nome": None, "erro": None, "fila": bool(d.get("fila", True)),
                            "auto": bool(d.get("auto", True)), "modo": d.get("modo", "duplo")}
                    with _LOCK_DL:
                        DOWNLOADS.append(item)
                        del DOWNLOADS[:-20]
                    threading.Thread(target=_baixar_link, args=(app, item), daemon=True).start()
                    return self._json({"ok": True, "id": item["id"]})

                if u.path == "/api/processar":  # compatível: põe uma música na fila
                    d = self._corpo_json()
                    app.caminho_seguro(app.musicas, d["musica"])
                    FILA.adicionar([d["musica"]], bool(d.get("auto", True)), d.get("modelo"), d.get("modo", "duplo"))
                    return self._json({"ok": True})

                if u.path == "/api/fila/adicionar":
                    d = self._corpo_json()
                    nomes = d.get("musicas")
                    if nomes == "todas":  # todas as que ainda não foram processadas
                        nomes = [m["nome"] for m in app.listar_musicas() if not m["processada"]]
                    for n in nomes:
                        app.caminho_seguro(app.musicas, n)
                    n = FILA.adicionar(nomes, bool(d.get("auto", True)), d.get("modelo"), d.get("modo", "duplo"))
                    return self._json({"ok": True, "adicionadas": n})

                if u.path == "/api/fila/remover":
                    FILA.remover(self._corpo_json()["musica"])
                    return self._json({"ok": True})

                if u.path == "/api/fila/limpar":
                    FILA.limpar_concluidas()
                    return self._json({"ok": True})

                if u.path == "/api/salvar":
                    d = self._corpo_json()
                    proj = app._projeto(d["projeto"])
                    try:
                        destino = _salvar_mix(proj, d)
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc()
                        return self._json({"erro": f"{type(e).__name__}: {e}"}, 500)
                    return self._json({"ok": True, "arquivo": destino.name})
                self._json({"erro": "não encontrado"}, 404)
            except (FileNotFoundError, KeyError) as e:
                self._json({"erro": f"não encontrado: {e}"}, 404)

    return H


# --------------------------------------------------------------------------- links do YouTube
DOWNLOADS: list[dict[str, Any]] = []
_LOCK_DL = threading.Lock()


def _baixar_link(app: "App", item: dict[str, Any]) -> None:
    from .youtube import baixar

    def prog(f: float) -> None:
        item["pct"] = round(f * 100)

    try:
        item["status"] = "baixando"
        destino = baixar(item["url"], app.musicas, prog)
        item.update(status="pronto", nome=destino.name, pct=100)
        if item.get("fila"):
            FILA.adicionar([destino.name], item.get("auto", True), None, item.get("modo", "duplo"))
            item["status"] = "na fila"
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        item.update(status="erro", erro=f"{type(e).__name__}: {e}"[:300])


def rodar(porta: int = 8765, musicas: str = "musicas", saida: str = "saida", abrir: bool = True) -> None:
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout)
    app = App(Path(musicas), Path(saida))
    FILA.configurar(app.musicas, app.saida)
    srv = ThreadingHTTPServer(("127.0.0.1", porta), _handler(app))
    url = f"http://localhost:{porta}"
    print(f"VS-Batera painel em {url}  (Ctrl+C para sair)")
    print(f"  músicas: {app.musicas}\n  saída:   {app.saida}")
    if abrir:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
