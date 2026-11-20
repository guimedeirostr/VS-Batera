"""Testes com áudio sintético (não precisam baixar modelos de IA)."""
import numpy as np
import pytest

from vsbatera import config as cfgmod
from vsbatera import pipeline as pl
from vsbatera.audio_io import carregar, salvar
from vsbatera.limpeza import remover_vazamento

SR = 44100
DUR = 6.0


def _bateria(n):
    x = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    for t in range(0, n, SR // 2):  # um ataque a cada 0,5 s
        k = min(4000, n - t)
        x[t : t + k] += rng.standard_normal(k).astype(np.float32) * np.exp(-np.arange(k) / 600)
    return np.stack([x, x]) * 0.5


def _tom(freq, n, amp=0.3):
    t = np.arange(n) / SR
    s = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([s, s])


@pytest.fixture
def projeto(tmp_path):
    n = int(SR * DUR)
    bat = _bateria(n)
    rng = np.random.default_rng(1)
    stems = {
        "bateria": bat,
        "baixo": _tom(55, n) + 0.25 * bat,
        "voz": _tom(440, n, 0.2) + 0.2 * bat + 0.01 * rng.standard_normal((2, n)).astype(np.float32),
        "outros": _tom(880, n, 0.1) + 0.3 * bat,
    }
    proj = tmp_path / "musica"
    for nome, a in stems.items():
        salvar(proj / "stems" / f"{nome}.wav", a, SR)
    mix = sum(stems.values())
    salvar(proj / "sem_bateria_residual.wav", mix - bat, SR)
    return proj, stems


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def test_remover_vazamento_reduz_bateria():
    n = int(SR * DUR)
    bat = _bateria(n)
    limpo = _tom(55, n)
    sujo = limpo + 0.3 * bat
    saida = remover_vazamento(sujo, bat, SR, forca=0.9)
    erro_antes = np.mean((sujo - limpo) ** 2)
    erro_depois = np.mean((saida[:, :n] - limpo[:, : saida.shape[1]]) ** 2)
    assert erro_depois < 0.5 * erro_antes


def test_fluxo_completo(projeto):
    proj, stems = projeto
    preset = cfgmod.carregar_preset(None)

    analise = pl.etapa_analisar(proj, preset)
    assert set(analise) == set(stems)
    assert analise["baixo"]["vazamento_bateria_pct"] > 0
    assert (proj / "preset_sugerido.yaml").exists()

    sugerido = cfgmod.carregar_preset(proj / "preset_sugerido.yaml")
    limpas = pl.etapa_limpar(proj, sugerido)
    assert set(limpas) == set(stems)

    final = pl.etapa_mixar(proj, sugerido)
    audio, sr = carregar(final)
    assert sr == SR and audio.shape[0] == 2
    assert np.max(np.abs(audio)) <= 10 ** (-1 / 20) + 1e-4
    n = min(audio.shape[1], stems["bateria"].shape[1])
    # a versão final deve ter muito menos bateria do que a soma crua das faixas
    crua = stems["baixo"] + stems["voz"] + stems["outros"]
    assert abs(_corr(audio[:, :n], stems["bateria"][:, :n])) < abs(_corr(crua[:, :n], stems["bateria"][:, :n]))


def test_previa_velocidade_e_guia(projeto):
    proj, _ = projeto
    preset = cfgmod.carregar_preset(None)
    preset["mix"].update(velocidade=0.8, bateria_guia_db=-20)
    pl.etapa_limpar(proj, preset, inicio=1, duracao=2)
    final = pl.etapa_mixar(proj, preset, inicio=1, duracao=2)
    assert "PREVIA" in final.name and "80%" in final.name
    audio, sr = carregar(final)
    assert abs(audio.shape[1] / sr - 2 / 0.8) < 0.1  # 2 s a 80% ≈ 2,5 s


def test_fonte_residual(projeto):
    proj, _ = projeto
    preset = cfgmod.carregar_preset(None)
    preset["mix"]["fonte"] = "residual"
    audio, _ = carregar(pl.etapa_mixar(proj, preset))
    assert audio.shape[1] > 0


def test_separar_demucs_com_modelo_falso(tmp_path, monkeypatch):
    """Exercita o caminho do Demucs com um modelo de pesos aleatórios (sem download)."""
    import demucs.pretrained
    from demucs.htdemucs import HTDemucs

    fontes = ["drums", "bass", "other", "vocals"]
    monkeypatch.setattr(
        demucs.pretrained, "get_model", lambda nome: HTDemucs(sources=fontes, channels=8, t_layers=1, bottom_channels=64)
    )
    musica = tmp_path / "x.wav"
    salvar(musica, _tom(220, SR * 3) + _bateria(SR * 3), SR)
    from vsbatera.separacao import separar

    caminhos = separar(musica, tmp_path / "proj", modelo="falso", dispositivo="cpu", shifts=0)
    assert {"bateria", "baixo", "outros", "voz", "sem_bateria_residual"} <= set(caminhos)
