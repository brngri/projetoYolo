#!/usr/bin/env python3
"""
diagnostico_mascara.py (amostragem por classe)

Pré-análise para calibrar blur_circularity_mask do spectral_embedding_learned10.py.

Diferença em relação à versão anterior:
  - Em vez de sortear N arquivos aleatórios, agrupa por classe (extraída do
    nome do arquivo) e pega N_POR_CLASSE amostra(s) de CADA classe.
  - Imprime e salva um gráfico da distribuição de classes (mostra o
    desbalanceamento).
  - Cada análise é marcada com a classe no console e no relatório.

Como a classe é inferida:
  Padrão esperado: {CLASSE}_{YYYY-MM-DD}_{HH-MM-SS}_corrigido.npy
  Ex: Nut2_2026-09-01_20-52-56_corrigido.npy -> classe "Nut2"
      ClassA_2026-09-01_20-43-30_corrigido.npy -> classe "ClassA"

  Se o seu padrão for outro, edite a função extrair_classe().

Uso:
  python diagnostico_mascara.py
"""

import os
import re
import glob
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =============================================================================
# CONFIGURAÇÃO — EDITE AQUI
# =============================================================================
INPUT_FOLDER = r"E:\ProjetoCrainwalnut\preprocessed\coleta_20260901\FX17\FX17"
OUTPUT_FOLDER = r"E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX17\diagnostico"

N_POR_CLASSE = 2     # quantas amostras pegar de CADA classe
SEED = 42

# Parâmetros atuais que você usou (pra comparar com alternativas)
PARAMS_ATUAIS = dict(
    band_idx=None, threshold=None,
    blur_sigma=3.0, mask_threshold=0.5,
    opening_radius=5, min_area_ratio=0.0005,
    min_circularity=0.0, exclude_right_cols=130,
)
# =============================================================================


# -----------------------------------------------------------------------------
# Extração de classe a partir do nome do arquivo
# -----------------------------------------------------------------------------
def extrair_classe(nome_arquivo: str) -> str:
    """
    Retorna o nome da classe a partir do nome do arquivo.

    Padrão observado:
        Nut2_2026-09-01_20-52-56_corrigido.npy   -> "Nut2"
        ClassA_2026-09-01_20-43-30_corrigido.npy -> "ClassA"

    Estratégia:
      1) Procura um padrão de data YYYY-MM-DD; pega tudo antes dele.
      2) Se não achar, cai no fallback: parte antes do primeiro '_'.

    >>> EDITE AQUI se o seu padrão de nome for diferente.
    """
    stem = os.path.splitext(os.path.basename(nome_arquivo))[0]
    m = re.match(r"^(.+?)_\d{4}-\d{2}-\d{2}", stem)
    if m:
        return m.group(1)
    return stem.split("_")[0]


def carregar_cubo(fpath):
    """Carrega .npy e padroniza eixos para (H, W, B)."""
    cube = np.load(fpath)
    if cube.ndim == 2 and cube.shape[1] == 224:
        total = cube.shape[0]
        h = int(math.sqrt(total))
        while total % h != 0:
            h -= 1
        w = total // h
        cube = cube.reshape(h, w, 224)
    elif cube.ndim == 3 and cube.shape[0] == 224:
        cube = np.transpose(cube, (1, 2, 0))
    if cube.ndim != 3:
        raise ValueError(f"formato inesperado: {cube.shape}")
    return cube

# -----------------------------------------------------------------------------
# Listagem e amostragem por classe
# -----------------------------------------------------------------------------
def listar_cubos_por_classe(folder: str, n_por_classe: int, seed: int):
    """
    Agrupa os .npy por classe, imprime a distribuição (desbalanceamento) e
    retorna uma lista de (classe, filepath) com até n_por_classe por classe.
    """
    files = sorted(glob.glob(os.path.join(folder, "**", "*.npy"), recursive=True))
    files = [f for f in files if "renomeados" not in f]
    if not files:
        raise FileNotFoundError(f"Nenhum .npy em {folder}")

    classes = defaultdict(list)
    for f in files:
        classes[extrair_classe(f)].append(f)

    n_classes = len(classes)
    tamanhos = [len(v) for v in classes.values()]
    total = sum(tamanhos)

    print("=" * 65)
    print(f"[info] {n_classes} classes encontradas, {total} arquivos no total")
    print(f"[info] menor classe: {min(tamanhos)} amostra(s)")
    print(f"[info] maior classe: {max(tamanhos)} amostra(s)")
    print(f"[info] média por classe: {total / n_classes:.1f}")
    print(f"[info] razão desbalanceamento (maior/menor): {max(tamanhos) / max(min(tamanhos), 1):.1f}x")
    print("=" * 65)

    # Top 10 mais frequentes
    ordenadas = sorted(classes.items(), key=lambda kv: -len(kv[1]))
    print("[info] 10 classes mais frequentes:")
    for c, lista in ordenadas[:10]:
        print(f"       {c:<25s} {len(lista):>4d} amostra(s)")
    if n_classes > 10:
        print(f"       ... + {n_classes - 10} outras classes")

    # Top 5 menos frequentes (útil pra ver as classes raras)
    if n_classes > 5:
        print("[info] 5 classes menos frequentes:")
        for c, lista in ordenadas[-5:]:
            print(f"       {c:<25s} {len(lista):>4d} amostra(s)")
    print("=" * 65)

    # Amostragem: até n_por_classe por classe
    rng = np.random.default_rng(seed)
    selecionados = []
    for classe in sorted(classes.keys()):
        lista = classes[classe]
        n = min(n_por_classe, len(lista))
        idx = rng.choice(len(lista), size=n, replace=False)
        for i in idx:
            selecionados.append((classe, lista[i]))

    print(f"[info] {len(selecionados)} cubos selecionados para análise")
    return selecionados, classes


# -----------------------------------------------------------------------------
# Gráfico da distribuição de classes
# -----------------------------------------------------------------------------
def plotar_distribuicao_classes(classes: dict, out_dir: Path):
    """
    Salva um gráfico de barras com a contagem de amostras por classe.
    Ajuda a visualizar o desbalanceamento de cara.
    """
    nomes = list(classes.keys())
    contagens = [len(classes[c]) for c in nomes]

    # ordena por contagem (decrescente) pra ficar mais legível
    ordem = np.argsort(contagens)[::-1]
    nomes_ord = [nomes[i] for i in ordem]
    contagens_ord = [contagens[i] for i in ordem]

    fig, ax = plt.subplots(figsize=(max(10, len(nomes) * 0.35), 6))
    bars = ax.bar(range(len(nomes_ord)), contagens_ord, color='steelblue')

    # destaca visualmente classes com poucas amostras
    limite_baixo = np.percentile(contagens_ord, 25)
    for i, (b, c) in enumerate(zip(bars, contagens_ord)):
        if c <= limite_baixo:
            b.set_color('crimson')

    ax.set_xticks(range(len(nomes_ord)))
    ax.set_xticklabels(nomes_ord, rotation=90, fontsize=6)
    ax.set_ylabel("nº de amostras")
    ax.set_title(f"Distribuição de amostras por classe ({len(nomes_ord)} classes)")
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "_distribuicao_classes.png"
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"[info] gráfico de distribuição salvo em: {out_path}")


# -----------------------------------------------------------------------------
# 1. Análise da imagem de referência (média e bandas individuais)
# -----------------------------------------------------------------------------
def analisar_bandas(cube, out_dir, nome):
    H, W, B = cube.shape
    bandas_teste = [0, B // 4, B // 2, 3 * B // 4, B - 1]
    refs = {"mean": cube.mean(axis=-1)}
    for b in bandas_teste:
        refs[f"band_{b}"] = cube[..., b]

    fig, axes = plt.subplots(2, len(refs), figsize=(4 * len(refs), 7))
    fig.suptitle(f"{nome} — histogramas e imagens de referência", fontsize=12)

    for i, (label, ref) in enumerate(refs.items()):
        lo, hi = np.percentile(ref, [1, 99])
        img_norm = np.clip((ref - lo) / (hi - lo + 1e-8), 0, 1)
        axes[0, i].imshow(img_norm, cmap='gray')
        axes[0, i].set_title(label, fontsize=9)
        axes[0, i].axis('off')

        axes[1, i].hist(ref.ravel(), bins=80, color='steelblue')
        axes[1, i].set_title("histograma", fontsize=9)
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{nome}_histogramas.png", dpi=100)
    plt.close()


# -----------------------------------------------------------------------------
# 2. Teste de métodos de threshold
# -----------------------------------------------------------------------------
def testar_thresholds(cube, out_dir, nome):
    from skimage.filters import threshold_otsu, threshold_local
    from skimage.morphology import white_tophat, disk

    ref_mean = cube.mean(axis=-1)
    resultados = {}

    try:
        t = threshold_otsu(ref_mean)
        resultados["otsu_global_mean"] = (ref_mean > t, t)
    except Exception:
        resultados["otsu_global_mean"] = (np.zeros_like(ref_mean, bool), None)

    b_mid = cube.shape[-1] // 2
    ref_band = cube[..., b_mid]
    try:
        t = threshold_otsu(ref_band)
        resultados["otsu_band_mid"] = (ref_band > t, t)
    except Exception:
        resultados["otsu_band_mid"] = (np.zeros_like(ref_band, bool), None)

    ref_corr = white_tophat(ref_mean, disk(50))
    try:
        t = threshold_otsu(ref_corr)
        resultados["otsu_tophat"] = (ref_corr > t, t)
    except Exception:
        resultados["otsu_tophat"] = (np.zeros_like(ref_corr, bool), None)

    try:
        local_thresh = threshold_local(ref_mean, block_size=51, offset=0)
        resultados["otsu_local"] = (ref_mean > local_thresh, "local")
    except Exception:
        resultados["otsu_local"] = (np.zeros_like(ref_mean, bool), None)

    p = np.percentile(ref_mean, 70)
    resultados["percentil_70"] = (ref_mean > p, p)

    n = len(resultados)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    fig.suptitle(f"{nome} — comparação de métodos de threshold", fontsize=12)

    lo, hi = np.percentile(ref_mean, [1, 99])
    img_norm = np.clip((ref_mean - lo) / (hi - lo + 1e-8), 0, 1)

    for i, (label, (mask, t)) in enumerate(resultados.items()):
        overlay = np.stack([img_norm] * 3, axis=-1)
        overlay[mask, 0] = 0.6 * overlay[mask, 0] + 0.4
        overlay[mask, 1] *= 0.6
        overlay[mask, 2] *= 0.6
        axes[0, i].imshow(overlay)
        axes[0, i].set_title(label, fontsize=9)
        axes[0, i].axis('off')

        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].axis('off')
        cobertura = mask.mean() * 100
        axes[1, i].set_xlabel(f"{cobertura:.1f}% dos pixels", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / f"{nome}_thresholds.png", dpi=100)
    plt.close()
    return resultados


# -----------------------------------------------------------------------------
# 3. Teste de parâmetros pós-threshold
# -----------------------------------------------------------------------------
def testar_pos_processamento(cube, out_dir, nome, threshold_base=None):
    from skimage import morphology, measure
    from scipy.ndimage import gaussian_filter

    ref = cube.mean(axis=-1)
    if threshold_base is None:
        from skimage.filters import threshold_otsu
        threshold_base = threshold_otsu(ref)

    mask1 = ref > threshold_base

    variacoes = [
        ("baseline",  dict(blur_sigma=3.0, opening_radius=5,  min_area_ratio=0.0005, min_circularity=0.0)),
        ("blur=1.5",  dict(blur_sigma=1.5, opening_radius=5,  min_area_ratio=0.0005, min_circularity=0.0)),
        ("blur=5.0",  dict(blur_sigma=5.0, opening_radius=5,  min_area_ratio=0.0005, min_circularity=0.0)),
        ("opening=2", dict(blur_sigma=3.0, opening_radius=2,  min_area_ratio=0.0005, min_circularity=0.0)),
        ("opening=10",dict(blur_sigma=3.0, opening_radius=10, min_area_ratio=0.0005, min_circularity=0.0)),
        ("area=0.002",dict(blur_sigma=3.0, opening_radius=5,  min_area_ratio=0.002,  min_circularity=0.0)),
        ("circ=0.4",  dict(blur_sigma=3.0, opening_radius=5,  min_area_ratio=0.0005, min_circularity=0.4)),
        ("circ=0.6",  dict(blur_sigma=3.0, opening_radius=5,  min_area_ratio=0.0005, min_circularity=0.6)),
    ]

    fig, axes = plt.subplots(2, len(variacoes), figsize=(2.8 * len(variacoes), 6))
    fig.suptitle(f"{nome} — efeito dos parâmetros pós-threshold", fontsize=12)

    stats = []
    for i, (label, params) in enumerate(variacoes):
        mask_blurred = gaussian_filter(mask1.astype(np.float32), sigma=params["blur_sigma"])
        m = mask_blurred > 0.5
        m = morphology.opening(m, morphology.disk(params["opening_radius"]))

        labeled = measure.label(m)
        props = measure.regionprops(labeled)
        total = m.shape[0] * m.shape[1]
        min_area = params["min_area_ratio"] * total

        final = np.zeros_like(m, bool)
        for p in props:
            if p.area < min_area:
                continue
            circ = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
            if circ >= params["min_circularity"]:
                final[labeled == p.label] = True

        lo, hi = np.percentile(ref, [1, 99])
        img_norm = np.clip((ref - lo) / (hi - lo + 1e-8), 0, 1)
        axes[0, i].imshow(img_norm, cmap='gray')
        axes[0, i].set_title(label, fontsize=9)
        axes[0, i].axis('off')

        axes[1, i].imshow(final, cmap='gray')
        axes[1, i].axis('off')
        n_comp = len(measure.regionprops(measure.label(final)))
        axes[1, i].set_xlabel(f"{n_comp} componentes", fontsize=8)

        stats.append((label, n_comp, final.mean() * 100))

    plt.tight_layout()
    plt.savefig(out_dir / f"{nome}_pos_processamento.png", dpi=100)
    plt.close()
    return stats


# -----------------------------------------------------------------------------
# 4. Relatório numérico de componentes
# -----------------------------------------------------------------------------
def relatorio_componentes(cube, out_dir, nome, threshold_base=None):
    from skimage import morphology, measure
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu

    ref = cube.mean(axis=-1)
    if threshold_base is None:
        threshold_base = threshold_otsu(ref)

    mask1 = ref > threshold_base
    mask_blurred = gaussian_filter(mask1.astype(np.float32), sigma=3.0)
    m = mask_blurred > 0.5
    m = morphology.opening(m, morphology.disk(5))

    labeled = measure.label(m)
    props = measure.regionprops(labeled)
    total = m.shape[0] * m.shape[1]

    linhas = [f"# {nome}", f"threshold Otsu = {threshold_base:.4f}", f"total pixels = {total}", ""]
    linhas.append(f"{'area':>10} {'area_ratio':>12} {'circularidade':>15} {'bbox':>20} {'passa_area(0.0005)':>20} {'passa_circ(0.4)':>18}")

    for p in sorted(props, key=lambda p: -p.area):
        circ = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
        area_ratio = p.area / total
        bbox = f"{p.bbox[0]},{p.bbox[1]},{p.bbox[2]},{p.bbox[3]}"
        passa_area = "SIM" if area_ratio >= 0.0005 else "nao"
        passa_circ = "SIM" if circ >= 0.4 else "nao"
        linhas.append(f"{p.area:>10} {area_ratio:>12.6f} {circ:>15.4f} {bbox:>20} {passa_area:>20} {passa_circ:>18}")

    texto = "\n".join(linhas)
    (out_dir / f"{nome}_componentes.txt").write_text(texto, encoding="utf-8")
    return props


# -----------------------------------------------------------------------------
# 5. Detecção automática de problemas
# -----------------------------------------------------------------------------
def detectar_problemas(props, cube, params):
    H, W = cube.shape[:2]
    alertas = []
    total = H * W

    areas = [p.area for p in props]
    if not areas:
        alertas.append("NENHUM componente sobreviveu — threshold muito alto ou tudo foi filtrado")
        return alertas

    n_grandes = sum(1 for a in areas if a / total >= 0.005)
    n_pequenos = sum(1 for a in areas if a / total < 0.0005)

    if n_grandes == 0:
        alertas.append("Nenhum componente com >= 0.5% da imagem — pode estar faltando objetos")
    if n_pequenos > n_grandes * 3:
        alertas.append(f"Muitos componentes pequenos ({n_pequenos}) — possivelmente ruido/reflexos")

    for p in props:
        if p.area / total < 0.001:
            continue
        minr, minc, maxr, maxc = p.bbox
        toca_esq = minc <= 2
        toca_dir = maxc >= W - 2
        toca_top = minr <= 2
        toca_bot = maxr >= H - 2
        if toca_dir and params.get("exclude_right_cols", 0) > 0:
            alertas.append("Componente grande toca borda DIREITA — verifique EXCLUDE_RIGHT_COLS")
            break
        if toca_esq or toca_top or toca_bot:
            alertas.append(f"Componente grande toca borda ({p.bbox}) — pode estar cortado")
            break

    for p in props:
        if p.area / total < 0.001:
            continue
        h = p.bbox[2] - p.bbox[0]
        w = p.bbox[3] - p.bbox[1]
        aspect = max(h, w) / (min(h, w) + 1e-8)
        if aspect > 5:
            alertas.append(f"Componente alongado (aspect={aspect:.1f}) — provavelmente bandeja/listra")
            break

    if not alertas:
        alertas.append("Sem problemas obvios detectados")

    return alertas


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    out_dir = Path(OUTPUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    cubos, classes_dict = listar_cubos_por_classe(INPUT_FOLDER, N_POR_CLASSE, SEED)
    plotar_distribuicao_classes(classes_dict, out_dir)

    print(f"[info] resultados em {out_dir}\n")

    relatorio_geral = []
    # resumo por classe: guarda os alertas de cada amostra
    resumo_por_classe = defaultdict(list)

    for i, (classe, fpath) in enumerate(cubos, 1):
        nome = os.path.splitext(os.path.basename(fpath))[0]
        print(f"[{i}/{len(cubos)}] [classe: {classe}] {nome}")
        try:
            cube = carregar_cubo(fpath)
        except Exception as e:
            print(f"  [erro] {e}")
            continue

        print(f"  shape: {cube.shape}")

        analisar_bandas(cube, out_dir, nome)
        testar_thresholds(cube, out_dir, nome)
        testar_pos_processamento(cube, out_dir, nome)

        from skimage.filters import threshold_otsu
        ref = cube.mean(axis=-1)
        t = threshold_otsu(ref)
        props = relatorio_componentes(cube, out_dir, nome, t)
        alertas = detectar_problemas(props, cube, PARAMS_ATUAIS)

        bloco = [f"\n{'=' * 60}", f"CLASSE: {classe}", f"ARQUIVO: {nome}", f"{'=' * 60}"]
        bloco.append(f"componentes detectados: {len(props)}")
        bloco.append("problemas:")
        for a in alertas:
            bloco.append(f"  - {a}")
        relatorio_geral.extend(bloco)

        resumo_por_classe[classe].append((nome, len(props), alertas))

        print(f"  componentes: {len(props)}")
        for a in alertas:
            print(f"  ! {a}")

    # ---- Resumo consolidado por classe ----
    resumo_classes = ["", "=" * 60, "RESUMO POR CLASSE", "=" * 60]
    for classe in sorted(resumo_por_classe.keys()):
        amostras = resumo_por_classe[classe]
        resumo_classes.append(f"\n{classe}: {len(amostras)} amostra(s) analisada(s)")
        for nome, n_comp, alertas in amostras:
            resumo_classes.append(f"  - {nome}: {n_comp} componentes")
            for a in alertas:
                resumo_classes.append(f"      ! {a}")

    # Escreve tudo num único arquivo
    (out_dir / "resumo_geral.txt").write_text(
        "\n".join(relatorio_geral + resumo_classes), encoding="utf-8"
    )

    print(f"\n[OK] Diagnóstico completo salvo em {out_dir}")
    print("Arquivos por cubo:")
    print("  - *_histogramas.png       -> compara média vs bandas individuais")
    print("  - *_thresholds.png        -> compara 5 métodos de threshold")
    print("  - *_pos_processamento.png -> efeito de cada parâmetro")
    print("  - *_componentes.txt       -> lista de componentes com área/circ")
    print("Arquivos globais:")
    print("  - _distribuicao_classes.png -> gráfico do desbalanceamento")
    print("  - resumo_geral.txt          -> relatório por amostra + por classe")


if __name__ == "__main__":
    main()