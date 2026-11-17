#!/usr/bin/env python3
"""
checar_yolo.py — diagnóstico standalone

Não importa nada do spectral_embedding_learned10.py.
Copia as funções essenciais aqui pra não depender de import.
"""

import os
import glob
import math
import numpy as np

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
INPUT_FOLDER = r"E:\ProjetoCrainwalnut\preprocessed\coleta_20260901\FX17\FX17"

# Parâmetros dos grupos (idênticos ao script principal)
PARAMS_GRUPO_A = dict(
    band_idx=None, threshold=None,
    blur_sigma=3.0, mask_threshold=0.5,
    opening_radius=15, min_area_ratio=0.001, min_circularity=0.3,
    exclude_left_cols=20, exclude_right_cols=130,
)
PARAMS_GRUPO_B = dict(
    band_idx=None, threshold=None,
    blur_sigma=3.0, mask_threshold=0.5,
    opening_radius=5, min_area_ratio=0.005, min_circularity=0.5,
    exclude_left_cols=0, exclude_right_cols=0,
)
# ============================================================================


def padronizar_cubo(cube):
    """Reorganiza .npy para (H, W, B)."""
    if cube.ndim == 2 and cube.shape[1] == 224:
        total_pixels = cube.shape[0]
        h = int(math.sqrt(total_pixels))
        while total_pixels % h != 0:
            h -= 1
        w = total_pixels // h
        cube = cube.reshape(h, w, 224)
    elif cube.ndim == 3 and cube.shape[0] == 224:
        cube = np.transpose(cube, (1, 2, 0))
    return cube


def blur_circularity_mask(cube, band_idx=None, threshold=None,
                          blur_sigma=3.0, mask_threshold=0.5,
                          opening_radius=5, min_area_ratio=0.002,
                          min_circularity=0.5,
                          exclude_left_cols=0, exclude_right_cols=0,
                          return_intermediate=False, debug=False):
    """Máscara de foreground: Otsu -> blur -> opening -> área/circularidade."""
    from skimage import morphology, measure
    from skimage.filters import threshold_otsu
    from scipy.ndimage import gaussian_filter

    cube = padronizar_cubo(cube)

    if band_idx is not None:
        ref = cube[..., band_idx]
    else:
        ref = cube.mean(axis=-1)

    # Registra ANTES do corte pra comparar
    ref_original = ref.copy()

    if exclude_left_cols > 0 or exclude_right_cols > 0:
        ref = ref.copy()
        if exclude_left_cols > 0:
            ref[:, :exclude_left_cols] = ref.min()
        if exclude_right_cols > 0:
            ref[:, -exclude_right_cols:] = ref.min()

    if threshold is None:
        threshold = threshold_otsu(ref)

    mask1 = ref > threshold
    mask_blurred = gaussian_filter(mask1.astype(np.float32), sigma=blur_sigma)
    mask = mask_blurred > mask_threshold
    mask = morphology.opening(mask, morphology.disk(opening_radius))

    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    total_pixels = mask.shape[0] * mask.shape[1]
    min_area = min_area_ratio * total_pixels

    clean = np.zeros_like(mask, dtype=bool)
    for p in props:
        if p.area < min_area:
            continue
        circ = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
        if circ >= min_circularity:
            clean[labeled == p.label] = True

    if return_intermediate:
        return clean, mask1
    return clean


def analisar_imagem(fpath, grupo, params):
    """Roda a máscara e reporta o que aconteceu nas bordas."""
    nome = os.path.basename(fpath)
    try:
        cube = padronizar_cubo(np.load(fpath))
    except Exception as e:
        print(f"  [ERRO] {nome}: {e}")
        return None

    if cube.ndim != 3:
        print(f"  [SKIP] {nome}: shape {cube.shape}")
        return None

    H, W, B = cube.shape
    mask, mask_raw = blur_circularity_mask(cube, return_intermediate=True, **params)

    # Estatísticas
    n_componentes = 0
    if mask.any():
        from skimage import measure
        n_componentes = len(measure.regionprops(measure.label(mask)))

    ativos_dir = int(mask[:, -params.get("exclude_right_cols", 0):].sum()) if params.get("exclude_right_cols", 0) > 0 else 0
    ativos_esq = int(mask[:, :params.get("exclude_left_cols", 0)].sum()) if params.get("exclude_left_cols", 0) > 0 else 0
    cobertura = 100.0 * mask.sum() / mask.size

    print(f"  {nome}")
    print(f"    shape: {cube.shape}")
    print(f"    componentes: {n_componentes}")
    print(f"    cobertura: {cobertura:.1f}%")
    print(f"    pixels ativos à DIREITA (nas {params.get('exclude_right_cols', 0)} colunas): {ativos_dir}")
    print(f"    pixels ativos à ESQUERDA (nas {params.get('exclude_left_cols', 0)} colunas): {ativos_esq}")
    print(f"    bbox do maior componente: ", end="")
    if mask.any():
        from skimage import measure
        props = measure.regionprops(measure.label(mask))
        maior = max(props, key=lambda p: p.area)
        print(f"{maior.bbox}  area={maior.area}")
    else:
        print("(vazio)")
    return mask


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print(f"[info] INPUT_FOLDER: {INPUT_FOLDER}\n")

    # --- Testa 2 amostras do Grupo B ---
    print("=" * 70)
    print("GRUPO B — amostras de Nut (esperado: 1 componente, sem corte)")
    print("=" * 70)
    for classe_b in ["Nut1", "Nut2", "Nut12"]:
        pasta = os.path.join(INPUT_FOLDER, classe_b)
        if not os.path.isdir(pasta):
            print(f"[skip] pasta não existe: {pasta}")
            continue
        arquivos = sorted(glob.glob(os.path.join(pasta, "*.npy")))[:1]
        for f in arquivos:
            analisar_imagem(f, "B", PARAMS_GRUPO_B)

    # --- Testa 2 amostras do Grupo A ---
    print("\n" + "=" * 70)
    print("GRUPO A — amostras de bandeja (esperado: vários componentes, "
          "zero pixels ativos à direita)")
    print("=" * 70)
    for classe_a in ["AdHul", "Dirty", "ClassA"]:
        pasta = os.path.join(INPUT_FOLDER, classe_a)
        if not os.path.isdir(pasta):
            print(f"[skip] pasta não existe: {pasta}")
            continue
        arquivos = sorted(glob.glob(os.path.join(pasta, "*.npy")))[:1]
        for f in arquivos:
            analisar_imagem(f, "A", PARAMS_GRUPO_A)

    # --- Testa Shrivel (caso ambíguo) ---
    print("\n" + "=" * 70)
    print("SHRIVEL — teste nos dois grupos pra ver qual faz sentido")
    print("=" * 70)
    pasta_shrivel = os.path.join(INPUT_FOLDER, "Shrivel")
    if os.path.isdir(pasta_shrivel):
        arquivos = sorted(glob.glob(os.path.join(pasta_shrivel, "*.npy")))[:1]
        for f in arquivos:
            print("\n--- Shrivel como GRUPO A ---")
            analisar_imagem(f, "A", PARAMS_GRUPO_A)
            print("\n--- Shrivel como GRUPO B ---")
            analisar_imagem(f, "B", PARAMS_GRUPO_B)