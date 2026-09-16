# -*- coding: utf-8 -*-
"""Classificadores: KMeans, KMeans Guiado, Random Forest, Otsu."""

import logging
from typing import Optional, Tuple

import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


# =============================================================================
# Util
# =============================================================================
def _nm_to_index(nm: float, n_bands: int,
                 start_nm: float = 400.0, end_nm: float = 1000.0) -> int:
    """Converte comprimento de onda (nm) em índice de banda, assumindo
    espaçamento linear entre start_nm e end_nm."""
    if n_bands <= 1:
        return 0
    step = (end_nm - start_nm) / (n_bands - 1)
    if abs(step) < 1e-12:
        return 0
    idx = int(round((nm - start_nm) / step))
    return max(0, min(idx, n_bands - 1))


# =============================================================================
# KMeans (não supervisionado) — com PCA opcional
# =============================================================================
def classify_kmeans(
    cube: np.ndarray,
    n_clusters: int = 2,
    kmeans_params: Optional[dict] = None,
) -> np.ndarray:
    h, w, b = cube.shape
    X = cube.reshape(-1, b).astype(np.float32)

    params = dict(kmeans_params or {})
    n_clusters = params.pop("n_clusters", n_clusters)
    pca_n = params.pop("pca_components", None)

    if pca_n is not None and 0 < pca_n < b:
        pca = PCA(n_components=pca_n, random_state=params.get("random_state", 42))
        X = pca.fit_transform(X)
        logger.info(f"PCA no KMeans: {b} -> {pca_n} componentes")
    else:
        logger.info(f"KMeans: usando todas as {b} bandas (PCA desativado)")

    km = KMeans(n_clusters=n_clusters, **params)
    labels = km.fit_predict(X).reshape(h, w)

    global_mean = X.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(km.cluster_centers_ - global_mean, axis=1)
    target_cluster = int(np.argmax(dists))
    logger.info(f"KMeans: cluster alvo = {target_cluster} (de {n_clusters})")

    return (labels == target_cluster).astype(np.uint8)


# =============================================================================
# KMeans Guiado
# =============================================================================
def classify_kmeans_guiado(
    cube: np.ndarray,
    signatures: np.ndarray,
    kmeans_params: Optional[dict] = None,
) -> np.ndarray:
    h, w, b = cube.shape
    X = cube.reshape(-1, b).astype(np.float32)
    sig = signatures.astype(np.float32)

    params = dict(kmeans_params or {})
    n_clusters = params.pop("n_clusters", 2)
    pca_n = params.pop("pca_components", None)
    params.pop("n_init", None)  # evita conflito
    random_state = params.get("random_state", 42)

    if pca_n is not None and 0 < pca_n < b:
        pca = PCA(n_components=pca_n, random_state=random_state)
        X = pca.fit_transform(X)
        sig = pca.transform(sig)
        logger.info(f"PCA no KMeans guiado: {b} -> {pca_n} componentes")

    mean_sig = np.mean(sig, axis=0)

    if n_clusters > 1:
        rng = np.random.RandomState(random_state)
        n_extra = n_clusters - 1
        idx_extra = rng.choice(X.shape[0], n_extra, replace=False)
        init_centers = np.vstack([mean_sig, X[idx_extra]])
    else:
        init_centers = mean_sig[None, :]

    km = KMeans(
        n_clusters=n_clusters,
        init=init_centers,
        n_init=1,
        **params,
    )
    labels = km.fit_predict(X).reshape(h, w)

    dists = np.linalg.norm(km.cluster_centers_ - mean_sig, axis=1)
    target_cluster = int(np.argmin(dists))
    logger.info(f"KMeans guiado: cluster alvo = {target_cluster} (de {n_clusters})")

    return (labels == target_cluster).astype(np.uint8)


# =============================================================================
# Preparação de treino (RF)
# =============================================================================
def prepare_training_data(
    cube: np.ndarray,
    signatures_target: np.ndarray,
    signatures_background: Optional[np.ndarray] = None,
    n_background: int = 5000,
    band_selector=None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)

    X_target = signatures_target.copy()
    if band_selector is not None:
        X_target = band_selector.transform(X_target)
    y_target = np.ones(len(X_target), dtype=int)

    if signatures_background is not None and len(signatures_background) > 0:
        X_bg = signatures_background.copy()
        if band_selector is not None:
            X_bg = band_selector.transform(X_bg)
        if len(X_bg) > n_background:
            idx = rng.choice(len(X_bg), n_background, replace=False)
            X_bg = X_bg[idx]
        logger.info(f"RF: fundo = {len(X_bg)} assinaturas de background")
    else:
        h, w, b = cube.shape
        X_all = cube.reshape(-1, b)
        n_bg = min(n_background, X_all.shape[0])
        idx_bg = rng.choice(X_all.shape[0], n_bg, replace=False)
        X_bg = X_all[idx_bg]
        if band_selector is not None:
            X_bg = band_selector.transform(X_bg)
        logger.warning(
            f"RF: nenhuma assinatura de background fornecida. "
            f"Usando {n_bg} pixels aleatórios (pode conter folha).")

    y_bg = np.zeros(len(X_bg), dtype=int)

    X_train = np.vstack([X_target, X_bg])
    y_train = np.hstack([y_target, y_bg])
    logger.info(f"Treino RF: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
    return X_train, y_train


# =============================================================================
# Random Forest
# =============================================================================
def classify_rf(
    cube: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    rf_params: dict,
    prob_threshold: float = 0.5,
    use_scaler: bool = True,
) -> np.ndarray:
    h, w, b = cube.shape
    X_all = cube.reshape(-1, b).astype(np.float32)

    if use_scaler:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_all_s = scaler.transform(X_all)
    else:
        X_train_s, X_all_s = X_train, X_all

    clf = RandomForestClassifier(**rf_params)
    clf.fit(X_train_s, y_train)
    probs = clf.predict_proba(X_all_s)

    prob_target = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
    mask = (prob_target >= prob_threshold).astype(np.uint8).reshape(h, w)
    logger.info(f"RF: {mask.sum()} pixels classificados como folha")
    return mask


# =============================================================================
# Otsu
# =============================================================================
def classify_otsu(
    cube: np.ndarray,
    otsu_params: Optional[dict] = None,
    cube_start_nm: float = 400.0,
    cube_end_nm: float = 1000.0,
) -> np.ndarray:
    """
    Segmentação via Otsu.

    - `use_ndvi=True`  -> calcula NDVI (nir-red)/(nir+red) e aplica Otsu.
    - `use_ndvi=False` -> usa a banda `band_nm` e aplica Otsu.
    - `invert=True`    -> inverte a máscara resultante.

    A conversão nm -> índice de banda é feita com `cube_start_nm`/`cube_end_nm`
    e o número de bandas do cubo.
    """
    h, w, b = cube.shape
    params = dict(otsu_params or {})
    use_ndvi = params.get("use_ndvi", False)
    invert = params.get("invert", False)

    if use_ndvi:
        red_nm = float(params.get("red_nm", 670.0))
        nir_nm = float(params.get("nir_nm", 800.0))
        red_idx = _nm_to_index(red_nm, b, cube_start_nm, cube_end_nm)
        nir_idx = _nm_to_index(nir_nm, b, cube_start_nm, cube_end_nm)
        red = cube[:, :, red_idx].astype(np.float32)
        nir = cube[:, :, nir_idx].astype(np.float32)
        denom = nir + red
        denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
        data = (nir - red) / denom
        logger.info(
            f"Otsu com NDVI (red={red_nm}nm idx={red_idx}, nir={nir_nm}nm idx={nir_idx})")
    else:
        band_nm = float(params.get("band_nm", 800.0))
        band_idx = _nm_to_index(band_nm, b, cube_start_nm, cube_end_nm)
        data = cube[:, :, band_idx].astype(np.float32)
        logger.info(f"Otsu com banda {band_nm}nm (idx={band_idx})")

    # Normaliza para 0-255 (uint8) para Otsu
    dmin, dmax = float(data.min()), float(data.max())
    if dmax - dmin < 1e-8:
        logger.warning("Banda/índice constante. Otsu retorna máscara vazia.")
        return np.zeros((h, w), dtype=np.uint8)

    data_norm = ((data - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
    thresh, binary = cv2.threshold(
        data_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    logger.info(f"Otsu threshold: {thresh:.1f} (normalizado 0-255)")

    mask = (binary > 0).astype(np.uint8)
    if invert:
        mask = 1 - mask
        logger.info("Máscara Otsu invertida.")

    logger.info(f"Otsu: {int(mask.sum())} pixels classificados como folha")
    return mask