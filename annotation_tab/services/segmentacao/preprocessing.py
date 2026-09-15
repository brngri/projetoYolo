# -*- coding: utf-8 -*-
"""Pré-processamento espectral: blur, Savitzky-Golay, seleção de bandas."""

import logging
from typing import Optional, Tuple

import numpy as np
import cv2
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)


def apply_blur(cube: np.ndarray, kind: str = "gaussian",
               ksize: int = 3) -> np.ndarray:
    """Aplica blur espacial banda a banda."""
    h, w, b = cube.shape
    out = np.empty_like(cube)

    if kind == "median":
        k = max(3, ksize | 1)
        for i in range(b):
            out[:, :, i] = cv2.medianBlur(cube[:, :, i], k)
        return out

    k = max(1, ksize)
    if k % 2 == 0:
        k += 1
    k_tuple = (k, k)

    for i in range(b):
        if kind == "gaussian":
            out[:, :, i] = cv2.GaussianBlur(cube[:, :, i], k_tuple, 0)
        elif kind == "box":
            out[:, :, i] = cv2.blur(cube[:, :, i], k_tuple)
        else:
            raise ValueError(f"Tipo de blur desconhecido: {kind}")
    return out


def apply_savgol(cube: np.ndarray, window: int = 11,
                 polyorder: int = 2) -> np.ndarray:
    """Aplica filtro Savitzky-Golay no eixo espectral."""
    h, w, b = cube.shape

    if window % 2 == 0:
        window += 1
    if window > b:
        window = b if b % 2 == 1 else b - 1
    if window < 3:
        logger.warning("Janela SavGol muito pequena, ignorando filtro")
        return cube

    if polyorder >= window:
        polyorder = max(1, window - 2)
        logger.warning(f"polyorder ajustado para {polyorder} (deve ser < window)")

    data_2d = cube.reshape(-1, b)
    filtered = savgol_filter(data_2d, window, polyorder, axis=1, mode='nearest')
    return filtered.reshape(h, w, b)


def select_bands(
    cube: np.ndarray,
    signatures: np.ndarray,
    method: Optional[str],
    params: Optional[dict],
) -> Tuple[np.ndarray, Optional[object]]:
    """
    Reduz bandas via PCA ou SelectKBest.
    Retorna (cube_transformado, band_selector).
    """
    if method is None:
        return cube, None

    h, w, b = cube.shape
    X = cube.reshape(-1, b)
    params = params or {}

    if method == "pca":
        if params.get("pca_components"):
            pca = PCA(n_components=params["pca_components"])
        else:
            pca = PCA(n_components=params.get("pca_variance_retained", 0.99))
        X_trans = pca.fit_transform(X)
        logger.info(f"PCA: {b} -> {pca.n_components_} componentes")
        return X_trans.reshape(h, w, pca.n_components_), pca

    elif method == "kbest":
        k = min(params.get("kbest_k", 30), b)
        n_fake = min(1000, X.shape[0])
        idx_fake = np.random.choice(X.shape[0], n_fake, replace=False)
        X_train = np.vstack([signatures, X[idx_fake]])
        y_train = np.hstack([np.ones(len(signatures)), np.zeros(n_fake)])
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_train, y_train)
        X_sel = selector.transform(X)
        logger.info(f"SelectKBest: {b} -> {k} bandas")
        return X_sel.reshape(h, w, k), selector

    else:
        raise ValueError(f"Método de seleção desconhecido: {method}")