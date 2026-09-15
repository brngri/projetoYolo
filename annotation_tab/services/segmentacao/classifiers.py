# -*- coding: utf-8 -*-
"""Classificadores: KMeans, KMeans Guiado, Random Forest."""

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


# =============================================================================
# KMeans (não supervisionado)
# =============================================================================
def classify_kmeans(
    cube: np.ndarray,
    n_clusters: int = 2,
    kmeans_params: Optional[dict] = None,
) -> np.ndarray:
    """
    KMeans puro (não supervisionado) com PCA opcional.

    - Se `kmeans_params['pca_components']` for um inteiro N > 0 e < n_bandas,
      aplica PCA reduzindo o cubo para N componentes ANTES do KMeans.
      Isso reduz drasticamente o uso de memória (224 -> N).
    - O cluster escolhido como "folha" é aquele cujo centroide está MAIS
      DISTANTE da média global (heurística: folha = cluster mais extremo).
    """
    h, w, b = cube.shape
    X = cube.reshape(-1, b).astype(np.float32)

    params = dict(kmeans_params or {})
    n_clusters = params.pop("n_clusters", n_clusters)
    pca_n = params.pop("pca_components", None)   # <-- removido antes do KMeans

    # ---------- PCA opcional ----------
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
# KMeans Guiado (semi-supervisionado)
# =============================================================================
def classify_kmeans_guiado(
    cube: np.ndarray,
    signatures: np.ndarray,
    kmeans_params: Optional[dict] = None,
) -> np.ndarray:
    """
    KMeans guiado: 1 centroide = média das assinaturas de folha;
    demais centroides = pixels aleatórios.

    Correções aplicadas:
    - `n_init` é removido do dict antes de passar para o KMeans
      (evita "multiple values for keyword argument 'n_init'").
    - Se `pca_components` estiver definido, aplica PCA no cubo E nas
      assinaturas (mesmo espaço), para que os centroides façam sentido.
    """
    h, w, b = cube.shape
    X = cube.reshape(-1, b).astype(np.float32)
    sig = signatures.astype(np.float32)

    params = dict(kmeans_params or {})
    n_clusters = params.pop("n_clusters", 2)
    pca_n = params.pop("pca_components", None)   # <-- removido antes do KMeans
    params.pop("n_init", None)                   # <-- CORREÇÃO: evita conflito
    random_state = params.get("random_state", 42)

    # ---------- PCA opcional (no cubo e nas assinaturas) ----------
    if pca_n is not None and 0 < pca_n < b:
        pca = PCA(n_components=pca_n, random_state=random_state)
        X = pca.fit_transform(X)
        sig = pca.transform(sig)
        logger.info(f"PCA no KMeans guiado: {b} -> {pca_n} componentes")
    else:
        logger.info(f"KMeans guiado: usando todas as {b} bandas (PCA desativado)")

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
        n_init=1,                 # determinístico (init já fornecido)
        **params,                 # <-- agora sem 'n_init' nem 'pca_components'
    )
    labels = km.fit_predict(X).reshape(h, w)

    dists = np.linalg.norm(km.cluster_centers_ - mean_sig, axis=1)
    target_cluster = int(np.argmin(dists))
    logger.info(f"KMeans guiado: cluster alvo = {target_cluster} (de {n_clusters})")

    return (labels == target_cluster).astype(np.uint8)


# =============================================================================
# Preparação de dados de treino (RF)
# =============================================================================
def prepare_training_data(
    cube: np.ndarray,
    signatures_target: np.ndarray,
    signatures_background: Optional[np.ndarray] = None,
    n_background: int = 5000,
    band_selector=None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monta X_train / y_train para o RF.

    Classe 1 (folha): assinaturas fornecidas.
    Classe 0 (fundo):
        - se signatures_background fornecido -> usa (preferencial)
        - senão -> amostragem aleatória do cubo (fallback, menos confiável)
    """
    rng = np.random.RandomState(random_state)

    # Classe 1: folha
    X_target = signatures_target.copy()
    if band_selector is not None:
        X_target = band_selector.transform(X_target)
    y_target = np.ones(len(X_target), dtype=int)

    # Classe 0: fundo
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
            f"Usando {n_bg} pixels aleatórios (pode conter folha)."
        )

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
    """Random Forest com threshold de probabilidade."""
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