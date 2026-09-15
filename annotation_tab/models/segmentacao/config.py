# -*- coding: utf-8 -*-
"""Configurações padrão e carregamento."""

import os
import logging
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    # Diretórios
    "root_dir": r"D:\dadosBruno_2\projetoDidion\corrRad\segmentacaoRF\teste",
    "out_root": r"D:\dadosBruno_2\projetoDidion\corrRad\segmentacaoRF\kmeans_guiado",
    "json_path": [
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\grao\Corn_29-28_2026-08-26_13-54-52_corrigido_polygons_20260910_095041.json",
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\grao\Corn_31-30_2026-08-25_18-15-47_corrigido_polygons_20260910_093921.json",
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\grao\Corn_75-74_2026-08-25_16-31-42_corrigido_polygons_20260910_094358.json",
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\grao\Corn_33-32_2026-08-25_18-11-35_corrigido_polygons_20260910_115544.json"
    ],
    "json_path_background": [
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\background\Corn_53-52_2026-08-25_17-10-44_corrigido_polygons_20260910_101529.json",
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\background\Corn_89-88_2026-08-25_16-09-04_corrigido_polygons_20260910_101613.json",
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\papel\Corn_9-8_2026-08-26_14-27-27_corrigido_polygons_20260910_101829.json",
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\papel\Corn_55-54_2026-08-25_17-07-09_corrigido_polygons_20260910_101732.json",
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\plastico\Corn_98-100_2026-08-25_15-58-53_corrigido_polygons_20260910_101952.json",
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\plastico\SampleCupEmpty_2026-08-25_15-28-54_corrigido_polygons_20260910_102105.json",
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\plastico\Corn_23-22_2026-08-26_14-05-43_corrigido_polygons_20260910_111114.json", #:P
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\plastico\Corn_91-90_2026-08-25_15-54-46_corrigido_polygons_20260910_114715.json",
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\plastico\Corn_27-26_2026-08-26_13-58-49_corrigido_polygons_20260910_115246.json",
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\plastico\Corn_49-48_2026-08-25_17-28-07_corrigido_polygons_20260910_144008.json",
        r"D:\dadosBruno_2\projetoDidion\corrRad\espectro_editor\papel\Corn_49-48_2026-08-25_17-28-48_corrigido_polygons_20260910_143136.json"
    ],

    # Método: "kmeans", "kmeans_guiado", "rf"
    "method": "kmeans_guiado",

    # -------------------------------------------------------------------------
    # KMeans
    # -------------------------------------------------------------------------
    # ATENÇÃO: "pca_components" é usado SOMENTE pelos classificadores KMeans
    # (é removido antes de passar o dict para o sklearn.cluster.KMeans).
    # - None  -> desativa PCA
    # - int N -> aplica PCA e reduz para N componentes antes do KMeans
    # -------------------------------------------------------------------------
    "kmeans_params": {
        "n_clusters": 2,
        "n_init": 10,               # ignorado em kmeans_guiado (usa n_init=1)
        "max_iter": 300,
        "random_state": 42,
        "pca_components": 10,       # <-- AJUSTE AQUI o N de componentes
    },

    # Random Forest
    "rf_params": {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 5,
        "random_state": 42,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "rf_prob_threshold": 0.5,
    "n_background_samples": 5000,

    # Pré-processamento
    "apply_blur": False,
    "blur_kind": "gaussian",
    "blur_ksize": 3,

    "apply_savgol": False,
    "savgol_window": 11,
    "savgol_polyorder": 2,

    "band_selection": None,     # None, "pca", "kbest"
    "band_selection_params": {
        "pca_components": 20,
        "pca_variance_retained": 0.99,
        "kbest_k": 30,
    },
    "use_scaler": True,

    # Pós-processamento
    "postprocess": {
        "majority_filter": True,
        "majority_size": 3,
        "morphological_closing": True,
        "closing_kernel": 11,
        "fill_holes": False,
        "min_area": 16000,
        "circularity_filter": True,
        "circularity_threshold": 0.15,
    },

    # Salvamento
    "save_masked_cube": True,
    "save_crops": True,
    "overwrite": False,
}


def load_config(config_path: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Carrega default + YAML + overrides CLI."""
    config = DEFAULT_CONFIG.copy()

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user_cfg = yaml.safe_load(f) or {}
        config.update(user_cfg)
        logger.info(f"Configuração carregada de {config_path}")

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                config[k] = v

    return config