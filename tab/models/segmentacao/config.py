# -*- coding: utf-8 -*-
"""Configurações padrão e carregamento."""

import os
import logging
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    # Diretórios
    "root_dir": r"",
    "out_root": r"",
    "json_path": [],
    "json_path_background": [],

    # Método: "kmeans", "kmeans_guiado", "rf", "otsu"
    "method": "kmeans",

    # -------------------------------------------------------------------------
    # KMeans
    # -------------------------------------------------------------------------
    "kmeans_params": {
        "n_clusters": 2,
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
        "pca_components": 20,       # None desativa PCA
    },

    # -------------------------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Otsu
    # -------------------------------------------------------------------------
    # Otsu é aplicado em uma banda única OU em um índice (NDVI).
    # Os comprimentos de onda são convertidos em índice de banda usando
    # `cube_start_nm` / `cube_end_nm` e o número de bandas do cubo.
    "otsu_params": {
        "use_ndvi": False,   # se True, calcula NDVI antes do Otsu
        "band_nm": 800.0,    # banda única (usada quando use_ndvi=False)
        "red_nm": 670.0,     # usado se use_ndvi=True
        "nir_nm": 800.0,     # usado se use_ndvi=True
        "invert": False,     # inverte a máscara resultante
    },
    "cube_start_nm": 400.0,
    "cube_end_nm": 1000.0,

    # Pré-processamento
    "apply_blur": False,
    "blur_kind": "gaussian",
    "blur_ksize": 3,

    "apply_savgol": False,
    "savgol_window": 11,
    "savgol_polyorder": 2,

    "band_selection": None,
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