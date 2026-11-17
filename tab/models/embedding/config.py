"""
Configuracao default da aba de embedding.
"""

import os

DEFAULT_CONFIG = {
    # ---------------------------------------------------------------
    # Caminhos base
    # ---------------------------------------------------------------
    "out_root": r"",

    # ---------------------------------------------------------------
    # Modo de operacao: "train" (treina novo) ou "load" (carrega .pt)
    # ---------------------------------------------------------------
    "mode": "train",

    # Caminhos especificos por modo
    "segmented_dir": r"",   # usado no modo "train"
    "model_path":    r"",   # usado no modo "load"
    "corrected_dir": r"",   # sempre obrigatorio (aplicacao)

    # ---------------------------------------------------------------
    # Estrutura do embedding (treino)
    # ---------------------------------------------------------------
    "n_channels": 3,
    "n_gaussians": 3,
    "hidden_dim": 128,
    "width_min": None,

    # ---------------------------------------------------------------
    # Loss (treino)
    # ---------------------------------------------------------------
    "alpha": 1.0,
    "beta": 1.0,

    # ---------------------------------------------------------------
    # Busca automatica (treino)
    # ---------------------------------------------------------------
    "use_auto_search": True,
    "search_width_min_fracs": [0.01, 0.02, 0.04, 0.08, 0.15],
    "search_gammas": [5.0, 10.0, 20.0, 40.0],
    "search_epochs": 40,
    "search_max_pixels": 15000,
    "save_search_candidates": False,

    # Usado quando busca desligada
    "gamma": 20.0,

    # ---------------------------------------------------------------
    # Treino final (treino)
    # ---------------------------------------------------------------
    "epochs": 200,
    "batch_size": 4096,
    "lr": 1e-3,
    "max_pixels_per_image": 50000,
    "seed": 0,

    # ---------------------------------------------------------------
    # Salvamento
    # ---------------------------------------------------------------
    "save_pseudo_rgb_npy": True,
    "overwrite": False,
}