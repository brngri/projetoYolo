"""
Carregador: le uma pasta de amostragens .npy (H, W, n_bands) e devolve
um unico array (N_pixels, n_bands) com os espectros da classe de interesse.
"""

import os
import glob
import numpy as np


def carregar_espectros_da_pasta(folder_path, max_pixels_por_amostra=None,
                                seed=0, log_cb=None):
    """
    Le todos os .npy de uma pasta (cada um (H,W,n_bands), contendo SO a
    classe de interesse), achata, descarta pixels zerados (fora da classe)
    e concatena tudo em (N_pixels, n_bands).
    """
    if log_cb is None:
        log_cb = lambda m: None

    arquivos = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum .npy encontrado em {folder_path}")

    rng = np.random.default_rng(seed)
    todos = []
    n_bands_ref = None

    for f in arquivos:
        cubo = np.load(f)

        if cubo.ndim != 3:
            log_cb(f"[aviso] {os.path.basename(f)}: shape {cubo.shape} "
                   f"nao e (H,W,n_bands) -- pulando")
            continue

        _, _, n_bands = cubo.shape
        if n_bands_ref is None:
            n_bands_ref = n_bands
        elif n_bands != n_bands_ref:
            log_cb(f"[aviso] {os.path.basename(f)}: {n_bands} bandas != "
                   f"{n_bands_ref} -- pulando")
            continue

        flat = cubo.reshape(-1, n_bands)
        valido = flat.sum(axis=1) > 0
        flat = flat[valido]

        if flat.shape[0] == 0:
            log_cb(f"[aviso] {os.path.basename(f)}: nenhum pixel valido -- pulando")
            continue

        if max_pixels_por_amostra is not None and flat.shape[0] > max_pixels_por_amostra:
            idx = rng.choice(flat.shape[0], size=max_pixels_por_amostra, replace=False)
            flat = flat[idx]

        todos.append(flat)
        log_cb(f"[dados] {os.path.basename(f)}: {flat.shape[0]} pixels validos")

    if not todos:
        raise RuntimeError("Nenhum pixel valido encontrado em nenhuma amostragem.")

    spectra_all = np.concatenate(todos, axis=0).astype(np.float32)
    log_cb(f"[dados] total: {spectra_all.shape[0]} pixels, {spectra_all.shape[1]} bandas")
    return spectra_all