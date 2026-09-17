"""
Visualizacao: plot dos filtros aprendidos + salvamento das pseudo-RGB.
"""

import os
import numpy as np


def plot_filters(filters, save_path,
                 title="Filtros Espectrais Aprendidos (Gaussian Mixture)"):
    """
    Gera PNG com uma linha por canal (peso do filtro x indice de banda).
    Retorna o caminho salvo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_channels, n_bands = filters.shape
    cores = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple",
             "tab:brown", "tab:pink", "tab:gray"]
    rotulos = {0: "R", 1: "G", 2: "B"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    band_axis = np.arange(n_bands)
    for c in range(n_channels):
        extra = f" ({rotulos[c]})" if c in rotulos and n_channels <= 3 else ""
        ax.plot(band_axis, filters[c], color=cores[c % len(cores)],
                linewidth=2, label=f"Canal {c}{extra}")

    ax.set_title(title)
    ax.set_xlabel("Indice da banda espectral")
    ax.set_ylabel("Peso do filtro")
    ax.legend()
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_pseudo_rgb_png(pseudo_rgb, save_path):
    """
    Salva array (H, W, 3) uint8 como PNG.
    Usa PIL se disponivel, senao cv2.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        from PIL import Image
        Image.fromarray(pseudo_rgb).save(save_path)
        return save_path
    except ImportError:
        pass

    try:
        import cv2
        cv2.imwrite(save_path, cv2.cvtColor(pseudo_rgb, cv2.COLOR_RGB2BGR))
        return save_path
    except ImportError:
        raise RuntimeError(
            "Nenhuma biblioteca de imagem disponivel (PIL ou cv2). "
            "Instale: pip install Pillow"
        )