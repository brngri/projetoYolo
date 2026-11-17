"""
Plot dos filtros espectrais aprendidos (learned_filters.npy)
==============================================================
Gera o mesmo tipo de grafico do PDF ("Exemplos de Filtros Espectrais
Bio-Inspirados"): um Gaussian mixture por canal, sobrepostos no eixo espectral.

O que olhar no resultado:
  - Os filtros devem cobrir regioes DIFERENTES do espectro (picos separados).
    Se todos colapsarem no mesmo lugar, o embedding nao aprendeu nada util.
  - Formato suave (gaussianas), sem ruido ou picos degenerados/muito estreitos.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURACAO - EDITE AQUI
# =============================================================================
FILTERS_PATH = r"E:\ProjetoDogChow\03_resultados\yolo\240726\filters\learned_filters.npy"   # gerado pelo script de treino
OUTPUT_PLOT_PATH = r"E:\ProjetoDogChow\03_resultados\yolo\240726\filters\learned_filters_plot.png"  # onde salvar a figura
# =============================================================================


def plot_learned_filters(filters: np.ndarray, save_path: str = None):
    """
    filters: array (n_channels, n_bands), como retornado por get_learned_filters()
    """
    n_channels, n_bands = filters.shape
    colors = ["#D85A30", "#639922", "#378ADD", "#7F77DD", "#D4537E", "#EF9F27"]

    plt.figure(figsize=(9, 5))
    for c in range(n_channels):
        label = f"Canal {c}" if n_channels != 3 else ["Canal 0 (R)", "Canal 1 (G)", "Canal 2 (B)"][c]
        plt.plot(
            np.arange(n_bands),
            filters[c],
            label=label,
            color=colors[c % len(colors)],
            linewidth=2,
        )

    plt.title("Filtros Espectrais Aprendidos (Gaussian Mixture)")
    plt.xlabel("Indice da banda espectral")
    plt.ylabel("Peso do filtro")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[info] plot salvo em {save_path}")

    plt.show()


if __name__ == "__main__":
    filters = np.load(FILTERS_PATH)
    print("Shape dos filtros carregados:", filters.shape)
    plot_learned_filters(filters, save_path=OUTPUT_PLOT_PATH)
