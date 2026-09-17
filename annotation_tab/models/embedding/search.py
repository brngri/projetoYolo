"""
Busca automatica de hiperparametros (width_min_frac x gamma) + metrica
de balanco dos filtros aprendidos.
"""

import numpy as np
import torch

from .network import SpectralEmbeddingSystem
from .losses import spectral_reconstruction_loss


# ---------------------------------------------------------------------------
# Metrica de balanco dos filtros (sem numeros magicos)
# ---------------------------------------------------------------------------
def compute_filter_balance_score(filters):
    """
    Mede duas coisas independentes sobre os filtros aprendidos:

    1. EQUILIBRIO individual -- altura de pico e largura efetiva de cada
       canal, via coeficiente de variacao (cv_peaks, cv_widths). MENOR e
       melhor (canais parecidos entre si em forma).

    2. SEPARACAO espectral -- distancia media entre os centros dos canais,
       normalizada pelo n_bands. MAIOR e melhor (cada canal olhando uma
       regiao diferente do espectro).

    filters: (n_channels, n_bands), cada linha soma 1
    """
    eps = 1e-8
    n_bands = filters.shape[1]

    peak_heights = filters.max(axis=1)
    cv_peaks = float(peak_heights.std() / (peak_heights.mean() + eps))

    band_axis = np.arange(n_bands)
    centers = (filters * band_axis[None, :]).sum(axis=1)
    variances = (filters * (band_axis[None, :] - centers[:, None]) ** 2).sum(axis=1)
    widths = np.sqrt(np.clip(variances, 0, None))
    cv_widths = float(widths.std() / (widths.mean() + eps))

    n_channels = filters.shape[0]
    distancias = [
        abs(centers[i] - centers[j])
        for i in range(n_channels) for j in range(i + 1, n_channels)
    ]
    separacao = float(np.mean(distancias)) / n_bands if distancias else 0.0

    return {
        "cv_peaks": cv_peaks,
        "cv_widths": cv_widths,
        "separacao": separacao,
        "peak_heights": peak_heights.tolist(),
        "widths": widths.tolist(),
        "centers": centers.tolist(),
    }


# ---------------------------------------------------------------------------
# Busca automatica
# ---------------------------------------------------------------------------
def automated_search(
    spectra_data,
    n_channels=3,
    n_gaussians=3,
    hidden_dim=128,
    alpha=1.0,
    beta=1.0,
    width_min_fracs=None,
    gammas=None,
    epochs_busca=40,
    max_pixels_busca=15000,
    val_fraction=0.15,
    batch_size=4096,
    lr=1e-3,
    device=None,
    seed=0,
    log_cb=None,
    candidate_cb=None,
):
    """
    Grid exaustivo sobre (width_min_frac x gamma). Treina UM candidato por
    vez, avalia e DESCARTA -- nunca mais que 1 modelo em memoria.

    log_cb(msg):           mensagens de log
    candidate_cb(idx, total, candidato_dict):
                            chamado apos cada candidato terminar

    Retorna: (melhor_width_min, melhor_gamma, tabela_resultados)
    """
    from ...services.embedding.trainer import train_embedding_system  # evita import circular

    if log_cb is None:
        log_cb = lambda m: None
    if candidate_cb is None:
        candidate_cb = lambda idx, tot, c: None

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_bands = spectra_data.shape[1]

    if width_min_fracs is None:
        width_min_fracs = [0.01, 0.02, 0.04, 0.08, 0.15]
    if gammas is None:
        gammas = [5.0, 10.0, 20.0, 40.0]

    rng = np.random.default_rng(seed)

    # amostra reduzida para a busca (rapido e leve)
    n_total = spectra_data.shape[0]
    n_amostra = min(max_pixels_busca, n_total)
    idx_amostra = rng.choice(n_total, size=n_amostra, replace=False)
    amostra = spectra_data[idx_amostra]

    n_val = max(1, int(n_amostra * val_fraction))
    perm = rng.permutation(n_amostra)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    spectra_train = amostra[train_idx]
    spectra_val = amostra[val_idx]

    X_val = torch.from_numpy(spectra_val.astype(np.float32)).to(device)

    total = len(width_min_fracs) * len(gammas)
    log_cb(f"[busca] {total} candidatos | {n_amostra} pixels "
           f"({n_amostra-n_val} treino / {n_val} val) | {epochs_busca} epocas cada")

    resultados = []
    idx_cand = 0

    for width_frac in width_min_fracs:
        width_min = max(1e-2, width_frac * n_bands)

        for gamma in gammas:
            idx_cand += 1

            torch.manual_seed(seed + idx_cand)

            model = train_embedding_system(
                spectra_train,
                n_channels=n_channels,
                n_gaussians=n_gaussians,
                hidden_dim=hidden_dim,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                width_min=width_min,
                epochs=epochs_busca,
                batch_size=min(batch_size, max(256, len(spectra_train) // 2)),
                lr=lr,
                device=device,
                verbose=False,
                log_cb=None,
                epoch_cb=None,
            )

            with torch.no_grad():
                _, s_hat_val = model(X_val)
                val_loss, _, _ = spectral_reconstruction_loss(
                    X_val, s_hat_val, alpha, beta
                )

            filters = model.filter_bank.get_filters().detach().cpu().numpy()
            balanco = compute_filter_balance_score(filters)

            cand = {
                "idx": idx_cand,
                "width_min_frac": float(width_frac),
                "width_min": float(width_min),
                "gamma": float(gamma),
                "val_loss": float(val_loss.item()),
                "cv_peaks": balanco["cv_peaks"],
                "cv_widths": balanco["cv_widths"],
                "separacao": balanco["separacao"],
            }

            log_cb(
                f"[busca {idx_cand}/{total}] wf={width_frac:.3f} "
                f"gamma={gamma:5.1f} -> val_loss={cand['val_loss']:.4f} "
                f"cv_peaks={cand['cv_peaks']:.3f} cv_widths={cand['cv_widths']:.3f} "
                f"sep={cand['separacao']:.3f}"
            )

            resultados.append(cand)
            candidate_cb(idx_cand, total, cand)

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # ranking por posicao (menor melhor / maior melhor)
    menor_melhor = ["val_loss", "cv_peaks", "cv_widths"]
    maior_melhor = ["separacao"]

    for chave in menor_melhor:
        ordem = sorted(range(len(resultados)), key=lambda i: resultados[i][chave])
        for pos, i in enumerate(ordem):
            resultados[i][f"rank_{chave}"] = pos

    for chave in maior_melhor:
        ordem = sorted(range(len(resultados)), key=lambda i: -resultados[i][chave])
        for pos, i in enumerate(ordem):
            resultados[i][f"rank_{chave}"] = pos

    chaves_rank = [f"rank_{c}" for c in menor_melhor + maior_melhor]
    for r in resultados:
        r["score_final"] = sum(r[c] for c in chaves_rank) / len(chaves_rank)

    resultados.sort(key=lambda r: r["score_final"])
    melhor = resultados[0]

    log_cb(
        f"[busca] MELHOR: wf={melhor['width_min_frac']:.3f} "
        f"gamma={melhor['gamma']:.1f} (val_loss={melhor['val_loss']:.4f}, "
        f"cv_peaks={melhor['cv_peaks']:.3f}, cv_widths={melhor['cv_widths']:.3f}, "
        f"sep={melhor['separacao']:.3f})"
    )

    return melhor["width_min"], melhor["gamma"], resultados