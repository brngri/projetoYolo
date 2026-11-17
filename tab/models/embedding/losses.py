"""
Losses do sistema: reconstrucao espectral + diversidade entre filtros.
"""

import torch
import torch.nn.functional as F


def spectral_reconstruction_loss(s, s_hat, alpha=1.0, beta=1.0):
    """
    D(s, s_hat) = alpha * |s - s_hat| + beta * |s' - s_hat'|
    """
    term1 = (s - s_hat).abs().sum(dim=-1).mean()
    ds = s[:, 1:] - s[:, :-1]
    ds_hat = s_hat[:, 1:] - s_hat[:, :-1]
    term2 = (ds - ds_hat).abs().sum(dim=-1).mean()
    return alpha * term1 + beta * term2, term1.item(), term2.item()


def filter_diversity_penalty(filters):
    """
    Penaliza similaridade entre os filtros dos diferentes canais, pra
    evitar colapso na mesma regiao do espectro.
    """
    filters_norm = F.normalize(filters, dim=-1)
    similarity_matrix = filters_norm @ filters_norm.T
    n_channels = filters.shape[0]
    mask = ~torch.eye(n_channels, dtype=torch.bool, device=filters.device)
    return similarity_matrix[mask].mean()