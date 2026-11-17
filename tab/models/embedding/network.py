"""
Arquitetura da rede:
  - GaussianMixtureFilterBank  -> o encoder (filtros gaussianos aprendidos)
  - SpectralDecoder            -> o MLP "professor" (so existe no treino)
  - SpectralEmbeddingSystem    -> junta os dois
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Banco de filtros espectrais (Gaussian Mixture "cones")
# ---------------------------------------------------------------------------
class GaussianMixtureFilterBank(nn.Module):
    """
    Cada canal de saida e uma soma ponderada de K gaussianas ao longo do
    eixo espectral (bio-inspirado nos cones da visao humana).
    """

    def __init__(self, n_bands, n_channels=3, n_gaussians=3, width_min=None):
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels
        self.n_gaussians = n_gaussians
        self.width_min = width_min if width_min is not None else max(1e-2, 0.01 * n_bands)

        band_axis = torch.arange(n_bands, dtype=torch.float32)
        self.register_buffer("band_axis", band_axis)

        init_centers = torch.stack([
            torch.linspace(0, n_bands - 1, n_gaussians) for _ in range(n_channels)
        ])
        self.centers = nn.Parameter(init_centers)
        self.log_widths = nn.Parameter(
            torch.full((n_channels, n_gaussians), np.log(n_bands / (2 * n_gaussians)))
        )
        self.raw_weights = nn.Parameter(torch.ones(n_channels, n_gaussians))

    def get_filters(self):
        """Retorna os filtros materializados: (n_channels, n_bands)."""
        widths = torch.exp(self.log_widths).clamp(min=self.width_min)
        weights = F.softplus(self.raw_weights)

        centers = self.centers.unsqueeze(-1)
        widths_ = widths.unsqueeze(-1)
        weights_ = weights.unsqueeze(-1)
        axis = self.band_axis.view(1, 1, -1)

        gaussians = weights_ * torch.exp(-0.5 * ((axis - centers) / widths_) ** 2)
        filters = gaussians.sum(dim=1)
        filters = filters / (filters.sum(dim=-1, keepdim=True) + 1e-8)
        return filters

    def forward(self, spectra):
        filters = self.get_filters()
        return spectra @ filters.T


# ---------------------------------------------------------------------------
# 2. Decoder (so existe no treino -- nao e usado na inferencia)
# ---------------------------------------------------------------------------
class SpectralDecoder(nn.Module):
    def __init__(self, n_channels, n_bands, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bands),
        )

    def forward(self, embedding):
        return self.net(embedding)


# ---------------------------------------------------------------------------
# 3. Sistema completo
# ---------------------------------------------------------------------------
class SpectralEmbeddingSystem(nn.Module):
    def __init__(self, n_bands, n_channels=3, n_gaussians=3,
                 hidden_dim=128, width_min=None):
        super().__init__()
        self.filter_bank = GaussianMixtureFilterBank(
            n_bands, n_channels, n_gaussians, width_min
        )
        self.decoder = SpectralDecoder(n_channels, n_bands, hidden_dim)

    def forward(self, spectra):
        embedding = self.filter_bank(spectra)
        reconstructed = self.decoder(embedding)
        return embedding, reconstructed