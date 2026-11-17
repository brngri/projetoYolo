"""
Gera o fake RGB (pseudo-RGB) de TODOS os .npy de uma pasta, usando o modelo
ja treinado. Script autocontido -- nao precisa importar spectral_embedding_learned.py.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# =============================================================================
# CONFIGURACAO - EDITE AQUI
# =============================================================================
MODEL_PATH = r"E:\ProjetoDogChow\01_dados\preprocessamento\06_07_2026_amostragem\amostras_gerais\furado\embedding\spectral_embedding_walnut.pt"
INPUT_FOLDER = r"E:\ProjetoDogChow\01_dados\preprocessamento\06_07_2026_correcao_radiometrica\amostras_gerais\furado"   # pasta com os .npy de entrada
OUTPUT_FOLDER = r"E:\ProjetoDogChow\01_dados\preprocessamento\06_07_2026_amostragem\amostras_gerais\furado\pseudo_rgb"               # onde salvar os .png

N_BANDS = 224
N_CHANNELS = 3
N_GAUSSIANS = 3   # precisa ser igual ao valor usado no treino
# =============================================================================


class GaussianMixtureFilterBank(nn.Module):
    def __init__(self, n_bands, n_channels=3, n_gaussians=3):
        super().__init__()
        band_axis = torch.arange(n_bands, dtype=torch.float32)
        self.register_buffer("band_axis", band_axis)
        init_centers = torch.stack([
            torch.linspace(0, n_bands - 1, n_gaussians) for _ in range(n_channels)
        ])
        self.centers = nn.Parameter(init_centers)
        self.log_widths = nn.Parameter(torch.full((n_channels, n_gaussians), np.log(n_bands / (2 * n_gaussians))))
        self.raw_weights = nn.Parameter(torch.ones(n_channels, n_gaussians))

    def get_filters(self):
        widths = torch.exp(self.log_widths).clamp(min=1e-2)
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
        return spectra @ self.get_filters().T


class SpectralDecoder(nn.Module):
    def __init__(self, n_channels, n_bands, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_channels, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_bands),
        )

    def forward(self, embedding):
        return self.net(embedding)


class SpectralEmbeddingSystem(nn.Module):
    def __init__(self, n_bands, n_channels=3, n_gaussians=3, hidden_dim=128):
        super().__init__()
        self.filter_bank = GaussianMixtureFilterBank(n_bands, n_channels, n_gaussians)
        self.decoder = SpectralDecoder(n_channels, n_bands, hidden_dim)

    def forward(self, spectra):
        embedding = self.filter_bank(spectra)
        return embedding, self.decoder(embedding)


def cube_to_pseudo_rgb(model, cube, device):
    H, W, n_bands = cube.shape
    flat = torch.from_numpy(cube.reshape(-1, n_bands).astype(np.float32)).to(device)

    with torch.no_grad():
        embedding = model.filter_bank(flat).cpu().numpy().reshape(H, W, -1)

    out = np.zeros_like(embedding, dtype=np.uint8)
    for c in range(embedding.shape[-1]):
        ch = embedding[..., c]
        lo, hi = np.percentile(ch, [1, 99])
        ch_norm = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
        out[..., c] = (ch_norm * 255).astype(np.uint8)

    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] usando device: {device}")

    model = SpectralEmbeddingSystem(N_BANDS, N_CHANNELS, N_GAUSSIANS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"[info] modelo carregado de {MODEL_PATH}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    npy_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.npy")))
    print(f"[info] {len(npy_files)} arquivos .npy encontrados em {INPUT_FOLDER}")

    if not npy_files:
        print(f"[erro] nenhum .npy encontrado em {INPUT_FOLDER} -- confira o caminho.")
        return

    for fpath in npy_files:
        cube = np.load(fpath)  # (H, W, n_bands)
        pseudo_rgb = cube_to_pseudo_rgb(model, cube, device)  # (H, W, 3) uint8

        out_name = os.path.splitext(os.path.basename(fpath))[0] + "_pseudo_rgb.png"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        Image.fromarray(pseudo_rgb).save(out_path)
        print(f"[info] salva: {out_path}")

    print(f"[info] concluido -- {len(npy_files)} imagens geradas em {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()