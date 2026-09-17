"""
Inferencia: aplicar embedding num cubo HSI -> pseudo-RGB.
Tambem: carregar modelo salvo em disco.
"""

import numpy as np
import torch

from ...models.embedding.network import SpectralEmbeddingSystem


@torch.no_grad()
def cube_to_pseudo_rgb(model, cube, device=None):
    """
    cube: (H, W, n_bands)
    retorna: (H, W, n_channels) uint8
    """
    device = device or next(model.parameters()).device
    H, W, n_bands = cube.shape
    flat = torch.from_numpy(cube.reshape(-1, n_bands).astype("float32")).to(device)

    embedding = model.filter_bank(flat)
    embedding = embedding.cpu().numpy().reshape(H, W, -1)

    out = np.zeros_like(embedding, dtype=np.uint8)
    for c in range(embedding.shape[-1]):
        ch = embedding[..., c]
        lo, hi = np.percentile(ch, [1, 99])
        ch_norm = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
        out[..., c] = (ch_norm * 255).astype(np.uint8)

    return out


def get_learned_filters(model):
    """Retorna os filtros aprendidos (n_channels, n_bands) como numpy."""
    return model.filter_bank.get_filters().detach().cpu().numpy()


def load_model_from_checkpoint(path, device=None):
    """
    Carrega um modelo salvo em disco.

    Formato aceito: dict com as chaves
        state_dict, n_bands, n_channels, n_gaussians, hidden_dim, width_min
    (esse e o formato gerado pela propria aba em modo "train").

    Retorna: (model, meta_dict)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Falha ao carregar '{path}': {e}")

    required = ["state_dict", "n_bands", "n_channels",
                "n_gaussians", "hidden_dim", "width_min"]
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Checkpoint invalido: esperado dict com {required}, "
            f"recebido {type(ckpt).__name__}."
        )
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError(
            f"Checkpoint invalido: faltando chaves {missing}. "
            f"Esperado dict com {required}."
        )

    model = SpectralEmbeddingSystem(
        n_bands=ckpt["n_bands"],
        n_channels=ckpt["n_channels"],
        n_gaussians=ckpt["n_gaussians"],
        hidden_dim=ckpt["hidden_dim"],
        width_min=ckpt["width_min"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    meta = {
        "n_bands":    ckpt["n_bands"],
        "n_channels": ckpt["n_channels"],
        "n_gaussians": ckpt["n_gaussians"],
        "hidden_dim": ckpt["hidden_dim"],
        "width_min":  ckpt["width_min"],
        "gamma":      ckpt.get("gamma"),
    }
    return model, meta