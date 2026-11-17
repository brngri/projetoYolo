"""
Loop de treino do sistema de embedding (final, nao a busca).
"""

import torch

from ...models.embedding.network import SpectralEmbeddingSystem
from ...models.embedding.losses import (
    spectral_reconstruction_loss,
    filter_diversity_penalty,
)


def train_embedding_system(
    spectra_data,
    n_channels=3, n_gaussians=3, hidden_dim=128,
    alpha=1.0, beta=1.0, gamma=20.0, width_min=None,
    epochs=200, batch_size=4096, lr=1e-3,
    device=None, verbose=True,
    log_cb=None,
    epoch_cb=None,
):
    """
    log_cb(msg): mensagens de log
    epoch_cb(epoch, total, metrics_dict): chamado a cada epoca

    Retorna o modelo treinado (em modo eval).
    """
    if log_cb is None:
        log_cb = lambda m: None
    if epoch_cb is None:
        epoch_cb = lambda e, t, m: None

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        log_cb(f"[info] treinando no device: {device}")

    n_bands = spectra_data.shape[1]
    X = torch.from_numpy(spectra_data.astype("float32"))

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )

    model = SpectralEmbeddingSystem(
        n_bands, n_channels, n_gaussians, hidden_dim, width_min
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        epoch_loss = epoch_t1 = epoch_t2 = epoch_div = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            _, s_hat = model(batch)
            recon_loss, t1, t2 = spectral_reconstruction_loss(batch, s_hat, alpha, beta)

            filters = model.filter_bank.get_filters()
            diversity = filter_diversity_penalty(filters)

            loss = recon_loss + gamma * diversity
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_t1 += t1
            epoch_t2 += t2
            epoch_div += diversity.item()
            n_batches += 1

        scheduler.step()

        metrics = {
            "loss": epoch_loss / n_batches,
            "amplitude": epoch_t1 / n_batches,
            "derivada": epoch_t2 / n_batches,
            "diversidade": epoch_div / n_batches,
        }
        epoch_cb(epoch + 1, epochs, metrics)

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            log_cb(
                f"[epoch {epoch+1:4d}/{epochs}] "
                f"loss={metrics['loss']:.4f}  "
                f"amplitude={metrics['amplitude']:.4f}  "
                f"derivada={metrics['derivada']:.4f}  "
                f"diversidade={metrics['diversidade']:.4f}"
            )

    model.eval()
    return model