#!/usr/bin/env python3
"""
spectral_embedding_learned10.py

Aprende um embedding espectral (224 bandas -> 3 canais) e gera pseudo-RGB pro YOLO.

ARQUITETURA EM DUAS FASES:
  FASE 1 - TREINO (roda uma vez):
      - Coleta pixels de foreground de TODAS as classes
      - Usa máscara calibrada POR GRUPO DE AQUISIÇÃO
      - Treina UM filter bank compartilhado
      - Salva .pt (embedding) e .npy (filtros)

  FASE 2 - APLICAÇÃO:
      - Carrega o .pt treinado
      - Para cada classe, aplica o embedding nas .npy dessa classe
      - Salva pseudo-RGB em pseudo_rgb/{classe}/*.png

GRUPOS:
  Grupo A (bandeja com múltiplas amêndoas + faixa de calibração):
      AdHul, BlackSpot, ClassA, ClassB, ClassC, ClassD, Dirty, Insect,
      Shrivel, Moisture1-12

  Grupo B (amêndoa individual em cena limpa):
      Nut1 - Nut30
      (as sub-classes físicas 1-4 dos Nuts são LABELS YOLO, não afetam a máscara)

MÉTODO DE THRESHOLD: Otsu global (otsu_global_mean).
  O diagnóstico mostrou que este método é robusto nesse dataset.
  otsu_local e percentil_70 foram descartados por gerarem máscaras ruins.
"""

import os
import re
import glob
import math
from pathlib import Path
from collections import defaultdict, Counter
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. Banco de filtros espectrais aprendidos (Gaussian Mixture "cones")
# ===========================================================================
class GaussianMixtureFilterBank(nn.Module):
    def __init__(self, n_bands: int, n_channels: int = 3, n_gaussians: int = 3):
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels
        self.n_gaussians = n_gaussians

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

    def get_filters(self) -> torch.Tensor:
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

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        filters = self.get_filters()
        return spectra @ filters.T


# ===========================================================================
# 2. Decoder
# ===========================================================================
class SpectralDecoder(nn.Module):
    def __init__(self, n_channels: int, n_bands: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bands),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


# ===========================================================================
# 3. Sistema completo
# ===========================================================================
class SpectralEmbeddingSystem(nn.Module):
    def __init__(self, n_bands: int, n_channels: int = 3,
                 n_gaussians: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.filter_bank = GaussianMixtureFilterBank(n_bands, n_channels, n_gaussians)
        self.decoder = SpectralDecoder(n_channels, n_bands, hidden_dim)

    def forward(self, spectra: torch.Tensor):
        embedding = self.filter_bank(spectra)
        reconstructed = self.decoder(embedding)
        return embedding, reconstructed


def spectral_reconstruction_loss(s, s_hat, alpha=1.0, beta=1.0):
    """D(s, ŝ) = α·Σ|s - ŝ| + β·Σ|s' - ŝ'|"""
    term1 = (s - s_hat).abs().sum(dim=-1).mean()
    ds = s[:, 1:] - s[:, :-1]
    ds_hat = s_hat[:, 1:] - s_hat[:, :-1]
    term2 = (ds - ds_hat).abs().sum(dim=-1).mean()
    loss = alpha * term1 + beta * term2
    return loss, term1.item(), term2.item()


def filter_diversity_penalty(filters: torch.Tensor) -> torch.Tensor:
    """Penaliza similaridade entre filtros via similaridade de cosseno."""
    filters_norm = F.normalize(filters, dim=-1)
    similarity_matrix = filters_norm @ filters_norm.T
    n_channels = filters.shape[0]
    mask = ~torch.eye(n_channels, dtype=torch.bool, device=filters.device)
    return similarity_matrix[mask].mean()


# ===========================================================================
# 4. Treino
# ===========================================================================
def train_embedding_system(
    spectra_data: np.ndarray,
    n_channels: int = 3,
    n_gaussians: int = 3,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 5.0,
    epochs: int = 200,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: str = None,
    verbose_every: int = 20,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] treinando no device: {device}")

    n_bands = spectra_data.shape[1]
    X = torch.from_numpy(spectra_data.astype(np.float32))

    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    model = SpectralEmbeddingSystem(n_bands, n_channels, n_gaussians).to(device)
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
        if (epoch + 1) % verbose_every == 0 or epoch == 0:
            print(
                f"[epoch {epoch+1:4d}/{epochs}] "
                f"loss={epoch_loss/n_batches:.4f}  "
                f"amplitude_term={epoch_t1/n_batches:.4f}  "
                f"derivative_term={epoch_t2/n_batches:.4f}  "
                f"diversity_penalty={epoch_div/n_batches:.4f}"
            )

    model.eval()
    return model


# ===========================================================================
# 5. Aplicar embedding -> pseudo-RGB
# ===========================================================================
@torch.no_grad()
def cube_to_pseudo_rgb(model, cube, mask=None, device=None):
    """
    Aplica embedding -> pseudo-RGB.
    Se mask for fornecida, normaliza cada canal usando SÓ os pixels do objeto
    (muito mais contraste do que normalizar sobre a imagem inteira).
    """
    device = device or next(model.parameters()).device
    H, W, n_bands = cube.shape
    flat = torch.from_numpy(cube.reshape(-1, n_bands).astype(np.float32)).to(device)

    embedding = model.filter_bank(flat).cpu().numpy().reshape(H, W, -1)

    out = np.zeros((H, W, embedding.shape[-1]), dtype=np.uint8)
    for c in range(embedding.shape[-1]):
        ch = embedding[..., c]
        if mask is not None and mask.any():
            vals = ch[mask]                    # <-- só os pixels da amêndoa
            lo, hi = np.percentile(vals, [1, 99])
        else:
            lo, hi = np.percentile(ch, [1, 99])
        ch_norm = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
        out[..., c] = (ch_norm * 255).astype(np.uint8)
    return out


def get_learned_filters(model: SpectralEmbeddingSystem) -> np.ndarray:
    return model.filter_bank.get_filters().detach().cpu().numpy()


# ===========================================================================
# 6. Padronização de eixos (com guard contra shapes suspeitos)
# ===========================================================================
def padronizar_cubo(cube: np.ndarray) -> np.ndarray:
    """
    Reorganiza .npy para (H, W, B).
    Aceita: (total, 224), (224, H, W), (H, W, 224).
    Levanta ValueError se o shape for suspeito (ex: Nut6 corrompido).
    """
    if cube.ndim == 2 and cube.shape[1] == 224:
        total_pixels = cube.shape[0]
        h = int(math.sqrt(total_pixels))
        while total_pixels % h != 0:
            h -= 1
        w = total_pixels // h
        cube = cube.reshape(h, w, 224)
    elif cube.ndim == 3 and cube.shape[0] == 224:
        cube = np.transpose(cube, (1, 2, 0))

    # Guard: rejeita shapes absurdos
    if cube.ndim == 3:
        H, W, B = cube.shape
        if H < 50 or W < 50 or B != 224:
            raise ValueError(f"shape suspeito: {cube.shape}")
    return cube


# ===========================================================================
# 7. Máscara de foreground (Otsu global + blur + opening + filtros)
# ===========================================================================
def blur_circularity_mask(
    cube: np.ndarray,
    band_idx: int = None,
    threshold: float = None,
    blur_sigma: float = 3.0,
    mask_threshold: float = 0.5,
    opening_radius: int = 5,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = None,     # <-- NOVO
    min_circularity: float = 0.5,
    exclude_left_cols: int = 0,
    exclude_right_cols: int = 0,
    return_intermediate: bool = False,
    debug: bool = False,
):
    """
    Máscara de foreground: Otsu global -> blur na máscara -> re-binarização
    -> opening -> filtro por área e circularidade.
    """
    from skimage import morphology, measure
    from scipy.ndimage import gaussian_filter

    cube = padronizar_cubo(cube)
    if cube.ndim != 3:
        raise ValueError(f"Esperado cubo 3D (H,W,B), obtido {cube.ndim}D com shape {cube.shape}")

    # Banda de referência: média espectral (band_idx=None)
    if band_idx is not None:
        ref = cube[..., band_idx]
    else:
        ref = cube.mean(axis=-1)

    if ref.ndim != 2:
        raise ValueError(f"ref deve ser 2D, obtido {ref.ndim}D")

    # Cortes de borda (faixa de calibração + bandeja)
    if exclude_left_cols > 0 or exclude_right_cols > 0:
        ref = ref.copy()
        if exclude_left_cols > 0:
            ref[:, :exclude_left_cols] = ref.min()
        if exclude_right_cols > 0:
            ref[:, -exclude_right_cols:] = ref.min()

    # 1. Otsu global (método confirmado como bom pelo diagnóstico)
    if threshold is None:
        try:
            from skimage.filters import threshold_otsu
            threshold = threshold_otsu(ref)
        except ImportError:
            threshold = np.percentile(ref, 20)

    mask1_otsu = ref > threshold

    # 2. Blur na máscara (suaviza serrilhado)
    mask_blurred = gaussian_filter(mask1_otsu.astype(np.float32), sigma=blur_sigma)
    mask = mask_blurred > mask_threshold

    # 3. Opening (corta pontes entre amêndoas)
    if mask.ndim == 2:
        mask = morphology.opening(mask, morphology.disk(opening_radius))

    # 4. Filtro por área e circularidade
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    total_pixels = mask.shape[0] * mask.shape[1]
    min_area = min_area_ratio * total_pixels
    max_area = max_area_ratio * total_pixels if max_area_ratio is not None else float("inf")

    if debug:
        print(f"[debug] total_pixels={total_pixels}  min_area={min_area:.0f}  "
              f"max_area={max_area:.0f}  min_circularity={min_circularity}")
        for p in sorted(props, key=lambda p: -p.area):
            circ = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
            print(f"[debug]   area={p.area:8.0f}  circ={circ:.3f}")

    clean_mask = np.zeros_like(mask, dtype=bool)
    for p in props:
        if p.area < min_area:
            continue
        if p.area > max_area:              # <-- NOVO
            continue
        circ = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
        if circ >= min_circularity:
            clean_mask[labeled == p.label] = True

    if return_intermediate:
        return clean_mask, mask1_otsu
    return clean_mask


def save_mask_visualization(cube, mask, img_name, masks_output_folder, mask_raw=None):
    """Salva 3 PNGs: máscara Otsu bruta, máscara final, e overlay."""
    from PIL import Image

    os.makedirs(masks_output_folder, exist_ok=True)

    if mask_raw is not None:
        Image.fromarray((mask_raw.astype(np.uint8) * 255)).save(
            os.path.join(masks_output_folder, f"{img_name}_mask1_otsu.png"))
    Image.fromarray((mask.astype(np.uint8) * 255)).save(
        os.path.join(masks_output_folder, f"{img_name}_mask2_final.png"))

    gray = cube.mean(axis=-1)
    lo, hi = np.percentile(gray, [1, 99])
    gray_norm = np.clip((gray - lo) / (hi - lo + 1e-8), 0, 1)
    gray_rgb = np.stack([gray_norm] * 3, axis=-1)

    overlay = gray_rgb.copy()
    overlay[mask, 0] = 0.6 * overlay[mask, 0] + 0.4
    overlay[mask, 1] = 0.6 * overlay[mask, 1]
    overlay[mask, 2] = 0.6 * overlay[mask, 2]
    Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(masks_output_folder, f"{img_name}_overlay.png"))


# ===========================================================================
# 8. Descoberta de classes
# ===========================================================================
def extrair_classe(nome_arquivo: str) -> str:
    """Ex: 'Nut2_2026-09-01_20-52-56_corrigido.npy' -> 'Nut2'"""
    stem = os.path.splitext(os.path.basename(nome_arquivo))[0]
    m = re.match(r"^(.+?)_\d{4}-\d{2}-\d{2}", stem)
    return m.group(1) if m else stem.split("_")[0]


IGNORAR_PASTAS = {"renomeados", "_tmp", "teste", "backup"}


def descobrir_classes(input_folder: str) -> dict:
    """Retorna {classe: [paths]}. Classe = nome da subpasta (ou prefixo do arquivo)."""
    classes = defaultdict(list)
    npy_files = sorted(glob.glob(os.path.join(input_folder, "**", "*.npy"), recursive=True))
    npy_files = [f for f in npy_files if "renomeados" not in f]

    for f in npy_files:
        rel = os.path.relpath(f, input_folder)
        parts = rel.split(os.sep)
        classe = parts[0] if len(parts) > 1 else extrair_classe(os.path.basename(f))
        if classe in IGNORAR_PASTAS:
            continue
        classes[classe].append(f)

    return dict(classes)


# ===========================================================================
# 9. Fase 1 - carregamento com máscara por grupo
# ===========================================================================
def load_dataset_de_grupo(
    classes_dict: dict,
    mask_fn,
    grupo_nome: str,
    max_pixels_per_image: int = 50000,
    masks_output_folder: str = None,
):
    """
    Coleta pixels de foreground para UM grupo específico.
    mask_fn: já parcializado com os parâmetros do grupo.
    """
    all_spectra = []
    rng = np.random.default_rng(0)
    erros = []

    for classe, arquivos in sorted(classes_dict.items()):
        print(f"\n[Fase 1][{grupo_nome}] classe={classe}  arquivos={len(arquivos)}")
        for fpath in arquivos:
            img_name = os.path.splitext(os.path.basename(fpath))[0]
            try:
                cube = np.load(fpath)
                cube = padronizar_cubo(cube)
            except Exception as e:
                print(f"  [skip] {img_name}: {e}")
                erros.append((img_name, str(e)))
                continue

            if masks_output_folder is not None:
                try:
                    mask, mask_raw = mask_fn(cube, return_intermediate=True)
                except TypeError:
                    mask = mask_fn(cube)
                    mask_raw = None
                save_mask_visualization(cube, mask, img_name,
                                        masks_output_folder, mask_raw=mask_raw)
            else:
                mask = mask_fn(cube)

            spectra_fg = cube[mask]
            if spectra_fg.shape[0] == 0:
                print(f"  [warn] {img_name}: 0 pixels de foreground, pulando")
                continue

            if spectra_fg.shape[0] > max_pixels_per_image:
                idx = rng.choice(spectra_fg.shape[0],
                                 size=max_pixels_per_image, replace=False)
                spectra_fg = spectra_fg[idx]

            all_spectra.append(spectra_fg)
            print(f"  [dataset] {img_name}: {spectra_fg.shape[0]} pixels")

    if not all_spectra:
        raise RuntimeError(f"Nenhum pixel de foreground coletado para o grupo {grupo_nome}.")

    if masks_output_folder is not None:
        print(f"\n[info][{grupo_nome}] mascaras salvas em {masks_output_folder}")

    if erros:
        print(f"\n[aviso][{grupo_nome}] {len(erros)} arquivos ignorados por erro:")
        for nome, msg in erros:
            print(f"  - {nome}: {msg}")

    spectra_all = np.concatenate(all_spectra, axis=0).astype(np.float32)
    print(f"\n[Fase 1][{grupo_nome}] total: {spectra_all.shape[0]} pixels, "
          f"{spectra_all.shape[1]} bandas")
    return spectra_all

# ===========================================================================
# 10. Fase 2 - aplicação por classe
# ===========================================================================
@torch.no_grad()
def aplicar_embedding_em_pasta(
    model, input_folder, output_folder, mask_fn=None, device=None,
):
    from PIL import Image
    device = device or next(model.parameters()).device
    os.makedirs(output_folder, exist_ok=True)

    npy_files = sorted(glob.glob(os.path.join(input_folder, "**", "*.npy"), recursive=True))
    npy_files = [f for f in npy_files if "renomeados" not in f]
    if not npy_files:
        print(f"  [warn] nenhum .npy em {input_folder}")
        return 0

    n_ok = 0
    for fpath in npy_files:
        try:
            cube = np.load(fpath)
            cube = padronizar_cubo(cube)
        except Exception as e:
            print(f"  [skip] {os.path.basename(fpath)}: {e}")
            continue

        # calcula máscara pra normalizar com contraste correto
        mask = None
        if mask_fn is not None:
            try:
                mask = mask_fn(cube)
            except Exception:
                mask = None

        pseudo_rgb = cube_to_pseudo_rgb(model, cube, mask=mask, device=device)
        out_name = os.path.splitext(os.path.basename(fpath))[0] + "_pseudo_rgb.png"
        Image.fromarray(pseudo_rgb).save(os.path.join(output_folder, out_name))
        n_ok += 1
    return n_ok


# ===========================================================================
# CONFIGURAÇÃO - EDITE AQUI
# ===========================================================================
INPUT_FOLDER = r"E:\ProjetoCrainwalnut\preprocessed\coleta_20260901\FX17\FX17"

# Raiz única onde tudo vai ficar (um subdiretório por grupo)
OUTPUT_ROOT = r"E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX171"

# ---- Caminhos por grupo ----
OUTPUT_MODEL_PATH_POR_GRUPO = {
    "A": os.path.join(OUTPUT_ROOT, "grupo_A", "spectral_embedding_A.pt"),
    "B": os.path.join(OUTPUT_ROOT, "grupo_B", "spectral_embedding_B.pt"),
    "C": os.path.join(OUTPUT_ROOT, "grupo_C", "spectral_embedding_C.pt"),
}
OUTPUT_FILTERS_PATH_POR_GRUPO = {
    "A": os.path.join(OUTPUT_ROOT, "grupo_A", "filters", "learned_filters_A.npy"),
    "B": os.path.join(OUTPUT_ROOT, "grupo_B", "filters", "learned_filters_B.npy"),
    "C": os.path.join(OUTPUT_ROOT, "grupo_C", "filters", "learned_filters_C.npy"),
}
OUTPUT_PSEUDO_RGB_FOLDER_POR_GRUPO = {
    "A": os.path.join(OUTPUT_ROOT, "grupo_A", "pseudo_rgb"),
    "B": os.path.join(OUTPUT_ROOT, "grupo_B", "pseudo_rgb"),
    "C": os.path.join(OUTPUT_ROOT, "grupo_C", "pseudo_rgb"),
}
MASKS_OUTPUT_FOLDER_POR_GRUPO = {
    "A": os.path.join(OUTPUT_ROOT, "grupo_A", "masks"),
    "B": os.path.join(OUTPUT_ROOT, "grupo_B", "masks"),
    "C": os.path.join(OUTPUT_ROOT, "grupo_C", "masks"),
}

# ---- Fases a executar ----
FAZER_TREINO    = True
FAZER_APLICACAO = True
FORCAR_RETREINO = False

# ===========================================================================
# GRUPOS DE AQUISIÇÃO
# ===========================================================================
# Grupo A: bandeja com múltiplas amêndoas + faixa de calibração na direita
#   (H ~470-630, múltiplos componentes detectados)
# Grupo B: amêndoa individual em cena limpa
#   (H ~170-380, 1 componente por imagem)
#
# Shrivel foi mantido no grupo A conforme sua decisão.
# Se a máscara não funcionar bem pra Shrivel (H~270, tipicamente individual),
# mova para o grupo B apenas removendo-o do set abaixo.
# ===========================================================================
CLASSES_GRUPO_A = {
    "AdHul", "BlackSpot",
    "ClassA", "ClassB", "ClassC", "ClassD",
    "Dirty", "Insect", "Shrivel",
}
# Moisture* sai do grupo A e vai pro grupo C (tratamento especial)
CLASSES_GRUPO_C = {f"Moisture{i}" for i in range(1, 13)}


def grupo_da_classe(classe: str) -> str:
    if classe in CLASSES_GRUPO_C:
        return "C"
    if classe in CLASSES_GRUPO_A:
        return "A"
    return "B"


# ---------------------------------------------------------------------------
# Sub-classes físicas dos Nuts (1-4). INFORMATIVO: viram LABELS YOLO na etapa
# de anotação. Não afetam a máscara nem o embedding (todos os Nuts são
# treinados juntos no grupo B).
# ---------------------------------------------------------------------------
SUB_CLASSES_NUTS = {
    1: ["Nut10", "Nut13", "Nut23", "Nut24"],
    2: ["Nut1", "Nut2", "Nut3", "Nut4", "Nut5", "Nut6", "Nut7", "Nut8", "Nut9",
        "Nut11", "Nut14", "Nut16", "Nut17", "Nut19", "Nut20", "Nut22",
        "Nut25", "Nut26", "Nut27", "Nut28", "Nut29"],
    3: ["Nut15", "Nut18", "Nut21", "Nut30"],
    4: ["Nut12"],
}

# ===========================================================================
# PARÂMETROS DA MÁSCARA POR GRUPO
# ===========================================================================
PARAMS_MASK_POR_GRUPO = {
    "A": dict(
        band_idx=None,
        blur_sigma=3.0,
        mask_threshold=0.5,
        opening_radius=5,
        min_area_ratio=0.0005,
        max_area_ratio=None,
        min_circularity=0.0,
        exclude_left_cols=0,
        exclude_right_cols=0,
        debug=False,
    ),
    "B": dict(
        band_idx=None,
        blur_sigma=3.0,
        mask_threshold=0.5,
        opening_radius=5,
        min_area_ratio=0.0005,
        max_area_ratio=None,
        min_circularity=0.0,
        exclude_left_cols=0,
        exclude_right_cols=0,
        debug=False,
    ),
    # Grupo C — Moisture*: amêndoas em embalagem
    # A embalagem é grande e retangular, então usamos:
    #   - max_area_ratio: rejeita blobs muito grandes (a embalagem)
    #   - min_circularity: rejeita blobs retangulares (a embalagem)
    "C": dict(
        band_idx=None,
        blur_sigma=3.0,
        mask_threshold=0.5,
        opening_radius=5,
        min_area_ratio=0.0005,
        max_area_ratio=0.03,       # rejeita blob > 3% da imagem
        min_circularity=0.55,      # circularidade alta = só forma redonda
        exclude_left_cols=0,
        exclude_right_cols=0,
        debug=False,
    ),
}

# ===========================================================================
# HIPERPARÂMETROS DO EMBEDDING
# ===========================================================================
N_CHANNELS = 3
N_GAUSSIANS = 3
ALPHA = 1.0
BETA = 1.0
GAMMA = 20.0
EPOCHS = 200
BATCH_SIZE = 4096
LR = 1e-3
MAX_PIXELS_PER_IMAGE = 50000


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    if not os.path.isdir(INPUT_FOLDER):
        raise FileNotFoundError(f"INPUT_FOLDER não existe: {INPUT_FOLDER}")

    # ---- Descobre classes ----
    classes_dict = descobrir_classes(INPUT_FOLDER)
    print(f"[info] {len(classes_dict)} classes encontradas:")
    for c in sorted(classes_dict.keys()):
        print(f"       {c:<15s} grupo={grupo_da_classe(c)}  arquivos={len(classes_dict[c])}")

    # ---- Sanity check dos grupos ----
    dist = Counter(grupo_da_classe(c) for c in classes_dict.keys())
    print(f"\n[sanity] distribuição dos grupos: {dict(dist)}")
    for g in ("A", "B", "C"):
        classes_g = sorted(c for c in classes_dict if grupo_da_classe(c) == g)
        print(f"[sanity] grupo {g} ({len(classes_g)} classes): {classes_g}")
    print()

    if "AdHul" in classes_dict and grupo_da_classe("AdHul") != "A":
        raise RuntimeError("AdHul caiu fora do grupo A — verifique CLASSES_GRUPO_A")

    # ---- Inspeção rápida dos Moisture (opcional) ----
    if "Moisture1" in classes_dict:
        from skimage import measure, morphology
        from skimage.filters import threshold_otsu
        from scipy.ndimage import gaussian_filter

        for fpath in classes_dict["Moisture1"][:1]:
            cube = padronizar_cubo(np.load(fpath))
            ref = cube.mean(axis=-1)
            mask1 = ref > threshold_otsu(ref)
            m = gaussian_filter(mask1.astype(np.float32), sigma=3.0) > 0.5
            m = morphology.opening(m, morphology.disk(5))
            labeled = measure.label(m)
            total = m.shape[0] * m.shape[1]
            print(f"\n[INSPEÇÃO] {os.path.basename(fpath)}  total_pixels={total}")
            for p in sorted(measure.regionprops(labeled), key=lambda p: -p.area):
                circ = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
                area_ratio = p.area / total
                print(f"  area={p.area:8.0f}  area_ratio={area_ratio:.5f}  "
                      f"circ={circ:.3f}  bbox={p.bbox}")

    grupos_map = {c: grupo_da_classe(c) for c in classes_dict.keys()}

    # =================================================================
    # LOOP POR GRUPO (A, B, C)
    # =================================================================
    for grupo in ("A", "B", "C"):
        classes_do_grupo = {c: v for c, v in classes_dict.items()
                            if grupos_map[c] == grupo}

        if not classes_do_grupo:
            print(f"\n[info] grupo {grupo}: nenhuma classe encontrada, pulando.")
            continue

        model_path    = OUTPUT_MODEL_PATH_POR_GRUPO[grupo]
        filters_path  = OUTPUT_FILTERS_PATH_POR_GRUPO[grupo]
        pseudo_folder = OUTPUT_PSEUDO_RGB_FOLDER_POR_GRUPO[grupo]
        masks_folder  = MASKS_OUTPUT_FOLDER_POR_GRUPO[grupo]
        mask_fn = partial(blur_circularity_mask, **PARAMS_MASK_POR_GRUPO[grupo])

        print("\n" + "#" * 70)
        print(f"# GRUPO {grupo} — {len(classes_do_grupo)} classes")
        for c in sorted(classes_do_grupo.keys()):
            print(f"#   {c}: {len(classes_do_grupo[c])} arquivos")
        print(f"# modelo:  {model_path}")
        print(f"# pseudo:  {pseudo_folder}")
        print(f"# masks:   {masks_folder}")
        print("#" * 70)

        # ---------------- FASE 1: TREINO ----------------
        if FAZER_TREINO and (FORCAR_RETREINO or not os.path.exists(model_path)):
            print(f"\n[Fase 1][grupo {grupo}] treinando embedding...")

            spectra_all = load_dataset_de_grupo(
                classes_dict=classes_do_grupo,
                mask_fn=mask_fn,
                grupo_nome=grupo,
                max_pixels_per_image=MAX_PIXELS_PER_IMAGE,
                masks_output_folder=masks_folder,
            )

            model = train_embedding_system(
                spectra_all,
                n_channels=N_CHANNELS,
                n_gaussians=N_GAUSSIANS,
                alpha=ALPHA, beta=BETA, gamma=GAMMA,
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                verbose_every=20,
            )

            filters = get_learned_filters(model)
            print(f"[grupo {grupo}] shape dos filtros:", filters.shape)

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"[info][grupo {grupo}] modelo salvo em {model_path}")

            os.makedirs(os.path.dirname(filters_path), exist_ok=True)
            np.save(filters_path, filters)
            print(f"[info][grupo {grupo}] filtros salvos em {filters_path}")

        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"FAZER_TREINO={FAZER_TREINO}, mas {model_path} não existe."
                )
            print(f"\n[info][grupo {grupo}] pulando Fase 1 (modelo já existe)")

        # ---------------- FASE 2: APLICAÇÃO ----------------
        if FAZER_APLICACAO:
            print(f"\n[Fase 2][grupo {grupo}] aplicando embedding por classe...")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Infere n_bands a partir de um .npy válido do grupo
            n_bands = None
            for c in classes_do_grupo:
                try:
                    s = padronizar_cubo(np.load(classes_do_grupo[c][0]))
                    n_bands = s.shape[-1]
                    break
                except Exception:
                    continue
            if n_bands is None:
                raise RuntimeError(f"Não consegui inferir n_bands do grupo {grupo}")

            model = SpectralEmbeddingSystem(
                n_bands=n_bands, n_channels=N_CHANNELS, n_gaussians=N_GAUSSIANS
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print(f"[info][grupo {grupo}] modelo carregado ({n_bands} bandas), device={device}")

            os.makedirs(pseudo_folder, exist_ok=True)

            resumo_grupo = []
            for classe in sorted(classes_do_grupo.keys()):
                out_classe = os.path.join(pseudo_folder, classe)
                print(f"\n[Fase 2][grupo {grupo}] classe={classe}  "
                      f"arquivos={len(classes_do_grupo[classe])}")

                n_ok = aplicar_embedding_em_pasta(
                    model=model,
                    input_folder=os.path.dirname(classes_do_grupo[classe][0]),
                    output_folder=out_classe,
                    mask_fn=mask_fn,
                    device=device,
                )
                resumo_grupo.append((classe, n_ok, len(classes_do_grupo[classe])))
                print(f"  [ok] {n_ok}/{len(classes_do_grupo[classe])} arquivos processados")

            print(f"\n[grupo {grupo}] resumo da aplicação:")
            print(f"  {'classe':<20s} {'ok/total':>10s}")
            for classe, n_ok, total in resumo_grupo:
                print(f"  {classe:<20s} {n_ok:>4d}/{total:<4d}")

    # =================================================================
    # RESUMO FINAL
    # =================================================================
    print("\n" + "=" * 70)
    print("CONCLUÍDO — resultados por grupo")
    print("=" * 70)
    for g in ("A", "B", "C"):
        print(f"\nGrupo {g}:")
        print(f"  modelo:  {OUTPUT_MODEL_PATH_POR_GRUPO[g]}")
        print(f"  filtros: {OUTPUT_FILTERS_PATH_POR_GRUPO[g]}")
        print(f"  pseudo:  {OUTPUT_PSEUDO_RGB_FOLDER_POR_GRUPO[g]}")
        print(f"  masks:   {MASKS_OUTPUT_FOLDER_POR_GRUPO[g]}")











































# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# # ---------------------------------------------------------------------------
# # 1. Banco de filtros espectrais aprendidos (Gaussian Mixture "cones")
# # ---------------------------------------------------------------------------
# class GaussianMixtureFilterBank(nn.Module):
#     """
#     Cada canal de saida (ex: 3, 5, 9...) e uma soma ponderada de K gaussianas
#     ao longo do eixo espectral (bio-inspirado nos cones da visao humana).
#     """

#     def __init__(self, n_bands: int, n_channels: int = 3, n_gaussians: int = 3):
#         super().__init__()
#         self.n_bands = n_bands
#         self.n_channels = n_channels
#         self.n_gaussians = n_gaussians

#         band_axis = torch.arange(n_bands, dtype=torch.float32)
#         self.register_buffer("band_axis", band_axis)

#         init_centers = torch.stack([
#             torch.linspace(0, n_bands - 1, n_gaussians) for _ in range(n_channels)
#         ])

#         self.centers = nn.Parameter(init_centers)
#         self.log_widths = nn.Parameter(torch.full((n_channels, n_gaussians), np.log(n_bands / (2 * n_gaussians))))
#         self.raw_weights = nn.Parameter(torch.ones(n_channels, n_gaussians))

#     def get_filters(self) -> torch.Tensor:
#         """Retorna os filtros materializados: shape (n_channels, n_bands)"""
#         widths = torch.exp(self.log_widths).clamp(min=1e-2)
#         weights = F.softplus(self.raw_weights)

#         centers = self.centers.unsqueeze(-1)
#         widths_ = widths.unsqueeze(-1)
#         weights_ = weights.unsqueeze(-1)
#         axis = self.band_axis.view(1, 1, -1)

#         gaussians = weights_ * torch.exp(-0.5 * ((axis - centers) / widths_) ** 2)
#         filters = gaussians.sum(dim=1)

#         filters = filters / (filters.sum(dim=-1, keepdim=True) + 1e-8)
#         return filters

#     def forward(self, spectra: torch.Tensor) -> torch.Tensor:
#         filters = self.get_filters()
#         embedding = spectra @ filters.T
#         return embedding


# # ---------------------------------------------------------------------------
# # 2. Bloco Decoder (reconstroi o espectro a partir do embedding)
# # ---------------------------------------------------------------------------
# class SpectralDecoder(nn.Module):
#     def __init__(self, n_channels: int, n_bands: int, hidden_dim: int = 128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_channels, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, n_bands),
#         )

#     def forward(self, embedding: torch.Tensor) -> torch.Tensor:
#         return self.net(embedding)


# # ---------------------------------------------------------------------------
# # 3. Sistema completo (filtro + decoder) e metrica D(s, s_hat)
# # ---------------------------------------------------------------------------
# class SpectralEmbeddingSystem(nn.Module):
#     def __init__(self, n_bands: int, n_channels: int = 3, n_gaussians: int = 3, hidden_dim: int = 128):
#         super().__init__()
#         self.filter_bank = GaussianMixtureFilterBank(n_bands, n_channels, n_gaussians)
#         self.decoder = SpectralDecoder(n_channels, n_bands, hidden_dim)

#     def forward(self, spectra: torch.Tensor):
#         embedding = self.filter_bank(spectra)
#         reconstructed = self.decoder(embedding)
#         return embedding, reconstructed


# def spectral_reconstruction_loss(s: torch.Tensor, s_hat: torch.Tensor, alpha: float = 1.0, beta: float = 1.0):
#     """
#     D(s, s_hat) = alpha * integral |s - s_hat| + beta * integral |s' - s_hat'|
#     """
#     term1 = (s - s_hat).abs().sum(dim=-1).mean()

#     ds = s[:, 1:] - s[:, :-1]
#     ds_hat = s_hat[:, 1:] - s_hat[:, :-1]
#     term2 = (ds - ds_hat).abs().sum(dim=-1).mean()

#     loss = alpha * term1 + beta * term2
#     return loss, term1.item(), term2.item()


# def filter_diversity_penalty(filters: torch.Tensor) -> torch.Tensor:
#     """
#     Penaliza similaridade entre os filtros dos diferentes canais, pra evitar
#     que colapsem na mesma regiao do espectro (o "minimo preguicoso" que a
#     loss de reconstrucao sozinha permite -- ela nao se importa se os canais
#     sao redundantes entre si, so se a reconstrucao final fica boa).

#     Calcula a similaridade de cosseno entre cada par de filtros (normalizados)
#     e retorna a media dos pares -- quanto mais parecidos os filtros, maior a
#     penalidade. Adicionar isso na loss forca os canais a se especializarem em
#     regioes diferentes do espectro.

#     filters: (n_channels, n_bands)
#     retorna: escalar (0 = todos os filtros ortogonais entre si, 1 = todos identicos)
#     """
#     filters_norm = F.normalize(filters, dim=-1)  # normaliza cada filtro pra norma 1
#     similarity_matrix = filters_norm @ filters_norm.T  # (n_channels, n_channels)

#     n_channels = filters.shape[0]
#     # pega so os pares fora da diagonal (nao compara filtro com ele mesmo)
#     mask = ~torch.eye(n_channels, dtype=torch.bool, device=filters.device)
#     pairwise_similarities = similarity_matrix[mask]

#     return pairwise_similarities.mean()


# # ---------------------------------------------------------------------------
# # 4. Loop de treinamento
# # ---------------------------------------------------------------------------
# def train_embedding_system(
#     spectra_data: np.ndarray,
#     n_channels: int = 3,
#     n_gaussians: int = 3,
#     alpha: float = 1.0,
#     beta: float = 1.0,
#     gamma: float = 5.0,
#     epochs: int = 200,
#     batch_size: int = 4096,
#     lr: float = 1e-3,
#     device: str = None,
#     verbose_every: int = 20,
# ):
#     """
#     gamma: peso da penalidade de diversidade entre filtros (evita colapso de
#         canais na mesma regiao espectral). 0 desativa; valores tipicos 1-10 --
#         se os filtros ainda colapsarem, aumente; se ficarem "forcados" demais
#         a se separar prejudicando a reconstrucao, diminua.
#     """
#     device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[info] treinando no device: {device}")

#     n_bands = spectra_data.shape[1]
#     X = torch.from_numpy(spectra_data.astype(np.float32))

#     dataset = torch.utils.data.TensorDataset(X)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#     model = SpectralEmbeddingSystem(n_bands, n_channels, n_gaussians).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

#     model.train()
#     for epoch in range(epochs):
#         epoch_loss, epoch_t1, epoch_t2, epoch_div, n_batches = 0.0, 0.0, 0.0, 0.0, 0
#         for (batch,) in loader:
#             batch = batch.to(device, non_blocking=True)
#             optimizer.zero_grad()
#             _, s_hat = model(batch)
#             recon_loss, t1, t2 = spectral_reconstruction_loss(batch, s_hat, alpha, beta)

#             filters = model.filter_bank.get_filters()
#             diversity = filter_diversity_penalty(filters)

#             loss = recon_loss + gamma * diversity
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()
#             epoch_t1 += t1
#             epoch_t2 += t2
#             epoch_div += diversity.item()
#             n_batches += 1

#         scheduler.step()
#         if (epoch + 1) % verbose_every == 0 or epoch == 0:
#             print(
#                 f"[epoch {epoch+1:4d}/{epochs}] "
#                 f"loss={epoch_loss/n_batches:.4f}  "
#                 f"amplitude_term={epoch_t1/n_batches:.4f}  "
#                 f"derivative_term={epoch_t2/n_batches:.4f}  "
#                 f"diversity_penalty={epoch_div/n_batches:.4f}"
#             )

#     model.eval()
#     return model


# # ---------------------------------------------------------------------------
# # 5. Aplicar embedding treinado num cubo HSI inteiro -> pseudo-RGB pro YOLO
# # ---------------------------------------------------------------------------
# @torch.no_grad()
# def cube_to_pseudo_rgb(model: SpectralEmbeddingSystem, cube: np.ndarray, device: str = None) -> np.ndarray:
#     device = device or next(model.parameters()).device
#     H, W, n_bands = cube.shape
#     flat = torch.from_numpy(cube.reshape(-1, n_bands).astype(np.float32)).to(device)

#     embedding = model.filter_bank(flat)
#     embedding = embedding.cpu().numpy().reshape(H, W, -1)

#     out = np.zeros_like(embedding, dtype=np.uint8)
#     for c in range(embedding.shape[-1]):
#         ch = embedding[..., c]
#         lo, hi = np.percentile(ch, [1, 99])
#         ch_norm = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
#         out[..., c] = (ch_norm * 255).astype(np.uint8)

#     return out


# def get_learned_filters(model: SpectralEmbeddingSystem) -> np.ndarray:
#     """Retorna os filtros aprendidos (n_channels, n_bands) para inspecao/plot."""
#     return model.filter_bank.get_filters().detach().cpu().numpy()


# # ---------------------------------------------------------------------------
# # 6. Carregamento de dataset real: um .npy por imagem (reflectancia JA corrigida)
# # ---------------------------------------------------------------------------
# def simple_foreground_mask(cube: np.ndarray, band_idx: int = None, threshold: float = None) -> np.ndarray:
#     """
#     Mascara simples walnut vs. fundo preto/bandeja, baseada em intensidade media.
#     Versao crua, sem tratamento de textura/bordas finas -- ver
#     blur_circularity_mask para a versao recomendada.
#     """
#     if band_idx is not None:
#         ref = cube[..., band_idx]
#     else:
#         ref = cube.mean(axis=-1)

#     if threshold is None:
#         try:
#             from skimage.filters import threshold_otsu
#             threshold = threshold_otsu(ref)
#         except ImportError:
#             threshold = np.percentile(ref, 20)

#     return ref > threshold

# def blur_circularity_mask(
#     cube: np.ndarray,
#     band_idx: int = None,
#     threshold: float = None,
#     blur_sigma: float = 4.0,
#     mask_threshold: float = 0.5,
#     opening_radius: int = 5,
#     min_area_ratio: float = 0.002,
#     min_circularity: float = 0.5,
#     exclude_right_cols: int = 0,
#     return_intermediate: bool = False,
#     debug: bool = False,
# ):
#     from skimage import morphology, measure
#     from scipy.ndimage import gaussian_filter
#     import math

#     # --- PADRONIZAÇÃO DE EIXOS ---
#     if cube.ndim == 2 and cube.shape[1] == 224:
#         total_pixels = cube.shape[0]
#         h = int(math.sqrt(total_pixels))
#         while total_pixels % h != 0:
#             h -= 1
#         w = total_pixels // h
#         cube = cube.reshape(h, w, 224)
#         print(f"[aviso] blur_circularity_mask remodelou para ({h}, {w}, 224)")
#     elif cube.ndim == 3 and cube.shape[0] == 224:
#         cube = np.transpose(cube, (1, 2, 0))
#     # Se ainda não for 3D, levanta erro
#     if cube.ndim != 3:
#         raise ValueError(f"Esperado cubo 3D (H,W,B), obtido {cube.ndim}D com shape {cube.shape}")

#     H, W, B = cube.shape

#     # Seleciona banda de referência
#     if band_idx is not None:
#         ref = cube[..., band_idx]
#     else:
#         ref = cube.mean(axis=-1)

#     # Garante que ref seja 2D
#     if ref.ndim != 2:
#         raise ValueError(f"ref deve ser 2D, obtido {ref.ndim}D")

#     # Corta região de calibração se necessário
#     if exclude_right_cols > 0:
#         ref = ref.copy()
#         ref[:, -exclude_right_cols:] = ref.min()

#     # 1. Otsu
#     if threshold is None:
#         try:
#             from skimage.filters import threshold_otsu
#             threshold = threshold_otsu(ref)
#         except ImportError:
#             threshold = np.percentile(ref, 20)

#     mask1_otsu = ref > threshold

#     # 2. Blur na máscara
#     mask_blurred = gaussian_filter(mask1_otsu.astype(np.float32), sigma=blur_sigma)
#     mask = mask_blurred > mask_threshold

#     # 3. Abertura morfológica (só se mask for 2D)
#     if mask.ndim == 2:
#         mask = morphology.opening(mask, morphology.disk(opening_radius))
#     else:
#         # Se mask for 1D, não há o que fazer, retorna vazia
#         print("[aviso] mask ficou 1D, ignorando abertura")
#         mask = np.zeros_like(ref, dtype=bool)

#     # 4. Filtro por área e circularidade
#     labeled = measure.label(mask)
#     props = measure.regionprops(labeled)
#     total_pixels = mask.shape[0] * mask.shape[1] if mask.ndim == 2 else 0
#     min_area = min_area_ratio * total_pixels

#     if debug:
#         print(f"[debug] total_pixels={total_pixels}  min_area={min_area:.0f}  min_circularity={min_circularity}")
#         for p in sorted(props, key=lambda p: -p.area):
#             circularity = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
#             area_ok = "OK " if p.area >= min_area else "AREA_BAIXA"
#             circ_ok = "OK " if circularity >= min_circularity else "CIRC_BAIXA"
#             print(f"[debug]   componente area={p.area:8.0f}  circularidade={circularity:.3f}  -> area:{area_ok}  circ:{circ_ok}")

#     clean_mask = np.zeros_like(mask, dtype=bool)
#     for p in props:
#         if p.area < min_area:
#             continue
#         circularity = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
#         if circularity >= min_circularity:
#             clean_mask[labeled == p.label] = True

#     mask2_final = clean_mask

#     if return_intermediate:
#         return mask2_final, mask1_otsu
#     return mask2_final


# def save_mask_visualization(
#     cube: np.ndarray,
#     mask: np.ndarray,
#     img_name: str,
#     masks_output_folder: str,
#     mask_raw: np.ndarray = None,
# ):
#     """
#     Salva imagens PNG pra inspecao visual da mascara de foreground:
#       - {img_name}_mask1_otsu.png  -> mascara bruta do Otsu (mask1), se
#                                        mask_raw for informado
#       - {img_name}_mask2_final.png -> mascara final, depois de toda a
#                                        limpeza (blur+abertura+circularidade)
#       - {img_name}_overlay.png     -> intensidade media do cubo com a
#                                        mascara final destacada em vermelho
#     """
#     import os
#     from PIL import Image

#     os.makedirs(masks_output_folder, exist_ok=True)

#     if mask_raw is not None:
#         mask1_img = (mask_raw.astype(np.uint8) * 255)
#         Image.fromarray(mask1_img).save(os.path.join(masks_output_folder, f"{img_name}_mask1_otsu.png"))

#     mask2_img = (mask.astype(np.uint8) * 255)
#     Image.fromarray(mask2_img).save(os.path.join(masks_output_folder, f"{img_name}_mask2_final.png"))

#     gray = cube.mean(axis=-1)
#     lo, hi = np.percentile(gray, [1, 99])
#     gray_norm = np.clip((gray - lo) / (hi - lo + 1e-8), 0, 1)
#     gray_rgb = np.stack([gray_norm] * 3, axis=-1)

#     overlay = gray_rgb.copy()
#     overlay[mask, 0] = 0.6 * overlay[mask, 0] + 0.4 * 1.0
#     overlay[mask, 1] = 0.6 * overlay[mask, 1]
#     overlay[mask, 2] = 0.6 * overlay[mask, 2]

#     overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
#     Image.fromarray(overlay_uint8).save(os.path.join(masks_output_folder, f"{img_name}_overlay.png"))


# def load_dataset_from_npy_folder(
#     folder_path: str,
#     mask_fn=blur_circularity_mask,
#     max_pixels_per_image: int = 50000,
#     masks_output_folder: str = None,
# ):
#     import glob
#     import os
#     import math

#     # npy_files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
#     # if not npy_files:
#     #     raise FileNotFoundError(f"Nenhum .npy encontrado em {folder_path}")

#     npy_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.npy"), recursive=True))
#     # Ignora arquivos dentro da pasta 'renomeados' (se existir)
#     npy_files = [f for f in npy_files if "renomeados" not in f]
#     if not npy_files:
#         raise FileNotFoundError(f"Nenhum .npy encontrado em {folder_path}")

#     all_spectra = []
#     rng = np.random.default_rng(0)

#     for fpath in npy_files:
#         img_name = os.path.splitext(os.path.basename(fpath))[0]   # <--- DEFINIDA AQUI
#         cube = np.load(fpath)

#         # --- PADRONIZAÇÃO DE EIXOS ---
#         if cube.ndim == 2 and cube.shape[1] == 224:
#             total_pixels = cube.shape[0]
#             h = int(math.sqrt(total_pixels))
#             while total_pixels % h != 0:
#                 h -= 1
#             w = total_pixels // h
#             cube = cube.reshape(h, w, 224)
#             print(f"[aviso] {img_name} remodelado para ({h}, {w}, 224)")
#         elif cube.ndim == 3 and cube.shape[0] == 224:
#             cube = np.transpose(cube, (1, 2, 0))
#         # Se ainda não for 3D, pula
#         if cube.ndim != 3:
#             print(f"[aviso] Pulando {img_name}: formato inesperado {cube.shape}")
#             continue

#         # Aplica máscara
#         if masks_output_folder is not None:
#             try:
#                 mask, mask_raw = mask_fn(cube, return_intermediate=True)
#             except TypeError:
#                 mask = mask_fn(cube)
#                 mask_raw = None
#             save_mask_visualization(cube, mask, img_name, masks_output_folder, mask_raw=mask_raw)
#         else:
#             mask = mask_fn(cube)

#         spectra_fg = cube[mask]

#         if spectra_fg.shape[0] > max_pixels_per_image:
#             idx = rng.choice(spectra_fg.shape[0], size=max_pixels_per_image, replace=False)
#             spectra_fg = spectra_fg[idx]

#         all_spectra.append(spectra_fg)
#         print(f"[dataset] {os.path.basename(fpath)}: {spectra_fg.shape[0]} pixels de foreground")

#     if masks_output_folder is not None:
#         print(f"[info] mascaras salvas em {masks_output_folder}")

#     spectra_all = np.concatenate(all_spectra, axis=0).astype(np.float32)
#     print(f"[dataset] total: {spectra_all.shape[0]} pixels, {spectra_all.shape[1]} bandas")
#     return spectra_all

# # =============================================================================
# # CONFIGURACAO - EDITE AQUI OS CAMINHOS E HIPERPARAMETROS
# # =============================================================================
# INPUT_FOLDER = r"E:\ProjetoCrainwalnut\preprocessed\coleta_20260901\FX17\FX17"
# OUTPUT_MODEL_PATH = r"E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX171\spectral_embedding_sakata.pt"
# OUTPUT_FILTERS_PATH = r"E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX171\filters\learned_filters.npy"

# OUTPUT_PSEUDO_RGB_FOLDER = r"E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX171\pseudo_rgb"     # ou None

# MASKS_OUTPUT_FOLDER = r"E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX171\yolo\masks"   # ou None

# # hiperparametros da mascara (threshold Otsu -> blur na mascara -> circularidade)
# BLUR_SIGMA = 3.0             # forca do blur aplicado na MASCARA (pos-Otsu) pra apagar contorno serrilhado
# MASK_THRESHOLD = 0.5         # limiar de re-binarizacao da mascara apos o blur
# OPENING_RADIUS = 5          # PRECISA cortar as pontes de bandeja que conectam nozes vizinhas (senao viram um blob so)
# MIN_AREA_RATIO = 0.0005      # area minima (fracao da imagem) pra contar como noz
# MIN_CIRCULARITY = 0.0        # 1.0 = circulo perfeito; listras/bordas ficam bem abaixo
# EXCLUDE_RIGHT_COLS = 130     # corta a faixa de calibracao fixa na borda direita (ajuste conforme sua imagem)
# MASK_DEBUG = True            # imprime area/circularidade de cada componente (ajuda a calibrar)

# # hiperparametros do embedding / treino
# N_CHANNELS = 3
# N_GAUSSIANS = 3           # obrigatorio manter em 3 -- a diversidade entre canais fica a cargo do GAMMA
# ALPHA = 1.0
# BETA = 1.0
# GAMMA = 20.0             # peso da penalidade de diversidade entre canais (evita colapso na mesma regiao)
# EPOCHS = 200
# BATCH_SIZE = 4096
# LR = 1e-3
# MAX_PIXELS_PER_IMAGE = 50000
# # =============================================================================


# if __name__ == "__main__":
#     import os
#     from functools import partial

#     mask_fn = partial(
#         blur_circularity_mask,
#         blur_sigma=BLUR_SIGMA,
#         mask_threshold=MASK_THRESHOLD,
#         opening_radius=OPENING_RADIUS,
#         min_area_ratio=MIN_AREA_RATIO,
#         min_circularity=MIN_CIRCULARITY,
#         exclude_right_cols=EXCLUDE_RIGHT_COLS,
#         debug=MASK_DEBUG,
#     )

#     if os.path.isdir(INPUT_FOLDER):
#         spectra_all = load_dataset_from_npy_folder(
#             INPUT_FOLDER,
#             mask_fn=mask_fn,
#             max_pixels_per_image=MAX_PIXELS_PER_IMAGE,
#             masks_output_folder=MASKS_OUTPUT_FOLDER,
#         )
#     else:
#         print(f"[aviso] INPUT_FOLDER '{INPUT_FOLDER}' nao encontrada - rodando smoke-test com dados sinteticos.")
#         print("        edite a variavel INPUT_FOLDER no topo do script pra apontar pro seu dataset real.")
#         n_bands = 224
#         n_pixels = 20000
#         rng = np.random.default_rng(42)
#         spectra_all = rng.random((n_pixels, n_bands)).astype(np.float32).cumsum(axis=1)
#         spectra_all /= spectra_all.max()

#     model = train_embedding_system(
#         spectra_all,
#         n_channels=N_CHANNELS,
#         n_gaussians=N_GAUSSIANS,
#         alpha=ALPHA,
#         beta=BETA,
#         gamma=GAMMA,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         lr=LR,
#         verbose_every=20,
#     )

#     filters = get_learned_filters(model)
#     print("Shape dos filtros aprendidos:", filters.shape)

#     os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
#     torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
#     print(f"[info] modelo salvo em {OUTPUT_MODEL_PATH}")

#     os.makedirs(os.path.dirname(OUTPUT_FILTERS_PATH), exist_ok=True)
#     np.save(OUTPUT_FILTERS_PATH, filters)
#     print(f"[info] filtros aprendidos salvos em {OUTPUT_FILTERS_PATH}")

#     if OUTPUT_PSEUDO_RGB_FOLDER is not None and os.path.isdir(INPUT_FOLDER):
#         import glob
#         from PIL import Image
#         import math

#         os.makedirs(OUTPUT_PSEUDO_RGB_FOLDER, exist_ok=True)
#         # npy_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.npy")))

#         npy_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "**", "*.npy"), recursive=True))
#         # Ignora arquivos dentro da pasta 'renomeados' (se existir)
#         npy_files = [f for f in npy_files if "renomeados" not in f]

#         for fpath in npy_files:
#             cube = np.load(fpath)
#             # --- PADRONIZAÇÃO DE EIXOS (igual à usada no carregamento) ---
#             if cube.ndim == 2 and cube.shape[1] == 224:
#                 total_pixels = cube.shape[0]
#                 h = int(math.sqrt(total_pixels))
#                 while total_pixels % h != 0:
#                     h -= 1
#                 w = total_pixels // h
#                 cube = cube.reshape(h, w, 224)
#                 print(f"[aviso] Gerando pseudo-RGB: remodelado para ({h}, {w}, 224)")
#             elif cube.ndim == 3 and cube.shape[0] == 224:
#                 cube = np.transpose(cube, (1, 2, 0))
#             # Se ainda não for 3D, pula este arquivo
#             if cube.ndim != 3:
#                 print(f"[aviso] Pulando {os.path.basename(fpath)}: formato inesperado {cube.shape} para pseudo-RGB")
#                 continue

#             pseudo_rgb = cube_to_pseudo_rgb(model, cube)
#             out_name = os.path.splitext(os.path.basename(fpath))[0] + "_pseudo_rgb.png"
#             out_path = os.path.join(OUTPUT_PSEUDO_RGB_FOLDER, out_name)
#             Image.fromarray(pseudo_rgb).save(out_path)
#             print(f"[info] pseudo-RGB salva em {out_path}")
