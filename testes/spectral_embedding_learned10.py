import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# 1. Banco de filtros espectrais aprendidos (Gaussian Mixture "cones")
# ---------------------------------------------------------------------------
class GaussianMixtureFilterBank(nn.Module):
    """
    Cada canal de saida (ex: 3, 5, 9...) e uma soma ponderada de K gaussianas
    ao longo do eixo espectral (bio-inspirado nos cones da visao humana).
    """

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
        self.log_widths = nn.Parameter(torch.full((n_channels, n_gaussians), np.log(n_bands / (2 * n_gaussians))))
        self.raw_weights = nn.Parameter(torch.ones(n_channels, n_gaussians))

    def get_filters(self) -> torch.Tensor:
        """Retorna os filtros materializados: shape (n_channels, n_bands)"""
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
        embedding = spectra @ filters.T
        return embedding


# ---------------------------------------------------------------------------
# 2. Bloco Decoder (reconstroi o espectro a partir do embedding)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3. Sistema completo (filtro + decoder) e metrica D(s, s_hat)
# ---------------------------------------------------------------------------
class SpectralEmbeddingSystem(nn.Module):
    def __init__(self, n_bands: int, n_channels: int = 3, n_gaussians: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.filter_bank = GaussianMixtureFilterBank(n_bands, n_channels, n_gaussians)
        self.decoder = SpectralDecoder(n_channels, n_bands, hidden_dim)

    def forward(self, spectra: torch.Tensor):
        embedding = self.filter_bank(spectra)
        reconstructed = self.decoder(embedding)
        return embedding, reconstructed


def spectral_reconstruction_loss(s: torch.Tensor, s_hat: torch.Tensor, alpha: float = 1.0, beta: float = 1.0):
    """
    D(s, s_hat) = alpha * integral |s - s_hat| + beta * integral |s' - s_hat'|
    """
    term1 = (s - s_hat).abs().sum(dim=-1).mean()

    ds = s[:, 1:] - s[:, :-1]
    ds_hat = s_hat[:, 1:] - s_hat[:, :-1]
    term2 = (ds - ds_hat).abs().sum(dim=-1).mean()

    loss = alpha * term1 + beta * term2
    return loss, term1.item(), term2.item()


def filter_diversity_penalty(filters: torch.Tensor) -> torch.Tensor:
    """
    Penaliza similaridade entre os filtros dos diferentes canais, pra evitar
    que colapsem na mesma regiao do espectro (o "minimo preguicoso" que a
    loss de reconstrucao sozinha permite -- ela nao se importa se os canais
    sao redundantes entre si, so se a reconstrucao final fica boa).

    Calcula a similaridade de cosseno entre cada par de filtros (normalizados)
    e retorna a media dos pares -- quanto mais parecidos os filtros, maior a
    penalidade. Adicionar isso na loss forca os canais a se especializarem em
    regioes diferentes do espectro.

    filters: (n_channels, n_bands)
    retorna: escalar (0 = todos os filtros ortogonais entre si, 1 = todos identicos)
    """
    filters_norm = F.normalize(filters, dim=-1)  # normaliza cada filtro pra norma 1
    similarity_matrix = filters_norm @ filters_norm.T  # (n_channels, n_channels)

    n_channels = filters.shape[0]
    # pega so os pares fora da diagonal (nao compara filtro com ele mesmo)
    mask = ~torch.eye(n_channels, dtype=torch.bool, device=filters.device)
    pairwise_similarities = similarity_matrix[mask]

    return pairwise_similarities.mean()


# ---------------------------------------------------------------------------
# 4. Loop de treinamento
# ---------------------------------------------------------------------------
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
    """
    gamma: peso da penalidade de diversidade entre filtros (evita colapso de
        canais na mesma regiao espectral). 0 desativa; valores tipicos 1-10 --
        se os filtros ainda colapsarem, aumente; se ficarem "forcados" demais
        a se separar prejudicando a reconstrucao, diminua.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] treinando no device: {device}")

    n_bands = spectra_data.shape[1]
    X = torch.from_numpy(spectra_data.astype(np.float32))

    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SpectralEmbeddingSystem(n_bands, n_channels, n_gaussians).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        epoch_loss, epoch_t1, epoch_t2, epoch_div, n_batches = 0.0, 0.0, 0.0, 0.0, 0
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


# ---------------------------------------------------------------------------
# 5. Aplicar embedding treinado num cubo HSI inteiro -> pseudo-RGB pro YOLO
# ---------------------------------------------------------------------------
@torch.no_grad()
def cube_to_pseudo_rgb(model: SpectralEmbeddingSystem, cube: np.ndarray, device: str = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    H, W, n_bands = cube.shape
    flat = torch.from_numpy(cube.reshape(-1, n_bands).astype(np.float32)).to(device)

    embedding = model.filter_bank(flat)
    embedding = embedding.cpu().numpy().reshape(H, W, -1)

    out = np.zeros_like(embedding, dtype=np.uint8)
    for c in range(embedding.shape[-1]):
        ch = embedding[..., c]
        lo, hi = np.percentile(ch, [1, 99])
        ch_norm = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
        out[..., c] = (ch_norm * 255).astype(np.uint8)

    return out


def get_learned_filters(model: SpectralEmbeddingSystem) -> np.ndarray:
    """Retorna os filtros aprendidos (n_channels, n_bands) para inspecao/plot."""
    return model.filter_bank.get_filters().detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 6. Carregamento de dataset real: um .npy por imagem (reflectancia JA corrigida)
# ---------------------------------------------------------------------------
def simple_foreground_mask(cube: np.ndarray, band_idx: int = None, threshold: float = None) -> np.ndarray:
    """
    Mascara simples walnut vs. fundo preto/bandeja, baseada em intensidade media.
    Versao crua, sem tratamento de textura/bordas finas -- ver
    blur_circularity_mask para a versao recomendada.
    """
    if band_idx is not None:
        ref = cube[..., band_idx]
    else:
        ref = cube.mean(axis=-1)

    if threshold is None:
        try:
            from skimage.filters import threshold_otsu
            threshold = threshold_otsu(ref)
        except ImportError:
            threshold = np.percentile(ref, 20)

    return ref > threshold


def blur_circularity_mask(
    cube: np.ndarray,
    band_idx: int = None,
    threshold: float = None,
    blur_sigma: float = 4.0,
    mask_threshold: float = 0.5,
    opening_radius: int = 5,
    min_area_ratio: float = 0.002,
    min_circularity: float = 0.5,
    exclude_right_cols: int = 0,
    return_intermediate: bool = False,
    debug: bool = False,
):
    """
    Mascara de foreground: threshold (Otsu) -> blur da mascara binaria ->
    re-binarizacao -> abertura -> filtro por circularidade.

    Por que suavizar a MASCARA (pos-Otsu) e nao a intensidade (pre-Otsu):
      O Otsu roda direto na intensidade crua, sem nenhuma alteracao -- assim
      o limiar de corte fica exatamente como o Otsu decidiria originalmente,
      sem o blur mudando os valores que ele enxerga. So DEPOIS de gerar a
      mascara binaria (0/1) e que aplicamos o blur nela: isso suaviza o
      contorno serrilhado (causado pela textura da casca virar buracos na
      mascara), sem nunca afetar a decisao de threshold em si. Depois do
      blur, a mascara volta a ser 0/1 re-binarizando em mask_threshold.

    Args:
        blur_sigma: desvio padrao do Gaussian blur aplicado na MASCARA
            binaria (nao na intensidade). Deve ser grande o suficiente pra
            apagar buracos de textura, mas pequeno o suficiente pra nao
            fundir nozes vizinhas. Aumente se ainda sobrar contorno
            serrilhado; diminua se nozes proximas comecarem a "grudar".
        mask_threshold: limiar (0 a 1) pra re-binarizar a mascara apos o
            blur. 0.5 = padrao (mantem o formato medio do blob). Valores
            menores fazem o blob "crescer" um pouco, maiores fazem "encolher".
        opening_radius: raio da abertura morfologica, remove bordas de
            bandeja e listras finas que sobrarem apos o blur.
        min_area_ratio: area minima de um componente (fracao do total de
            pixels da imagem) pra ser considerado noz.
        min_circularity: circularidade minima (4*pi*area/perimetro^2) pra
            aceitar o componente. 1.0 = circulo perfeito. Ajuste pra cima se
            ainda sobrar listra/retangulo, pra baixo se estiver cortando
            nozes ovais.
        return_intermediate: se True, retorna tambem a mascara "crua" (mask1,
            direto do Otsu, antes de qualquer limpeza), alem da mascara final
            (mask2, depois do blur+abertura+circularidade). Util pra
            visualizar o antes/depois lado a lado.
        debug: se True, imprime area e circularidade de CADA componente
            conectado encontrado (antes do filtro final), indicando se cada
            um passaria no criterio de area e de circularidade. Use isso pra
            calibrar min_area_ratio e min_circularity com base nos valores
            reais dos seus dados, em vez de tentativa e erro.
        exclude_right_cols: numero de colunas (pixels) a IGNORAR na borda
            direita da imagem antes de qualquer coisa (threshold incluso).
            Use isso se seu setup tem um alvo de calibracao/referencia de cor
            fixo sempre na mesma regiao do campo de visao -- e mais robusto
            que tentar filtrar essas listras so por circularidade, porque a
            posicao delas e fixa e conhecida (nao muda entre capturas).

    cube: (H, W, n_bands)
    retorna: mask final booleana (H, W), ou (mask_final, mask_otsu_bruta) se
        return_intermediate=True
    """
    from skimage import morphology, measure
    from scipy.ndimage import gaussian_filter

    if band_idx is not None:
        ref = cube[..., band_idx]
    else:
        ref = cube.mean(axis=-1)

    # corta a regiao fixa de calibracao/referencia ANTES de qualquer coisa,
    # inclusive antes do Otsu -- assim esses pixels nem entram na estatistica
    # que decide o threshold
    if exclude_right_cols > 0:
        ref = ref.copy()
        ref[:, -exclude_right_cols:] = ref.min()

    # 1. Otsu na intensidade CRUA (sem blur), igual ao simple_foreground_mask
    #    -> essa e a "mask1": mascara bruta, direto do threshold
    if threshold is None:
        try:
            from skimage.filters import threshold_otsu
            threshold = threshold_otsu(ref)
        except ImportError:
            threshold = np.percentile(ref, 20)

    mask1_otsu = ref > threshold

    # 2. suaviza a MASCARA BINARIA (nao a intensidade) e re-binariza
    mask_blurred = gaussian_filter(mask1_otsu.astype(np.float32), sigma=blur_sigma)
    mask = mask_blurred > mask_threshold

    # remove estruturas finas residuais (bordas de bandeja, listras)
    mask = morphology.opening(mask, morphology.disk(opening_radius))

    # filtra componentes conectados por area e circularidade
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    total_pixels = mask.shape[0] * mask.shape[1]
    min_area = min_area_ratio * total_pixels

    if debug:
        print(f"[debug] total_pixels={total_pixels}  min_area={min_area:.0f}  min_circularity={min_circularity}")
        for p in sorted(props, key=lambda p: -p.area):
            circularity = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
            area_ok = "OK " if p.area >= min_area else "AREA_BAIXA"
            circ_ok = "OK " if circularity >= min_circularity else "CIRC_BAIXA"
            print(f"[debug]   componente area={p.area:8.0f}  circularidade={circularity:.3f}  -> area:{area_ok}  circ:{circ_ok}")

    clean_mask = np.zeros_like(mask)
    for p in props:
        if p.area < min_area:
            continue
        circularity = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
        if circularity >= min_circularity:
            clean_mask[labeled == p.label] = True

    # "mask2": mascara final, depois de toda a limpeza
    mask2_final = clean_mask

    if return_intermediate:
        return mask2_final, mask1_otsu
    return mask2_final


def save_mask_visualization(
    cube: np.ndarray,
    mask: np.ndarray,
    img_name: str,
    masks_output_folder: str,
    mask_raw: np.ndarray = None,
):
    """
    Salva imagens PNG pra inspecao visual da mascara de foreground:
      - {img_name}_mask1_otsu.png  -> mascara bruta do Otsu (mask1), se
                                       mask_raw for informado
      - {img_name}_mask2_final.png -> mascara final, depois de toda a
                                       limpeza (blur+abertura+circularidade)
      - {img_name}_overlay.png     -> intensidade media do cubo com a
                                       mascara final destacada em vermelho
    """
    import os
    from PIL import Image

    os.makedirs(masks_output_folder, exist_ok=True)

    if mask_raw is not None:
        mask1_img = (mask_raw.astype(np.uint8) * 255)
        Image.fromarray(mask1_img).save(os.path.join(masks_output_folder, f"{img_name}_mask1_otsu.png"))

    mask2_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask2_img).save(os.path.join(masks_output_folder, f"{img_name}_mask2_final.png"))

    gray = cube.mean(axis=-1)
    lo, hi = np.percentile(gray, [1, 99])
    gray_norm = np.clip((gray - lo) / (hi - lo + 1e-8), 0, 1)
    gray_rgb = np.stack([gray_norm] * 3, axis=-1)

    overlay = gray_rgb.copy()
    overlay[mask, 0] = 0.6 * overlay[mask, 0] + 0.4 * 1.0
    overlay[mask, 1] = 0.6 * overlay[mask, 1]
    overlay[mask, 2] = 0.6 * overlay[mask, 2]

    overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(overlay_uint8).save(os.path.join(masks_output_folder, f"{img_name}_overlay.png"))


def load_dataset_from_npy_folder(
    folder_path: str,
    mask_fn=blur_circularity_mask,
    max_pixels_per_image: int = 50000,
    masks_output_folder: str = None,
):
    """
    Carrega todas as imagens .npy de uma pasta (uma por captura de walnut,
    ja corrigidas por white/dark), aplica mascara de foreground em cada uma,
    e empilha os espectros de foreground de todas as imagens num unico array.
    """
    import glob
    import os

    npy_files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"Nenhum .npy encontrado em {folder_path}")

    all_spectra = []
    rng = np.random.default_rng(0)

    for fpath in npy_files:
        img_name = os.path.splitext(os.path.basename(fpath))[0]
        cube = np.load(fpath)

        if masks_output_folder is not None:
            # tenta pedir a mascara intermediaria (mask1_otsu) tambem, se a
            # mask_fn suportar (blur_circularity_mask suporta; funcoes mais
            # simples como simple_foreground_mask nao tem esse parametro)
            try:
                mask, mask_raw = mask_fn(cube, return_intermediate=True)
            except TypeError:
                mask = mask_fn(cube)
                mask_raw = None
            save_mask_visualization(cube, mask, img_name, masks_output_folder, mask_raw=mask_raw)
        else:
            mask = mask_fn(cube)

        spectra_fg = cube[mask]

        if spectra_fg.shape[0] > max_pixels_per_image:
            idx = rng.choice(spectra_fg.shape[0], size=max_pixels_per_image, replace=False)
            spectra_fg = spectra_fg[idx]

        all_spectra.append(spectra_fg)
        print(f"[dataset] {os.path.basename(fpath)}: {spectra_fg.shape[0]} pixels de foreground")

    if masks_output_folder is not None:
        print(f"[info] mascaras salvas em {masks_output_folder}")

    spectra_all = np.concatenate(all_spectra, axis=0).astype(np.float32)
    print(f"[dataset] total: {spectra_all.shape[0]} pixels, {spectra_all.shape[1]} bandas")
    return spectra_all


# =============================================================================
# CONFIGURACAO - EDITE AQUI OS CAMINHOS E HIPERPARAMETROS
# =============================================================================
INPUT_FOLDER = r"E:\LDC\img_preprocessed\FX17_defeitos\img_corr"
OUTPUT_MODEL_PATH = r"E:\LDC\img_preprocessed\FX17_defeitos\embedding\spectral_embedding_walnut.pt"
OUTPUT_FILTERS_PATH = r"E:\LDC\img_preprocessed\FX17_defeitos\filters\learned_filters.npy"

OUTPUT_PSEUDO_RGB_FOLDER = r"E:\LDC\img_preprocessed\FX17_defeitos\pseudo_rgb"     # ou None

MASKS_OUTPUT_FOLDER = r"E:\LDC\img_preprocessed\FX17_defeitos\masks"   # ou None

# hiperparametros da mascara (threshold Otsu -> blur na mascara -> circularidade)
BLUR_SIGMA = 3.0             # forca do blur aplicado na MASCARA (pos-Otsu) pra apagar contorno serrilhado
MASK_THRESHOLD = 0.5         # limiar de re-binarizacao da mascara apos o blur
OPENING_RADIUS = 20          # PRECISA cortar as pontes de bandeja que conectam nozes vizinhas (senao viram um blob so)
MIN_AREA_RATIO = 0.005       # area minima (fracao da imagem) pra contar como noz
MIN_CIRCULARITY = 0.45       # 1.0 = circulo perfeito; listras/bordas ficam bem abaixo
EXCLUDE_RIGHT_COLS = 130     # corta a faixa de calibracao fixa na borda direita (ajuste conforme sua imagem)
MASK_DEBUG = True            # imprime area/circularidade de cada componente (ajuda a calibrar)

# hiperparametros do embedding / treino
N_CHANNELS = 3
N_GAUSSIANS = 3           # obrigatorio manter em 3 -- a diversidade entre canais fica a cargo do GAMMA
ALPHA = 1.0
BETA = 1.0
GAMMA = 20.0             # peso da penalidade de diversidade entre canais (evita colapso na mesma regiao)
EPOCHS = 200
BATCH_SIZE = 4096
LR = 1e-3
MAX_PIXELS_PER_IMAGE = 50000
# =============================================================================


if __name__ == "__main__":
    import os
    from functools import partial

    mask_fn = partial(
        blur_circularity_mask,
        blur_sigma=BLUR_SIGMA,
        mask_threshold=MASK_THRESHOLD,
        opening_radius=OPENING_RADIUS,
        min_area_ratio=MIN_AREA_RATIO,
        min_circularity=MIN_CIRCULARITY,
        exclude_right_cols=EXCLUDE_RIGHT_COLS,
        debug=MASK_DEBUG,
    )

    if os.path.isdir(INPUT_FOLDER):
        spectra_all = load_dataset_from_npy_folder(
            INPUT_FOLDER,
            mask_fn=mask_fn,
            max_pixels_per_image=MAX_PIXELS_PER_IMAGE,
            masks_output_folder=MASKS_OUTPUT_FOLDER,
        )
    else:
        print(f"[aviso] INPUT_FOLDER '{INPUT_FOLDER}' nao encontrada - rodando smoke-test com dados sinteticos.")
        print("        edite a variavel INPUT_FOLDER no topo do script pra apontar pro seu dataset real.")
        n_bands = 224
        n_pixels = 20000
        rng = np.random.default_rng(42)
        spectra_all = rng.random((n_pixels, n_bands)).astype(np.float32).cumsum(axis=1)
        spectra_all /= spectra_all.max()

    model = train_embedding_system(
        spectra_all,
        n_channels=N_CHANNELS,
        n_gaussians=N_GAUSSIANS,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        verbose_every=20,
    )

    filters = get_learned_filters(model)
    print("Shape dos filtros aprendidos:", filters.shape)

    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"[info] modelo salvo em {OUTPUT_MODEL_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_FILTERS_PATH), exist_ok=True)
    np.save(OUTPUT_FILTERS_PATH, filters)
    print(f"[info] filtros aprendidos salvos em {OUTPUT_FILTERS_PATH}")

    if OUTPUT_PSEUDO_RGB_FOLDER is not None and os.path.isdir(INPUT_FOLDER):
        import glob

        os.makedirs(OUTPUT_PSEUDO_RGB_FOLDER, exist_ok=True)
        npy_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.npy")))

        for fpath in npy_files:
            cube = np.load(fpath)
            pseudo_rgb = cube_to_pseudo_rgb(model, cube)
            out_name = os.path.splitext(os.path.basename(fpath))[0] + "_pseudo_rgb.npy"
            out_path = os.path.join(OUTPUT_PSEUDO_RGB_FOLDER, out_name)
            np.save(out_path, pseudo_rgb)
            print(f"[info] pseudo-RGB salva em {out_path}")
