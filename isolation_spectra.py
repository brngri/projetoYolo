import os
import numpy as np
import cv2
from tqdm import tqdm
from spectral import envi
from sklearn.cluster import KMeans
import shutil

# =========================
# Configurações
# =========================
ROOT_DIR = r"C:\userMarcos\ProjetoSuzano\download\Aluminio_original"
DEST_DIR = r"C:\userMarcos\ProjetoSuzano\processado_classificado"
K_CLUSTERS = 2
K_N_INIT = 10
OVERWRITE = False

# =========================
# Utilidades
# =========================
def is_capture(path: str) -> bool:
    return os.path.basename(path).lower() == "capture"

def list_samples(capture_dir: str):
    """Lista arquivos .hdr de amostra (exclui DARKREF_...)."""
    for f in sorted(os.listdir(capture_dir)):
        fl = f.lower()
        if fl.endswith(".hdr") and not fl.startswith("darkref_"):
            yield os.path.join(capture_dir, f)

def load_envi_cube(hdr_path: str) -> np.ndarray:
    """Carrega cubo ENVI (H, W, B) como float32 (RAW, sem calibração)."""
    img = envi.open(hdr_path)
    arr = np.array(img.load()).astype(np.float32, copy=False)
    return arr  # (H, W, B)

def segmentar_folha_esteira(img: np.ndarray, n_clusters: int = K_CLUSTERS):
    """
    Segmenta no espaço espectral (B bandas).
    'Esteira' = cluster com menor média de intensidade (mais escuro).
    'Folha' = todos os outros clusters combinados.
    Retorna duas máscaras binárias uint8 (0/255), shape (H, W).
    """
    h, w, b = img.shape
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=K_N_INIT)
    labels = km.fit_predict(img.reshape(-1, b)).reshape(h, w)

    # Identifica qual cluster é a folha (menor intensidade)
    medias = [float(np.mean(img[labels == i])) for i in range(n_clusters)]
    classe_folha = int(np.argmin(medias))
    
    # Cria máscaras
    mask_folha = (labels != classe_folha).astype(np.uint8) * 255
    mask_esteira = (labels == classe_folha).astype(np.uint8) * 255
    
    return mask_folha, mask_esteira

def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask == 255)
    if ys.size == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return int(y0), int(y1), int(x0), int(x1)

def criar_estrutura_destino():
    """Cria a estrutura de pastas de destino organizada"""
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # Pastas principais para cada classe
    for classe in ['folha', 'esteira']:
        classe_dir = os.path.join(DEST_DIR, classe)
        os.makedirs(classe_dir, exist_ok=True)
        
        # Subpastas para cada tipo de arquivo
        os.makedirs(os.path.join(classe_dir, 'mascaras'), exist_ok=True)
        os.makedirs(os.path.join(classe_dir, 'blocos_espectrais'), exist_ok=True)
        os.makedirs(os.path.join(classe_dir, 'originais'), exist_ok=True)

# =========================
# Pipeline por amostra
# =========================
def process_sample(hdr_path: str):
    cap_dir = os.path.dirname(hdr_path)
    base = os.path.splitext(os.path.basename(hdr_path))[0]
    
    # 1) Carrega cubo RAW
    cube = load_envi_cube(hdr_path)  # (H, W, B)
    if cube.ndim != 3:
        print(f"[SKIP] {hdr_path}: shape inesperado {cube.shape}")
        return
    H, W, B = cube.shape

    # 2) Segmentação folha vs esteira
    mask_folha, mask_esteira = segmentar_folha_esteira(cube, n_clusters=K_CLUSTERS)

    # Processa FOLHA
    processar_classe(cube, mask_folha, base, 'folha', hdr_path)
    
    # Processa ESTEIRA  
    processar_classe(cube, mask_esteira, base, 'esteira', hdr_path)

def processar_classe(cube: np.ndarray, mask: np.ndarray, base: str, classe: str, hdr_path: str):
    """Processa e salva os arquivos para uma classe específica"""
    
    # Caminhos de destino organizados
    classe_dir = os.path.join(DEST_DIR, classe)
    
    mask_png = os.path.join(classe_dir, 'mascaras', f"{base}_{classe}_mask.png")
    block_npy = os.path.join(classe_dir, 'blocos_espectrais', f"{base}_{classe}_block.npy")
    hdr_dest = os.path.join(classe_dir, 'originais', f"{base}.hdr")
    
    if not OVERWRITE and os.path.exists(mask_png) and os.path.exists(block_npy):
        print(f" - pulando {classe} (já existem): {os.path.basename(mask_png)}")
        return

    # BBox e bloco espectral
    bbox = bbox_from_mask(mask)
    if bbox is None:
        print(f" [AVISO] {base}: máscara de {classe} vazia.")
        return
        
    y0, y1, x0, x1 = bbox

    crop_cube = cube[y0:y1, x0:x1, :].copy()
    crop_mask = mask[y0:y1, x0:x1]

    # Zera o fundo dentro do recorte
    crop_cube = np.where((crop_mask > 0)[:, :, None], crop_cube, 0.0)

    # Salva os arquivos
    cv2.imwrite(mask_png, mask)
    np.save(block_npy, crop_cube)
    
    # Copia o arquivo .hdr original apenas na primeira classe processada (folha)
    if classe == 'folha':
        try:
            shutil.copy2(hdr_path, hdr_dest)
            img_orig = hdr_path.replace('.hdr', '.img')
            img_dest = hdr_dest.replace('.hdr', '.img')
            if os.path.exists(img_orig):
                shutil.copy2(img_orig, img_dest)
        except Exception as e:
            print(f" [AVISO] Não foi possível copiar arquivos originais: {e}")

    print(f" - {base}: {classe} → {crop_cube.shape[0]}x{crop_cube.shape[1]}x{crop_cube.shape[2]}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Cria estrutura de pastas
    criar_estrutura_destino()
    print(f"[INFO] Pasta de destino criada: {DEST_DIR}")
    
    captures = []
    for dirpath, _, _ in os.walk(ROOT_DIR):
        if is_capture(dirpath):
            captures.append(dirpath)

    if not captures:
        print(f"[WARN] Nenhuma pasta 'capture' encontrada em {ROOT_DIR}")
        raise SystemExit(0)

    print(f"[INFO] Encontradas {len(captures)} pastas 'capture'.")
    
    # Contadores para estatísticas
    contadores = {'folha': 0, 'esteira': 0}
    
    for cap_dir in captures:
        hdrs = list(list_samples(cap_dir))
        if not hdrs:
            print(f"[INFO] Sem .hdr (amostras) em: {cap_dir}")
            continue

        print(f"[INFO] {cap_dir} — {len(hdrs)} amostras")
        for hdr_path in tqdm(hdrs, desc=f"Processando {cap_dir}", leave=False):
            try:
                process_sample(hdr_path)
                # Conta uma amostra processada (cada amostra gera ambos folha e esteira)
                contadores['folha'] += 1
                contadores['esteira'] += 1
            except Exception as e:
                print(f"[ERRO] {hdr_path}: {e}")

    # Mostra estatísticas finais
    print("\n[INFO] Processamento concluído!")
    print(f"[INFO] Pasta de destino: {DEST_DIR}")
    print("[INFO] Estrutura criada:")
    print("  - folha/")
    print("    ├── mascaras/")
    print("    ├── blocos_espectrais/") 
    print("    └── originais/")
    print("  - esteira/")
    print("    ├── mascaras/")
    print("    ├── blocos_espectrais/")
    print("    └── originais/")
    print(f"[INFO] Amostras processadas: {contadores['folha']} (cada amostra gera dados para folha e esteira)")



# import os
# import numpy as np
# import cv2
# from tqdm import tqdm
# from spectral import envi
# from sklearn.cluster import KMeans

# # =========================
# # Configurações
# # =========================
# ROOT_DIR   = r"C:\userMarcos\ProjetoSuzano\download\Aluminio_original"  # raiz com várias subpastas que contêm ...\capture\
# K_CLUSTERS = 2
# K_N_INIT   = 10        # compatível com versões antigas do sklearn
# OVERWRITE  = False     # False => não reescreve se *_mask.png e *_graos_block.npy já existirem

# # =========================
# # Utilidades
# # =========================
# def is_capture(path: str) -> bool:
#     return os.path.basename(path).lower() == "capture"

# def list_samples(capture_dir: str):
#     """Lista arquivos .hdr de amostra (exclui DARKREF_...)."""
#     for f in sorted(os.listdir(capture_dir)):
#         fl = f.lower()
#         if fl.endswith(".hdr") and not fl.startswith("darkref_"):
#             yield os.path.join(capture_dir, f)

# def load_envi_cube(hdr_path: str) -> np.ndarray:
#     """Carrega cubo ENVI (H, W, B) como float32 (RAW, sem calibração)."""
#     img = envi.open(hdr_path)
#     arr = np.array(img.load()).astype(np.float32, copy=False)
#     return arr  # (H, W, B)

# def aplicar_kmeans(img: np.ndarray, n_clusters: int = K_CLUSTERS) -> np.ndarray:
#     """
#     Segmenta no espaço espectral (B bandas).
#     'Grãos' = cluster com menor média de intensidade (mais escuro).
#     Retorna máscara binária uint8 (0/255), shape (H, W).
#     """
#     h, w, b = img.shape
#     km = KMeans(n_clusters=n_clusters, random_state=42, n_init=K_N_INIT)
#     labels = km.fit_predict(img.reshape(-1, b)).reshape(h, w)

#     medias = [float(np.mean(img[labels == i])) for i in range(n_clusters)]
#     classe_graos = int(np.argmin(medias))
#     mask = (labels == classe_graos).astype(np.uint8) * 255
#     return mask

# def bbox_from_mask(mask: np.ndarray):
#     ys, xs = np.where(mask == 255)
#     if ys.size == 0:
#         return None
#     y0, y1 = ys.min(), ys.max() + 1
#     x0, x1 = xs.min(), xs.max() + 1
#     return int(y0), int(y1), int(x0), int(x1)

# # =========================
# # Pipeline por amostra
# # =========================
# def process_sample(hdr_path: str):
#     cap_dir  = os.path.dirname(hdr_path)
#     base     = os.path.splitext(os.path.basename(hdr_path))[0]
#     mask_png = os.path.join(cap_dir, f"{base}_mask.png")
#     block_npy= os.path.join(cap_dir, f"{base}_graos_block.npy")

#     if not OVERWRITE and os.path.exists(mask_png) and os.path.exists(block_npy):
#         print(f" - pulando (já existem): {os.path.basename(mask_png)}, {os.path.basename(block_npy)}")
#         return

#     # 1) Carrega cubo RAW
#     cube = load_envi_cube(hdr_path)  # (H, W, B)
#     if cube.ndim != 3:
#         print(f"[SKIP] {hdr_path}: shape inesperado {cube.shape}")
#         return
#     H, W, B = cube.shape

#     # 2) KMeans para máscara dos grãos
#     mask = aplicar_kmeans(cube, n_clusters=K_CLUSTERS)

#     # 3) BBox e bloco espectral (recorte + fundo zerado com np.where → evita erro de broadcast)
#     bbox = bbox_from_mask(mask)
#     if bbox is None:
#         print(f"[SKIP] {hdr_path}: máscara vazia.")
#         return
#     y0, y1, x0, x1 = bbox

#     crop_cube = cube[y0:y1, x0:x1, :].copy()    # (Hc, Wc, B)
#     crop_mask = mask[y0:y1, x0:x1]              # (Hc, Wc)

#     # zera o fundo dentro do recorte (broadcast seguro)
#     crop_cube = np.where((crop_mask > 0)[:, :, None], crop_cube, 0.0)

#     # 4) Salva APENAS os dois novos artefatos na pasta original
#     cv2.imwrite(mask_png, mask)
#     np.save(block_npy, crop_cube)

#     print(f" - {base}: salvos {os.path.basename(mask_png)} e {os.path.basename(block_npy)} "
#           f"(crop: {crop_cube.shape[0]}x{crop_cube.shape[1]}x{B})")

# # =========================
# # Main
# # =========================
# if __name__ == "__main__":
#     captures = []
#     for dirpath, _, _ in os.walk(ROOT_DIR):
#         if is_capture(dirpath):
#             captures.append(dirpath)

#     if not captures:
#         print(f"[WARN] Nenhuma pasta 'capture' encontrada em {ROOT_DIR}")
#         raise SystemExit(0)

#     print(f"[INFO] Encontradas {len(captures)} pastas 'capture'.")
#     for cap_dir in captures:
#         hdrs = list(list_samples(cap_dir))
#         if not hdrs:
#             print(f"[INFO] Sem .hdr (amostras) em: {cap_dir}")
#             continue

#         print(f"[INFO] {cap_dir} — {len(hdrs)} amostras")
#         for hdr_path in tqdm(hdrs, desc=f"Processando {cap_dir}", leave=False):
#             try:
#                 process_sample(hdr_path)
#             except Exception as e:
#                 print(f"[ERRO] {hdr_path}: {e}")