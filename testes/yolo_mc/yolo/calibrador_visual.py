"""
CALIBRADOR SIMPLES E OBJETIVO
=============================
Testa a segmentação em UMA imagem escolhida, exibe a máscara final 
sobreposta e sugere ajustes de parâmetros.
Salva os parâmetros em parametros_recomendados.json dentro de um 
subdiretório 'parametros_calibrados/' na mesma pasta da imagem.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure, filters
from scipy.ndimage import gaussian_filter
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# ================= CONFIGURAÇÕES (ajuste aqui) =================
BLUR_SIGMA = 7.0          # Suavização da máscara (aumente para tampar furos)
MASK_THRESHOLD = 0.4      # Limiar pós-blur (diminua para preencher mais)
OPENING_RADIUS = 15       # Raio de abertura (para separar objetos grudados)
MIN_AREA_RATIO = 0.005    # Área mínima como fração da imagem total
MIN_CIRCULARITY = 0.35    # Circularidade mínima (1 = círculo perfeito)
EXCLUDE_RIGHT_COLS = 0    # Corte de faixa (só para .npy com calibração)
# ==============================================================

def load_image(file_path):
    """Carrega .npy ou imagem RGB, retornando um cubo (H,W,C)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.npy':
        cube = np.load(file_path)
        if cube.ndim == 2:
            cube = cube[..., np.newaxis]
        return cube
    else:
        img = Image.open(file_path).convert('RGB')
        arr = np.array(img)  # (H, W, 3)
        return arr.astype(np.float32) / 255.0

def segmentar(cube, params):
    """Aplica a pipeline de segmentação e retorna a máscara final."""
    H, W, B = cube.shape
    # Banda de referência (média)
    ref = cube.mean(axis=-1)
    if params['exclude_right_cols'] > 0:
        ref = ref.copy()
        ref[:, -params['exclude_right_cols']:] = ref.min()
    
    # Otsu
    thresh = filters.threshold_otsu(ref)
    mask_otsu = ref > thresh
    
    # Blur na máscara
    mask_blurred = gaussian_filter(mask_otsu.astype(np.float32), sigma=params['blur_sigma'])
    mask_blurred_bin = mask_blurred > params['mask_threshold']
    
    # Abertura
    mask_opened = morphology.opening(mask_blurred_bin, morphology.disk(params['opening_radius']))
    
    # Filtro por área e circularidade
    labeled = measure.label(mask_opened)
    props = measure.regionprops(labeled)
    total_pixels = H * W
    min_area = params['min_area_ratio'] * total_pixels
    
    mask_final = np.zeros_like(mask_opened, dtype=bool)
    for p in props:
        circularity = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
        if p.area >= min_area and circularity >= params['min_circularity']:
            mask_final[labeled == p.label] = True
    return mask_final, mask_otsu, thresh

def mostrar_resultado(imagem_original, mask_final, params):
    """Exibe a imagem original com a máscara sobreposta em vermelho."""
    # Normaliza a imagem para exibição
    if imagem_original.shape[-1] == 3:
        # É RGB
        img_display = imagem_original.copy()
    else:
        # É um cubo -> média
        img_display = imagem_original.mean(axis=-1)
        lo, hi = np.percentile(img_display, [1, 99])
        img_display = np.clip((img_display - lo) / (hi - lo + 1e-8), 0, 1)
        img_display = np.stack([img_display]*3, axis=-1)
    
    # Overlay
    overlay = img_display.copy()
    overlay[mask_final, 0] = 0.6 * overlay[mask_final, 0] + 0.4 * 1.0  # vermelho
    overlay[mask_final, 1] = 0.6 * overlay[mask_final, 1]
    overlay[mask_final, 2] = 0.6 * overlay[mask_final, 2]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_display)
    plt.title("Imagem original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(overlay, 0, 1))
    plt.title(f"Máscara final (pixels: {mask_final.sum()})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def sugerir_ajustes(mask_final, mask_otsu, params):
    """Analisa a máscara e sugere ajustes."""
    sugestoes = []
    area_final = mask_final.sum()
    area_otsu = mask_otsu.sum()
    if area_final == 0:
        sugestoes.append("Nenhum objeto detectado. Tente:")
        sugestoes.append("  - Aumentar 'blur_sigma' (para tampar buracos)")
        sugestoes.append("  - Diminuir 'mask_threshold' (para preencher mais)")
        sugestoes.append("  - Diminuir 'min_area_ratio' e 'min_circularity'")
    elif area_final < 0.01 * area_otsu:
        sugestoes.append("Máscara final muito pequena comparada ao Otsu. Tente:")
        sugestoes.append("  - Diminuir 'min_area_ratio'")
        sugestoes.append("  - Diminuir 'min_circularity'")
        sugestoes.append("  - Aumentar 'mask_threshold' (para encolher menos)")
    else:
        sugestoes.append("Máscara parece razoável. Para melhorar:")
        sugestoes.append("  - Se ainda houver buracos, aumente 'blur_sigma'")
        sugestoes.append("  - Se estiver pegando fundo, aumente 'min_circularity'")
    return sugestoes

def main():
    # Selecionar arquivo
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecione uma imagem (.npy, .png, .jpg) para testar",
        filetypes=[("Todos os suportados", "*.npy *.png *.jpg *.jpeg *.bmp *.tif")]
    )
    if not file_path:
        print("Nenhum arquivo selecionado.")
        return
    
    print(f"\nProcessando: {os.path.basename(file_path)}")
    
    # Carregar
    cube = load_image(file_path)
    
    # Parâmetros atuais
    params = {
        'blur_sigma': BLUR_SIGMA,
        'mask_threshold': MASK_THRESHOLD,
        'opening_radius': OPENING_RADIUS,
        'min_area_ratio': MIN_AREA_RATIO,
        'min_circularity': MIN_CIRCULARITY,
        'exclude_right_cols': EXCLUDE_RIGHT_COLS
    }
    
    # Segmentar
    mask_final, mask_otsu, thresh = segmentar(cube, params)
    
    # Exibir resultado
    mostrar_resultado(cube, mask_final, params)
    
    # Sugestões
    sugestoes = sugerir_ajustes(mask_final, mask_otsu, params)
    print("\n🔍 Sugestões de ajuste:")
    for s in sugestoes:
        print("  ", s)
    
    # Perguntar se quer salvar
    resposta = input("\nDeseja salvar estes parâmetros para uso nos scripts de treino? (s/n): ")
    if resposta.lower() == 's':
        # Define o diretório de saída: subdiretório 'parametros_calibrados' na pasta da imagem
        pasta_imagem = os.path.dirname(file_path)
        pasta_destino = os.path.join(pasta_imagem, 'parametros_calibrados')
        os.makedirs(pasta_destino, exist_ok=True)
        json_path = os.path.join(pasta_destino, 'parametros_recomendados.json')
        
        with open(json_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"✅ Parâmetros salvos em: {json_path}")
        print("   Este arquivo será lido pelo spectral_embedding_learned10.py")
    else:
        print("Parâmetros não salvos.")
    
    print("\nPara testar novos valores, edite as variáveis no topo do script e execute novamente.")

if __name__ == "__main__":
    main()






















































# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import morphology, measure, filters
# from scipy.ndimage import gaussian_filter
# import os
# import glob

# # =====================================================================
# # CONFIGURAÇÃO - EDITE AQUI
# # =====================================================================
# INPUT_FOLDER = r"E:\ProjetoDogChow\01_dados\preprocessamento\06_07_2026_correcao_radiometrica\amostras_gerais\furado"  # <--- PASTA COM TODOS OS .npy

# # PARÂMETROS QUE VOCÊ QUER TESTAR (brinque com eles aqui)
# PARAMS = {
#     "band_idx": None,              # Mantém média de todas as bandas
#     "threshold": None,             # Otsu automático
#     "blur_sigma": 7.0,             # ⬆️ AUMENTE bastante (antes era 3.0)
#     "mask_threshold": 0.4,         # ⬇️ Baixe um pouco (antes 0.5) – ajuda a "expandir" o branco
#     "opening_radius": 15,          # Mantém pequeno (limpa bordas, não mexe nos buracos)
#     "min_area_ratio": 0.005,       # Mantém (ou use o sugerido pelo script)
#     "min_circularity": 0.35,       # ⬇️ Relaxe um pouco (antes 0.45) – bordas podem ficar menos circulares com o blur forte
#     "exclude_right_cols": 130,     # Mantém o corte da faixa de calibração
# }

# # ONDE SALVAR OS DEBUGS
# OUTPUT_DEBUG_FOLDER = os.path.join(INPUT_FOLDER, "calibration_debug")
# # =====================================================================

# def process_single_image(npy_path, params, save_plot=True):
#     """Processa uma imagem, retorna as estatísticas do maior objeto e plota o debug."""
    
#     img_name = os.path.splitext(os.path.basename(npy_path))[0]
#     cube = np.load(npy_path)
#     H, W, B = cube.shape

#     # 1. Seleciona a banda de referência
#     if params["band_idx"] is not None:
#         ref = cube[..., params["band_idx"]]
#     else:
#         ref = cube.mean(axis=-1)
#     ref_original = ref.copy()

#     # 2. Corte da faixa direita
#     if params["exclude_right_cols"] > 0:
#         ref = ref.copy()
#         ref[:, -params["exclude_right_cols"]:] = ref.min()

#     # 3. Otsu
#     if params["threshold"] is None:
#         thresh = filters.threshold_otsu(ref)
#     else:
#         thresh = params["threshold"]
#     mask1_otsu = ref > thresh

#     # 4. Blur
#     mask_blurred = gaussian_filter(mask1_otsu.astype(np.float32), sigma=params["blur_sigma"])
#     mask2_blurred_bin = mask_blurred > params["mask_threshold"]

#     # 5. Abertura
#     mask3_opened = morphology.opening(mask2_blurred_bin, morphology.disk(params["opening_radius"]))

#     # 6. Filtro por Área e Circularidade (calcula para TODOS os componentes)
#     labeled = measure.label(mask3_opened)
#     props = measure.regionprops(labeled)
#     total_pixels = H * W
#     min_area = params["min_area_ratio"] * total_pixels

#     # --- Estatísticas do MAIOR OBJETO (presumivelmente o alvo) ---
#     # Ordena por área para pegar o maior
#     if props:
#         largest_prop = max(props, key=lambda p: p.area)
#         largest_area = largest_prop.area
#         largest_perimeter = largest_prop.perimeter
#         largest_circularity = 4 * np.pi * largest_area / (largest_perimeter ** 2 + 1e-8)
#         largest_label = largest_prop.label
#     else:
#         largest_area = 0
#         largest_circularity = 0
#         largest_label = -1

#     # --- Aplica o filtro final (igual ao script principal) ---
#     mask_final = np.zeros_like(mask3_opened, dtype=bool)
#     for p in props:
#         circularity = 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8)
#         if p.area >= min_area and circularity >= params["min_circularity"]:
#             mask_final[labeled == p.label] = True

#     # --- PLOTAGEM (Grid de 6 imagens) ---
#     if save_plot:
#         fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#         ax = axes.ravel()

#         # Normaliza a intensidade
#         ref_display = ref_original.copy()
#         lo, hi = np.percentile(ref_display, [1, 99])
#         ref_display = np.clip((ref_display - lo) / (hi - lo + 1e-8), 0, 1)

#         ax[0].imshow(ref_display, cmap='gray')
#         ax[0].set_title(f"1. Intensidade ({img_name})")
#         ax[0].axis('off')

#         ax[1].imshow(mask1_otsu, cmap='gray')
#         ax[1].set_title(f"2. Otsu Cru (Thr={thresh:.2f})")
#         ax[1].axis('off')

#         ax[2].imshow(mask2_blurred_bin, cmap='gray')
#         ax[2].set_title(f"3. Pós-Blur (Sigma={params['blur_sigma']})")
#         ax[2].axis('off')

#         ax[3].imshow(mask3_opened, cmap='gray')
#         ax[3].set_title(f"4. Abertura (Raio={params['opening_radius']})")
#         ax[3].axis('off')

#         ax[4].imshow(mask_final, cmap='gray')
#         ax[4].set_title(f"5. Final (Circ={params['min_circularity']})")
#         ax[4].axis('off')

#         # Overlay
#         overlay = np.stack([ref_display] * 3, axis=-1)
#         overlay[mask_final, 0] = 0.6 * overlay[mask_final, 0] + 0.4 * 1.0
#         overlay[mask_final, 1] = 0.6 * overlay[mask_final, 1]
#         overlay[mask_final, 2] = 0.6 * overlay[mask_final, 2]
#         ax[5].imshow(np.clip(overlay, 0, 1))
#         ax[5].set_title("6. Sobreposição (Objeto em Vermelho)")
#         ax[5].axis('off')

#         plt.tight_layout()
#         os.makedirs(OUTPUT_DEBUG_FOLDER, exist_ok=True)
#         save_path = os.path.join(OUTPUT_DEBUG_FOLDER, f"{img_name}_debug.png")
#         plt.savefig(save_path, dpi=150)
#         plt.close(fig)  # Fecha a figura para não entupir a memória

#     # Retorna os dados do maior objeto
#     return {
#         "file": img_name,
#         "largest_area": largest_area,
#         "largest_circularity": largest_circularity,
#         "total_pixels": total_pixels,
#         "mask_pixels_final": np.sum(mask_final),
#         "num_components": len(props)
#     }

# def calibrate_multiple_images(folder_path, params):
#     """Varre todos os .npy da pasta e processa um por um."""
    
#     npy_files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
#     if not npy_files:
#         print(f"ERRO: Nenhum arquivo .npy encontrado em {folder_path}")
#         return

#     print(f"Encontrados {len(npy_files)} arquivos .npy. Processando...")
#     print("="*80)
    
#     results = []
#     for fpath in npy_files:
#         print(f"Processando: {os.path.basename(fpath)} ...")
#         stats = process_single_image(fpath, params, save_plot=True)
#         results.append(stats)
#         print(f"  -> Maior objeto: Área={stats['largest_area']:.0f} px, Circularidade={stats['largest_circularity']:.3f}")

#     # --- Exibe o RESUMO FINAL para você calibrar os parâmetros ---
#     print("\n" + "="*80)
#     print("RESUMO GERAL PARA CALIBRAÇÃO DOS PARÂMETROS")
#     print("="*80)
    
#     areas = [r["largest_area"] for r in results if r["largest_area"] > 0]
#     circs = [r["largest_circularity"] for r in results if r["largest_circularity"] > 0]
    
#     if areas and circs:
#         min_area_found = min(areas)
#         max_area_found = max(areas)
#         min_circ_found = min(circs)
#         max_circ_found = max(circs)
        
#         # Pega o total de pixels da primeira imagem (assume que todas têm o mesmo tamanho)
#         total_px = results[0]["total_pixels"]
        
#         print(f"Tamanho da imagem: {total_px} pixels")
#         print("-"*80)
#         print("📊 Estatísticas do MAIOR OBJETO em cada imagem:")
#         print(f"  - Área mínima encontrada: {min_area_found:.0f} px")
#         print(f"  - Área máxima encontrada: {max_area_found:.0f} px")
#         print(f"  - Circularidade mínima encontrada: {min_circ_found:.3f}")
#         print(f"  - Circularidade máxima encontrada: {max_circ_found:.3f}")
#         print("-"*80)
        
#         # SUGESTÃO AUTOMÁTICA de parâmetros seguros
#         suggested_min_area = int(min_area_found * 0.7)  # 70% da menor área (um pouco de folga)
#         suggested_ratio = suggested_min_area / total_px
#         suggested_circ = max(0.1, min_circ_found * 0.85)  # 85% da menor circularidade
        
#         print("🎯 SUGESTÃO DE PARÂMETROS (copie para o script principal):")
#         print(f"  min_area_ratio = {suggested_ratio:.6f}   # (equivale a {suggested_min_area} pixels)")
#         print(f"  min_circularity = {suggested_circ:.3f}")
#         print("="*80)
#         print(f"✅ Debug de todas as imagens salvo em: {OUTPUT_DEBUG_FOLDER}")
#     else:
#         print("⚠️ Nenhum objeto foi detectado em nenhuma imagem. Verifique o threshold e o corte.")

# if __name__ == "__main__":
#     if not os.path.exists(INPUT_FOLDER):
#         print(f"ERRO: Pasta não encontrada em {INPUT_FOLDER}")
#         print("Por favor, edite a variável INPUT_FOLDER no topo do script.")
#     else:
#         calibrate_multiple_images(INPUT_FOLDER, PARAMS)