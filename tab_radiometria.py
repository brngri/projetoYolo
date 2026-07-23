import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import cv2
import json
from datetime import datetime
from spectral import envi
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter, uniform_filter

import spectral_utils as su
import isolation_spectra as kis

# =========================
# CONFIGURAÇÕES E CONSTANTES
# =========================
START_ROOT_DIR = r"C:\COLOQUE\A_PASTA_RAIZ_COM_AS_CAPTURES"
START_OUT_DIR  = r"C:\COLOQUE\PASTA_DE_SAIDA_PARA_NPY_E_CSV"
DEFAULT_FILENAME = "resumo_medias"

THRESHOLD_REFLECTANCE_DEFAULT = 0.015
DBSCAN_EPS_DEFAULT = "0.01"
DBSCAN_MIN_SAMPLES_DEFAULT = "5"
K_N_INIT = 10

# =========================
# FUNÇÕES AUXILIARES
# =========================

def _is_capture_dir(path: str) -> bool:
    return os.path.basename(path).lower() == "capture"

def _find_all_capture_dirs(root_dir: str):
    capture_dirs = []
    for dirpath, _, _ in os.walk(root_dir):
        if _is_capture_dir(dirpath):
            capture_dirs.append(dirpath)
    return capture_dirs

def _list_hdrs_amostra(capture_dir: str):
    for f in sorted(os.listdir(capture_dir)):
        if f.lower().endswith(".hdr"):
            yield os.path.join(capture_dir, f)

def _classify_type(hdr_name: str):
    fname = os.path.basename(hdr_name).lower()
    if fname.startswith("darkref_"):
        return "dark"
    if fname.startswith("whiteref_") or fname.startswith("r99_") or fname.startswith("r99-"):
        return "white"
    return "sample"

def _load_and_correct(img_hdr_path: str, dark_hdr_path: str, white_hdr_path: str) -> np.ndarray:
    return su.open_img_normalize(
        img_path=img_hdr_path,
        dark_path=dark_hdr_path,
        white_path=white_hdr_path
    ).astype(np.float32, copy=False)

def _processar_mascara_completa(cube: np.ndarray, mask_bin: np.ndarray):
    """Processa a máscara na imagem completa sem recortar."""
    if mask_bin is None or np.sum(mask_bin > 0) == 0:
        return None, None

    # Aplica a máscara no cubo completo
    masked_cube = np.where((mask_bin > 0)[:, :, None], cube, 0.0)
    return masked_cube.astype(np.float32, copy=False), mask_bin

def _calc_media_espectral_completa(masked_cube: np.ndarray, mask_bin: np.ndarray):
    """Calcula média espectral usando a imagem completa com máscara aplicada."""
    if masked_cube is None or mask_bin is None:
        return None, 0

    valid_pixels = masked_cube[mask_bin > 0, :]
    
    if valid_pixels.size == 0:
        return None, 0

    keep_mask = np.any(valid_pixels != 0.0, axis=1)
    valid_pixels = valid_pixels[keep_mask]
    
    num_pixels = valid_pixels.shape[0]

    if num_pixels == 0:
        return None, 0

    mean_spec = np.mean(valid_pixels, axis=0)
    return mean_spec, num_pixels

def _save_spectral_json(spectral_data: dict, file_path: str):
    """Salva dados espectrais em formato JSON."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(spectral_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Erro ao salvar JSON {file_path}: {e}")
        return False

def _save_mask_png(mask: np.ndarray, file_path: str):
    """Salva máscara binária como PNG (branco=1, preto=0)"""
    try:
        # Converte para imagem 8-bit (0-255)
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(file_path, mask_uint8)
        return True
    except Exception as e:
        print(f"Erro ao salvar PNG {file_path}: {e}")
        return False

def _save_rgb_png(cube: np.ndarray, file_path: str, rgb_bands: tuple = (30, 20, 10)):
    """Salva imagem RGB a partir do cubo hiperespectral"""
    try:
        # Seleciona bandas para RGB e normaliza
        r = cube[:, :, rgb_bands[0]]
        g = cube[:, :, rgb_bands[1]] 
        b = cube[:, :, rgb_bands[2]]
        
        # Normaliza para 0-255
        rgb_img = np.stack([r, g, b], axis=2)
        rgb_img = (rgb_img / np.max(rgb_img) * 255).astype(np.uint8)
        
        cv2.imwrite(file_path, rgb_img)
        return True
    except Exception as e:
        print(f"Erro ao salvar RGB PNG {file_path}: {e}")
        return False
    
# =========================
# CLASSE DA ABA
# =========================
class RadiometricTab:
    def __init__(self, parent):
        self.parent = parent
        self.log_queue = queue.Queue()
        self.processing = False  # Variável de controle para parar o processamento
        
        # Variáveis de Configuração
        self.segmentation_enabled_var = tk.BooleanVar(value=True)  # Controle principal
        self.k_clusters_var = tk.StringVar(value="2")
        self.threshold_var = tk.StringVar(value=str(THRESHOLD_REFLECTANCE_DEFAULT))
        self.segmentation_method_var = tk.StringVar(value="K-Means")
        self.dbscan_eps_var = tk.StringVar(value=DBSCAN_EPS_DEFAULT)
        self.dbscan_min_samples_var = tk.StringVar(value=DBSCAN_MIN_SAMPLES_DEFAULT)
        self.max_clusters_var = tk.StringVar(value="5")
        self.class_names_vars = []
        
        # Variáveis de Blur
        self.aplicar_blur_var = tk.BooleanVar(value=False)
        self.tipo_blur_var = tk.StringVar(value="uniforme")
        self.tamanho_blur_var = tk.StringVar(value="3")
        self.sigma_blur_var = tk.StringVar(value="1.0")
        
        # Frame geral
        outer = ttk.Frame(parent, padding=10)
        outer.pack(fill="both", expand=True)

        # ====== PARTE SUPERIOR: seleção de pastas ======
        paths_frame = ttk.LabelFrame(outer, text="Configuração de Pastas")
        paths_frame.pack(fill="x", pady=5)

        # Configurações de pastas
        paths_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(paths_frame, text="Pasta raiz (contém 'capture'):").grid(row=0, column=0, sticky="w")
        self.entry_root = ttk.Entry(paths_frame, width=80)
        self.entry_root.grid(row=0, column=1, padx=5, pady=2, sticky="we")
        self.entry_root.insert(0, START_ROOT_DIR)
        btn_root = ttk.Button(paths_frame, text="Browse", command=self._browse_root)
        btn_root.grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(paths_frame, text="Pasta de saída (.npy / .json / excel):").grid(row=1, column=0, sticky="w")
        self.entry_out = ttk.Entry(paths_frame, width=80)
        self.entry_out.grid(row=1, column=1, padx=5, pady=2, sticky="we")
        self.entry_out.insert(0, START_OUT_DIR)
        btn_out = ttk.Button(paths_frame, text="Browse", command=self._browse_out)
        btn_out.grid(row=1, column=2, padx=5, pady=2)

        ttk.Label(paths_frame, text="Nome do arquivo final:").grid(row=2, column=0, sticky="w")
        self.entry_filename = ttk.Entry(paths_frame, width=30)
        self.entry_filename.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self.entry_filename.insert(0, "resumo_medias")

        self.format_var = tk.StringVar(value=".xlsx")
        format_combo = ttk.Combobox(paths_frame, textvariable=self.format_var, 
                                    values=[".xlsx", ".csv"], width=8, state="readonly")
        format_combo.grid(row=2, column=1, padx=(200, 5), pady=2, sticky="w")
        
        # ====== Seção de Calibração e Segmentação ======
        config_frame = ttk.LabelFrame(outer, text="Configuração de Processamento")
        config_frame.pack(fill="x", pady=5)
        config_frame.grid_columnconfigure(1, weight=1)

        # LINHA 0: Caminho do DARK (Override) - INDEPENDENTE
        ttk.Label(config_frame, text="Dir. Dark (Opcional - Override):").grid(row=0, column=0, sticky="w")
        self.entry_dark_dir = ttk.Entry(config_frame, width=80)
        self.entry_dark_dir.grid(row=0, column=1, padx=5, pady=2, sticky="we")
        btn_dark = ttk.Button(config_frame, text="Browse", command=self._browse_dark)
        btn_dark.grid(row=0, column=2, padx=5, pady=2)
        
        # LINHA 1: Caminho do WHITE (Override) - INDEPENDENTE
        ttk.Label(config_frame, text="Dir. White (Opcional - Override):").grid(row=1, column=0, sticky="w")
        self.entry_white_dir = ttk.Entry(config_frame, width=80)
        self.entry_white_dir.grid(row=1, column=1, padx=5, pady=2, sticky="we")
        btn_white = ttk.Button(config_frame, text="Browse", command=self._browse_white)
        btn_white.grid(row=1, column=2, padx=5, pady=2)
        
        # LINHA 2: Checkbox principal para habilitar/desabilitar segmentação
        segmentation_check_frame = ttk.Frame(config_frame)
        segmentation_check_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=5)
        
        self.segmentation_check = ttk.Checkbutton(
            segmentation_check_frame, 
            text="Habilitar Segmentação por Clusters", 
            variable=self.segmentation_enabled_var,
            command=self._toggle_segmentation_section
        )
        self.segmentation_check.pack(side="left")
        
        # LINHA 3: THRESHOLD_REFLECTANCE
        ttk.Label(config_frame, text="Limiar de Reflectância (Fundo):").grid(row=3, column=0, sticky="w")
        self.entry_threshold = ttk.Entry(config_frame, textvariable=self.threshold_var, width=10)
        self.entry_threshold.grid(row=3, column=1, padx=5, pady=2, sticky="w")
        
        # LINHA 4: Configurações de Blur
        self.blur_frame = ttk.Frame(config_frame)
        self.blur_frame.grid(row=4, column=0, columnspan=3, sticky="we", pady=5)
        
        # Checkbox para aplicar blur
        self.blur_check = ttk.Checkbutton(self.blur_frame, text="Aplicar Blur antes da classificação", 
                                         variable=self.aplicar_blur_var)
        self.blur_check.pack(side="left", padx=(0, 10))
        
        # Tipo de blur
        ttk.Label(self.blur_frame, text="Tipo:").pack(side="left", padx=(0, 5))
        self.tipo_blur_combo = ttk.Combobox(self.blur_frame, textvariable=self.tipo_blur_var,
                                           values=["uniforme", "gaussiano"], width=10, state="readonly")
        self.tipo_blur_combo.pack(side="left", padx=(0, 10))
        
        # Tamanho do kernel
        ttk.Label(self.blur_frame, text="Kernel:").pack(side="left", padx=(0, 5))
        self.tamanho_blur_combo = ttk.Combobox(self.blur_frame, textvariable=self.tamanho_blur_var,
                                              values=["3", "5", "7"], width=5, state="readonly")
        self.tamanho_blur_combo.pack(side="left", padx=(0, 10))
        
        # Sigma para blur gaussiano
        ttk.Label(self.blur_frame, text="Sigma:").pack(side="left", padx=(0, 5))
        self.sigma_blur_entry = ttk.Entry(self.blur_frame, textvariable=self.sigma_blur_var, width=5)
        self.sigma_blur_entry.pack(side="left", padx=(0, 10))
        
        # LINHA 5: Escolha do Método de Segmentação
        self.segment_method_frame = ttk.Frame(config_frame)
        self.segment_method_frame.grid(row=5, column=0, columnspan=3, sticky="we", pady=5)

        ttk.Label(self.segment_method_frame, text="Método de Segmentação:").pack(side="left", padx=(0, 5))
        self.method_combo = ttk.Combobox(self.segment_method_frame, textvariable=self.segmentation_method_var, 
                                    values=["K-Means", "DBSCAN"], width=10, state="readonly")
        self.method_combo.pack(side="left", padx=(0, 20))
        self.method_combo.bind("<<ComboboxSelected>>", self._toggle_segmentation_fields)

        # Frame para os parâmetros K-Means/DBSCAN
        self.segment_params_frame = ttk.Frame(self.segment_method_frame)
        self.segment_params_frame.pack(side="left", fill="x", expand=True)

        # LINHA 6: Configuração de Nomes de Classes
        self.class_names_outer_frame = ttk.Frame(config_frame)
        self.class_names_outer_frame.grid(row=6, column=0, columnspan=3, sticky="we", pady=5)
        ttk.Label(self.class_names_outer_frame, text="Nomes das Classes (0, 1, ...):").pack(side="left", padx=(0, 5))
        
        self.class_names_frame = ttk.Frame(self.class_names_outer_frame)
        self.class_names_frame.pack(side="left", fill="x", expand=True)
        
        # Configurações iniciais
        self._toggle_segmentation_section()  # Inicializar estado da seção
        self._toggle_segmentation_fields()

        # ====== BOTÕES DE AÇÃO ======
        action_frame = ttk.Frame(outer)
        action_frame.pack(fill="x", pady=5)

        self.btn_run = ttk.Button(action_frame, text="Processar Tudo", command=self._start_thread)
        self.btn_run.pack(side="left")

        self.btn_stop = ttk.Button(action_frame, text="Parar Processamento", command=self._stop_processing, state="disabled")
        self.btn_stop.pack(side="left", padx=(10, 0))

        self.btn_clear_log = ttk.Button(action_frame, text="Limpar Console", command=self._clear_log)
        self.btn_clear_log.pack(side="left", padx=(10, 0))

        # ====== LOG / CONSOLE ======
        log_frame = ttk.LabelFrame(outer, text="Log de Execução")
        log_frame.pack(fill="both", expand=True, pady=5)

        self.txt_log = tk.Text(log_frame, height=15, wrap="none", state="normal", bg="#111", fg="#0f0")
        self.txt_log.pack(fill="both", expand=True)

        scroll_y = ttk.Scrollbar(self.txt_log, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")

        self._after_id = self.txt_log.after(100, self._poll_log_queue)

    # -------------------------
    # Controle de Habilitação da Seção de Segmentação
    # -------------------------
    def _toggle_segmentation_section(self):
        """Habilita ou desabilita toda a seção de segmentação baseado no checkbox principal"""
        enabled = self.segmentation_enabled_var.get()
        
        # Lista de widgets a controlar
        widgets_to_control = [
            self.entry_threshold,           # Limiar de Reflectância
            self.blur_check,                # Checkbox de Blur
            self.tipo_blur_combo,           # Tipo de Blur
            self.tamanho_blur_combo,        # Tamanho do Blur
            self.sigma_blur_entry,          # Sigma do Blur
            self.method_combo,              # Método de Segmentação
        ]
        
        # Adicionar widgets dinâmicos de parâmetros de segmentação
        if hasattr(self, 'entry_k'):
            widgets_to_control.append(self.entry_k)
        if hasattr(self, 'entry_eps'):
            widgets_to_control.append(self.entry_eps)
        if hasattr(self, 'entry_min_samples'):
            widgets_to_control.append(self.entry_min_samples)
        if hasattr(self, 'entry_max_clusters'):
            widgets_to_control.append(self.entry_max_clusters)
        
        # Adicionar widgets de nomes de classes
        for widget in self.class_names_frame.winfo_children():
            widgets_to_control.append(widget)
        
        # Aplicar estado a todos os widgets
        for widget in widgets_to_control:
            if widget:
                widget.configure(state="normal" if enabled else "disabled")
        
        # Log do estado
        estado = "HABILITADA" if enabled else "DESABILITADA"
        self._log(f"[CONFIG] Segmentação por clusters {estado}")

    # -------------------------
    # Navegação de pastas
    # -------------------------
    def _browse_root(self):
        d = filedialog.askdirectory(title="Selecione a pasta raiz (com subpastas capture/)")
        if d:
            self.entry_root.delete(0, "end")
            self.entry_root.insert(0, d)

    def _browse_out(self):
        d = filedialog.askdirectory(title="Selecione a pasta de saída")
        if d:
            self.entry_out.delete(0, "end")
            self.entry_out.insert(0, d)

    def _browse_dark(self):
        d = filedialog.askdirectory(title="Selecione o diretório do arquivo Dark (DARKREF_*.hdr)")
        if d:
            self.entry_dark_dir.delete(0, "end")
            self.entry_dark_dir.insert(0, d)

    def _browse_white(self):
        d = filedialog.askdirectory(title="Selecione o diretório do arquivo White (WHITEREF_*.hdr)")
        if d:
            self.entry_white_dir.delete(0, "end")
            self.entry_white_dir.insert(0, d)

    # -------------------------
    # Controle de Processamento
    # -------------------------
    def _stop_processing(self):
        """Para o processamento em andamento"""
        self.processing = False
        self._log("[INFO] Solicitando parada do processamento... Aguarde a finalização da amostra atual.")
        self.btn_stop.config(state="disabled")

    # -------------------------
    # Gerenciamento dinâmico de classes e parâmetros
    # -------------------------
    def _toggle_segmentation_fields(self, event=None):
        """Alterna a exibição dos parâmetros de K-Means ou DBSCAN."""
        method = self.segmentation_method_var.get()

        for widget in self.segment_params_frame.winfo_children():
            widget.destroy()

        if method == "K-Means":
            ttk.Label(self.segment_params_frame, text="Nº de Classes (K):").pack(side="left", padx=(0, 5))
            self.entry_k = ttk.Entry(self.segment_params_frame, textvariable=self.k_clusters_var, width=5)
            self.entry_k.pack(side="left", padx=(0, 10))
            self.k_clusters_var.trace_add("write", self._update_class_names_fields)
            
        elif method == "DBSCAN":
            ttk.Label(self.segment_params_frame, text="Epsilon (eps):").pack(side="left", padx=(0, 5))
            self.entry_eps = ttk.Entry(self.segment_params_frame, textvariable=self.dbscan_eps_var, width=8)
            self.entry_eps.pack(side="left", padx=(0, 10))
            
            ttk.Label(self.segment_params_frame, text="Min Samples:").pack(side="left", padx=(0, 5))
            self.entry_min_samples = ttk.Entry(self.segment_params_frame, textvariable=self.dbscan_min_samples_var, width=5)
            self.entry_min_samples.pack(side="left", padx=(0, 10))
            
            ttk.Label(self.segment_params_frame, text="Máx. Clusters Esperados:").pack(side="left", padx=(20, 5))
            self.entry_max_clusters = ttk.Entry(self.segment_params_frame, textvariable=self.max_clusters_var, width=5)
            self.entry_max_clusters.pack(side="left", padx=(0, 10))
            self.max_clusters_var.trace_add("write", self._update_class_names_fields)
            
        self._update_class_names_fields()
        
        # Aplicar estado de habilitação aos novos widgets
        self._toggle_segmentation_section()

    def _update_class_names_fields(self, *args):
        """Atualiza a exibição dos campos de nome de classe com base no método."""
        method = self.segmentation_method_var.get()
        
        # Limpa o frame de nomes de classes
        for widget in self.class_names_frame.winfo_children():
            widget.destroy()

        # Determina quantos campos mostrar
        try:
            if method == "K-Means":
                k = int(self.k_clusters_var.get())
                if k < 1 or k > 20:
                    k = 2
                num_fields = k
                
            elif method == "DBSCAN":
                max_clusters = int(self.max_clusters_var.get())
                if max_clusters < 1 or max_clusters > 20:
                    max_clusters = 5
                num_fields = max_clusters
                
        except ValueError:
            num_fields = 2

        # Garante que temos variáveis suficientes
        while len(self.class_names_vars) < num_fields:
            self.class_names_vars.append(tk.StringVar(value=f"Classe {len(self.class_names_vars)}"))

        # Cria novos campos de entrada
        for i in range(num_fields):
            ttk.Label(self.class_names_frame, text=f"C{i}:").pack(side="left", padx=(5, 0))
            entry = ttk.Entry(self.class_names_frame, textvariable=self.class_names_vars[i], width=12)
            entry.pack(side="left", padx=(0, 5))

        # Aplicar estado de habilitação aos novos widgets
        self._toggle_segmentation_section()

    # -------------------------
    # Funções de Blur
    # -------------------------
    def _aplicar_blur_imagem(self, imagem: np.ndarray, tipo_blur: str, tamanho_kernel: int, sigma: float = 1.0) -> np.ndarray:
        """
        Aplica blur na imagem hiperespectral.
        
        Args:
            imagem: Array numpy (H, W, B) com a imagem hiperespectral
            tipo_blur: 'uniforme' ou 'gaussiano'
            tamanho_kernel: Tamanho do kernel (3, 5, 7, etc.)
            sigma: Parâmetro sigma para blur gaussiano
        
        Returns:
            Imagem com blur aplicado
        """
        if imagem is None:
            return None
            
        imagem_blur = np.zeros_like(imagem)
        h, w, b = imagem.shape
        
        self._log(f"[BLUR] Aplicando {tipo_blur} {tamanho_kernel}x{tamanho_kernel} (sigma: {sigma})")
        
        for banda in range(b):
            banda_original = imagem[:, :, banda]
            
            if tipo_blur == "uniforme":
                # Blur uniforme (média)
                imagem_blur[:, :, banda] = uniform_filter(banda_original, size=tamanho_kernel)
            elif tipo_blur == "gaussiano":
                # Blur gaussiano
                imagem_blur[:, :, banda] = gaussian_filter(banda_original, sigma=sigma)
            else:
                # Sem blur - copia original
                imagem_blur[:, :, banda] = banda_original
        
        # Estatísticas para debug
        diff = np.mean(np.abs(imagem_blur - imagem))
        self._log(f"[BLUR] Diferença média após blur: {diff:.6f}")
        
        return imagem_blur

    # -------------------------
    # Funções de Segmentação Corrigidas
    # -------------------------
    def _segmentar_amostra_com_debug(self, cube_corr: np.ndarray, mask_sample: np.ndarray, 
                                    method: str, k_clusters: int, eps: float, min_samples: int):
        
        # Verificar se o processamento foi interrompido
        if not self.processing:
            return np.full(cube_corr.shape[:2], -1, dtype=np.int32), []
            
        h, w, b = cube_corr.shape
        pixels_sample = cube_corr[mask_sample, :]
        
        num_pixels_validos = pixels_sample.shape[0]
        
        # DEBUG DETALHADO
        self._log(f"[DEBUG_SEG] Método: {method}")
        self._log(f"[DEBUG_SEG] Pixels na máscara: {num_pixels_validos}")
        self._log(f"[DEBUG_SEG] Parâmetros - eps: {eps}, min_samples: {min_samples}, k: {k_clusters}")
        
        if num_pixels_validos == 0:
            self._log(f"[DEBUG_SEG] ERRO: Nenhum pixel na máscara!")
            return np.full((h, w), -1, dtype=np.int32), []
        
        # Verificar estatísticas dos pixels
        if num_pixels_validos > 0:
            mean_spectrum = np.mean(pixels_sample, axis=0)
            std_spectrum = np.std(pixels_sample, axis=0)
            self._log(f"[DEBUG_SEG] Reflectância média: {np.mean(mean_spectrum):.6f}")
            self._log(f"[DEBUG_SEG] Std dev média: {np.mean(std_spectrum):.6f}")
            self._log(f"[DEBUG_SEG] Min pixel: {np.min(pixels_sample):.6f}, Max pixel: {np.max(pixels_sample):.6f}")
        
        if method == "kmeans" and num_pixels_validos < (k_clusters * 2):
            self._log(f"[DEBUG_SEG] INSUFICIENTE KMEANS: {num_pixels_validos} < {k_clusters * 2}")
            return np.full((h, w), -1, dtype=np.int32), []
            
        if method == "dbscan" and num_pixels_validos < min_samples:
            self._log(f"[DEBUG_SEG] INSUFICIENTE DBSCAN: {num_pixels_validos} < {min_samples}")
            return np.full((h, w), -1, dtype=np.int32), []
        
        # Executar algoritmo de segmentação
        labels_sample = None
        unique_labels = []

        try:
            if method == "kmeans":
                km = kis.KMeans(n_clusters=k_clusters, random_state=42, n_init=K_N_INIT)
                labels_sample = km.fit_predict(pixels_sample)
                unique_labels = list(range(k_clusters))
                self._log(f"[DEBUG_SEG] KMeans executado - labels: {np.unique(labels_sample)}")
                
            elif method == "dbscan":
                # USANDO DBSCAN DO SKLEARN DIRETAMENTE
                db = DBSCAN(eps=eps, min_samples=min_samples)
                labels_sample = db.fit_predict(pixels_sample)
                unique_labels_raw = np.unique(labels_sample)
                unique_labels = [l for l in unique_labels_raw if l != -1]
                
                num_noise = np.sum(labels_sample == -1)
                num_clusters = len(unique_labels)
                
                self._log(f"[DEBUG_SEG] DBSCAN resultado:")
                self._log(f"[DEBUG_SEG]   - Clusters encontrados: {num_clusters}")
                self._log(f"[DEBUG_SEG]   - Pixels como ruído: {num_noise}/{num_pixels_validos}")
                self._log(f"[DEBUG_SEG]   - Labels únicos: {unique_labels_raw}")
                
        except Exception as e:
            self._log(f"[DEBUG_SEG] ERRO no algoritmo: {e}")
            return np.full((h, w), -1, dtype=np.int32), []

        if labels_sample is None or len(unique_labels) == 0:
            self._log(f"[DEBUG_SEG] FALHA: Nenhum cluster válido gerado")
            return np.full((h, w), -1, dtype=np.int32), []
        
        # Mapear de volta para imagem completa
        labels_full = np.full((h, w), -1, dtype=np.int32)
        labels_full[mask_sample] = labels_sample
        
        self._log(f"[DEBUG_SEG] SUCESSO: {len(unique_labels)} clusters gerados")
        return labels_full, unique_labels

    def _reordenar_clusters_consistentemente(self, cube_corr: np.ndarray, labels_full: np.ndarray, 
                                           unique_labels: list, class_names: list):
        """
        Reordena os clusters para manter consistência entre amostras.
        Baseado na reflectância média - cluster mais claro primeiro.
        """
        # Verificar se o processamento foi interrompido
        if not self.processing:
            return labels_full, unique_labels
            
        if len(unique_labels) <= 1:
            return labels_full, unique_labels

        # Calcular reflectância média para cada cluster
        cluster_reflectances = []
        for cluster_id in unique_labels:
            mask_cluster = (labels_full == cluster_id)
            pixels_cluster = cube_corr[mask_cluster, :]
            if pixels_cluster.size > 0:
                mean_reflectance = np.mean(pixels_cluster)
                cluster_reflectances.append((cluster_id, mean_reflectance))
            else:
                cluster_reflectances.append((cluster_id, 0.0))

        # Ordenar clusters pela reflectância (mais claro primeiro)
        cluster_reflectances.sort(key=lambda x: x[1], reverse=True)
        
        self._log(f"[REORDENAR] Reflectâncias dos clusters:")
        for orig_id, reflectance in cluster_reflectances:
            self._log(f"[REORDENAR]   Cluster {orig_id}: {reflectance:.6f}")

        # Criar mapeamento de reordenação
        reorder_map = {}
        for new_id, (orig_id, _) in enumerate(cluster_reflectances):
            reorder_map[orig_id] = new_id
            self._log(f"[REORDENAR]   Cluster {orig_id} -> {new_id} ('{class_names[new_id]}')")

        # Aplicar reordenação
        labels_ordered = np.full_like(labels_full, -1)
        for orig_id, new_id in reorder_map.items():
            labels_ordered[labels_full == orig_id] = new_id

        unique_labels_ordered = list(range(len(unique_labels)))
        
        return labels_ordered, unique_labels_ordered

    # -------------------------
    # NOVA FUNÇÃO: Processar sem segmentação (apenas correção radiométrica)
    # -------------------------
    def _processar_sem_segmentacao(self, cube_corr: np.ndarray, sample_hdr: str, dark_hdr: str, 
                                 white_hdr: str, base_name: str, out_dir: str, threshold_reflectance: float):
        """Processa imagem apenas com correção radiométrica, sem segmentação."""
        H, W, B = cube_corr.shape
        
        self._log(f"[CORRECAO] Aplicando correção radiométrica em {base_name}")
        
        # Calcular estatísticas da imagem corrigida
        mean_reflectance = np.mean(cube_corr, axis=2)
        std_reflectance = np.std(cube_corr, axis=2)
        
        # Calcular média espectral de toda a imagem
        mean_spectral = np.mean(cube_corr, axis=(0, 1))
        median_spectral = np.median(cube_corr, axis=(0, 1))
        
        # Salvar arquivo .npy com dados espectrais completos
        npy_filename = f"{base_name}_corrigido.npy"
        npy_path = os.path.join(out_dir, npy_filename)
        np.save(npy_path, cube_corr)
        self._log(f"[SAVE] NPY salvo: {npy_filename}")

        png_rgb_filename = f"{base_name}_corrigido_rgb.png"
        png_rgb_path = os.path.join(out_dir, png_rgb_filename)
        _save_rgb_png(cube_corr, png_rgb_path)
        self._log(f"[SAVE] PNG salvo: {png_rgb_filename}")
        
        # Preparar dados para JSON
        spectral_data = {
            "arquivo_origem": os.path.basename(sample_hdr),
            "caminho_origem": sample_hdr,
            "data_processamento": datetime.now().isoformat(),
            "operacao": "correcao_radiometrica",
            "segmentacao_aplicada": False,
            "shape_imagem": [H, W, B],
            "estatisticas_gerais": {
                "reflectancia_media_geral": float(np.mean(mean_reflectance)),
                "reflectancia_std_geral": float(np.mean(std_reflectance)),
                "reflectancia_min": float(np.min(cube_corr)),
                "reflectancia_max": float(np.max(cube_corr)),
                "pixels_totais": H * W
            },
            "media_espectral": mean_spectral.tolist(),
            "mediana_espectral": median_spectral.tolist(),
            "parametros_correcao": {
                "dark_ref": os.path.basename(dark_hdr),
                "white_ref": os.path.basename(white_hdr),
                "threshold_aplicado": threshold_reflectance
            },
            "arquivos_gerados": {
                "npy": npy_filename,
                "json": f"{base_name}_corrigido.json",
                "png": png_rgb_filename
            }
        }
        
        # Salvar arquivo JSON
        json_filename = f"{base_name}_corrigido.json"
        json_path = os.path.join(out_dir, json_filename)
        if _save_spectral_json(spectral_data, json_path):
            self._log(f"[SAVE] JSON salvo: {json_filename}")
        else:
            self._log(f"[ERRO] Falha ao salvar JSON: {json_filename}")
        
        # Preparar linha para Excel/CSV
        row_dict = {
            "amostra_id": base_name,
            "classe": "imagem_completa",
            "cluster_id": 0,
            "operacao": "correcao_radiometrica",
            "pixels_totais": H * W,
            "reflectancia_media_geral": float(np.mean(mean_reflectance)),
            "num_bandas": B,
        }
        
        # Adicionar bandas espectrais
        for i, v in enumerate(mean_spectral):
            row_dict[f"band_{i}"] = float(v)
            
        return row_dict

    # -------------------------
    # Log / Thread wrappers
    # -------------------------
    def _start_thread(self):
        try:
            # Verificar se segmentação está habilitada
            segmentation_enabled = self.segmentation_enabled_var.get()
            
            if segmentation_enabled:
                method = self.segmentation_method_var.get()
                if method == "K-Means":
                    k = int(self.k_clusters_var.get())
                    if k < 1 or k > 20:
                        raise ValueError("K deve ser entre 1 e 20.")
                    class_names = [self.class_names_vars[i].get().strip() for i in range(k)]
                elif method == "DBSCAN":
                    eps = float(self.dbscan_eps_var.get())
                    min_samples = int(self.dbscan_min_samples_var.get())
                    max_clusters = int(self.max_clusters_var.get())
                    if eps <= 0 or min_samples <= 0:
                        raise ValueError("Epsilon e Min Samples devem ser positivos.")
                    if max_clusters < 1 or max_clusters > 20:
                        raise ValueError("Máx. Clusters deve ser entre 1 e 20.")
                    class_names = [self.class_names_vars[i].get().strip() for i in range(max_clusters)]
                
                threshold = float(self.threshold_var.get())
                if threshold < 0:
                    raise ValueError("Limiar de reflectância deve ser positivo ou zero.")

                # Verifica se os nomes das classes estão preenchidos
                if not all(class_names):
                    raise ValueError("Todos os nomes de classes devem ser preenchidos.")
            else:
                # Quando segmentação desabilitada, usar configuração padrão
                class_names = ["imagem_completa"]
                threshold = float(self.threshold_var.get()) if self.threshold_var.get() else 0.0

        except ValueError as e:
            messagebox.showerror("Erro de Configuração", f"Parâmetro inválido: {e}")
            return

        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.processing = True
        t = threading.Thread(target=self._pipeline_full, daemon=True)
        t.start()
        
    def _log(self, msg: str):
        self.log_queue.put(msg)

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.txt_log.insert("end", msg + "\n")
                self.txt_log.see("end")
        except queue.Empty:
            pass
        self._after_id = self.txt_log.after(100, self._poll_log_queue)

    def _clear_log(self):
        self.txt_log.delete("1.0", "end")

    # ----------------------------------------------------
    # FUNÇÃO PRINCIPAL DE PIPELINE CORRIGIDA
    # ----------------------------------------------------
    def _pipeline_full(self):
        root_dir = self.entry_root.get().strip()
        out_dir  = self.entry_out.get().strip()
        base_filename = self.entry_filename.get().strip()
        file_format = self.format_var.get()
        dark_dir_override = self.entry_dark_dir.get().strip()
        white_dir_override = self.entry_white_dir.get().strip()
        
        # Verificar se segmentação está habilitada
        segmentation_enabled = self.segmentation_enabled_var.get()
        
        # Obter configurações de segmentação apenas se habilitada
        seg_method = None
        seg_params = {}
        class_names = []
        
        if segmentation_enabled:
            seg_method = self.segmentation_method_var.get()
            seg_params = {"method": seg_method}
            
            if seg_method == "K-Means":
                k_clusters = int(self.k_clusters_var.get())
                class_names = [self.class_names_vars[i].get().strip() for i in range(k_clusters)]
                seg_params["k_clusters"] = k_clusters
                seg_params["k_n_init"] = K_N_INIT
                max_expected_clusters = k_clusters
                
            elif seg_method == "DBSCAN":
                eps = float(self.dbscan_eps_var.get())
                min_samples = int(self.dbscan_min_samples_var.get())
                max_expected_clusters = int(self.max_clusters_var.get())
                class_names = [self.class_names_vars[i].get().strip() for i in range(max_expected_clusters)]
                seg_params["eps"] = eps
                seg_params["min_samples"] = min_samples
                seg_params["max_expected_clusters"] = max_expected_clusters
        else:
            # Quando segmentação desabilitada, usar configuração padrão
            class_names = ["imagem_completa"]
            max_expected_clusters = 1
        
        # Obter configurações de blur APENAS se estiver habilitado e segmentação ativa
        aplicar_blur = self.aplicar_blur_var.get() and segmentation_enabled
        blur_params = None
        if aplicar_blur:
            try:
                blur_params = {
                    "tipo_blur": self.tipo_blur_var.get(),
                    "tamanho_blur": int(self.tamanho_blur_var.get()),
                    "sigma_blur": float(self.sigma_blur_var.get())
                }
            except ValueError as e:
                self._log(f"[ERRO] Parâmetros de blur inválidos: {e}")
                self.btn_run.config(state="normal")
                self.btn_stop.config(state="disabled")
                self.processing = False
                return
        
        try:
            threshold_reflectance = float(self.threshold_var.get()) if self.threshold_var.get() else 0.0
            
        except Exception as e:
            self._log(f"[ERRO] Configuração inválida: {e}")
            self.btn_run.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.processing = False
            return
            
        if not base_filename:
            base_filename = DEFAULT_FILENAME
            
        if not os.path.isdir(root_dir):
            self._log(f"[ERRO] Pasta raiz inválida: {root_dir}")
            self.btn_run.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.processing = False
            return

        os.makedirs(out_dir, exist_ok=True)
        
        # Estruturas para armazenar resultados
        rows_excel = []
        
        capture_dirs = _find_all_capture_dirs(root_dir)
        
        # Log do modo de operação
        if segmentation_enabled:
            self._log(f"[INFO] MODO: Segmentação HABILITADA - Método: **{seg_method}**")
            self._log(f"[INFO] Limiar de Fundo: {threshold_reflectance}")
        else:
            self._log(f"[INFO] MODO: Segmentação DESABILITADA - Apenas correção radiométrica")
            self._log(f"[INFO] Limiar de Fundo: {threshold_reflectance} (para estatísticas)")

        self._log(f"[INFO] Encontradas {len(capture_dirs)} pastas 'capture'.")

        for cap in capture_dirs:
            # Verificar se o processamento foi interrompido
            if not self.processing:
                self._log("[INFO] Processamento interrompido pelo usuário.")
                break
                
            self._log(f"[INFO] Processando capture: {cap}")

            dark_hdrs = []
            white_hdrs = []
            sample_hdrs = []

            for hdr_path in _list_hdrs_amostra(cap):
                tipo = _classify_type(hdr_path)
                if tipo == "dark":
                    dark_hdrs.append(hdr_path)
                elif tipo == "white":
                    white_hdrs.append(hdr_path)
                else:
                    sample_hdrs.append(hdr_path)

            # LÓGICA DE OVERRIDE PARA DARK
            dark_hdr = None
            if dark_dir_override and os.path.isdir(dark_dir_override):
                for f in os.listdir(dark_dir_override):
                    if f.lower().startswith("darkref_") and f.lower().endswith(".hdr"):
                        dark_hdr = os.path.join(dark_dir_override, f)
                        self._log(f"[INFO] Usando Dark override: {os.path.basename(dark_hdr)}")
                        break
            
            # Se não encontrou no override ou override não foi especificado, usa o local
            if not dark_hdr and dark_hdrs:
                dark_hdr = dark_hdrs[0]
                if dark_dir_override:
                    self._log(f"[WARN] Dark não encontrado no override, usando local: {os.path.basename(dark_hdr)}")
                else:
                    self._log(f"[INFO] Usando Dark local: {os.path.basename(dark_hdr)}")

            # LÓGICA DE OVERRIDE PARA WHITE (agora incluindo R99_)
            white_hdr = None
            if white_dir_override and os.path.isdir(white_dir_override):
                for f in os.listdir(white_dir_override):
                    f_lower = f.lower()
                    if (f_lower.startswith("whiteref_") or f_lower.startswith("r99_") or f_lower.startswith("r99-")) and f_lower.endswith(".hdr"):
                        white_hdr = os.path.join(white_dir_override, f)
                        self._log(f"[INFO] Usando White override: {os.path.basename(white_hdr)}")
                        break
            
            # Se não encontrou no override ou override não foi especificado, usa o local
            if not white_hdr and white_hdrs:
                white_hdr = white_hdrs[0]
                if white_dir_override:
                    self._log(f"[WARN] White não encontrado no override, usando local: {os.path.basename(white_hdr)}")
                else:
                    self._log(f"[INFO] Usando White local: {os.path.basename(white_hdr)}")

            if not dark_hdr or not white_hdr:
                self._log(f"[WARN] DARKREF ou WHITEREF não encontrado para {cap}, pulando pasta.")
                continue
            
            self._log(f"[INFO] DARKREF: {os.path.basename(dark_hdr)} | WHITEREF: {os.path.basename(white_hdr)}")

            if not sample_hdrs:
                continue
                
            for sample_hdr in sample_hdrs:
                # Verificar se o processamento foi interrompido antes de cada amostra
                if not self.processing:
                    self._log("[INFO] Processamento interrompido pelo usuário.")
                    break
                    
                base_name = os.path.splitext(os.path.basename(sample_hdr))[0]
                self._log(f"[INFO] -> Amostra: {base_name}")

                try:
                    # Aplicar correção radiométrica
                    cube_corr = _load_and_correct(
                        img_hdr_path=sample_hdr,
                        dark_hdr_path=dark_hdr,
                        white_hdr_path=white_hdr
                    )
                    H, W, B = cube_corr.shape
                    
                    # ========== MODO SEM SEGMENTAÇÃO ==========
                    if not segmentation_enabled:
                        # Processar apenas com correção radiométrica
                        row = self._processar_sem_segmentacao(
                            cube_corr=cube_corr,
                            sample_hdr=sample_hdr,
                            dark_hdr=dark_hdr,
                            white_hdr=white_hdr,
                            base_name=base_name,
                            out_dir=out_dir,
                            threshold_reflectance=threshold_reflectance
                        )
                        if row:
                            rows_excel.append(row)
                            self._log(f"[SUCESSO] Correção radiométrica concluída para {base_name}")
                    
                    # ========== MODO COM SEGMENTAÇÃO ==========
                    else:
                        # Aplicar blur se configurado
                        if aplicar_blur and blur_params is not None:
                            self._log(f"[BLUR] Aplicando blur na amostra {base_name}...")
                            cube_corr = self._aplicar_blur_imagem(
                                cube_corr, 
                                blur_params["tipo_blur"], 
                                blur_params["tamanho_blur"], 
                                blur_params["sigma_blur"]
                            )
                        
                        # Aplicar threshold para criar máscara
                        mean_reflectance = np.mean(cube_corr, axis=2)
                        mask_sample = mean_reflectance > threshold_reflectance
                        num_pixels_mask = np.sum(mask_sample)
                        
                        self._log(f"[SEG] Pixels acima do threshold: {num_pixels_mask}/{H*W}")
                        
                        if num_pixels_mask == 0:
                            self._log(f"[WARN] Nenhum pixel acima do threshold, pulando amostra.")
                            continue
                            
                        # Executar segmentação
                        labels_full, unique_labels = self._segmentar_amostra_com_debug(
                            cube_corr=cube_corr, 
                            mask_sample=mask_sample, 
                            method=seg_method.lower().replace('-', ''),
                            k_clusters=seg_params.get("k_clusters", 0), 
                            eps=seg_params.get("eps", 0),
                            min_samples=seg_params.get("min_samples", 0)
                        )

                        if not unique_labels:
                            self._log(f"[WARN] Segmentação não gerou clusters válidos, pulando.")
                            continue

                        # Reordenar clusters
                        labels_full, unique_labels = self._reordenar_clusters_consistentemente(
                            cube_corr=cube_corr,
                            labels_full=labels_full,
                            unique_labels=unique_labels,
                            class_names=class_names
                        )

                        # Processar cada cluster
                        for cluster_id in unique_labels:
                            if not self.processing:
                                break
                                
                            if cluster_id >= len(class_names):
                                continue
                                 
                            class_name = class_names[cluster_id]
                            mask_cluster = (labels_full == cluster_id)
                            
                            if np.sum(mask_cluster) == 0:
                                continue

                            # Processar cluster
                            masked_cube, processed_mask = _processar_mascara_completa(cube_corr, mask_cluster)
                            
                            if masked_cube is None:
                                continue

                            mean_spec, num_pixels = _calc_media_espectral_completa(masked_cube, processed_mask)
                            
                            if mean_spec is None:
                                continue

                            # Salvar NPY
                            npy_filename = f"{base_name}_C{cluster_id}_{class_name.replace(' ', '_').lower()}.npy"
                            npy_path = os.path.join(out_dir, npy_filename)
                            np.save(npy_path, masked_cube)

                            png_mask_filename = f"{base_name}_C{cluster_id}_{class_name.replace(' ', '_').lower()}_mask.png"
                            png_mask_path = os.path.join(out_dir, png_mask_filename)
                            _save_mask_png(mask_cluster, png_mask_path)

                            # Preparar dados para JSON
                            spectral_data = {
                                "arquivo_origem": os.path.basename(sample_hdr),
                                "caminho_origem": sample_hdr,
                                "data_processamento": datetime.now().isoformat(),
                                "operacao": "segmentacao",
                                "segmentacao_aplicada": True,
                                "classe": class_name,
                                "cluster_id": cluster_id,
                                "shape_imagem": [H, W, B],
                                "pixels_selecionados": num_pixels,
                                "media_espectral": mean_spec.tolist(),
                                "parametros_segmentacao": seg_params,
                                "parametros_correcao": {
                                    "dark_ref": os.path.basename(dark_hdr),
                                    "white_ref": os.path.basename(white_hdr),
                                    "threshold_aplicado": threshold_reflectance
                                },
                                "arquivos_gerados": {
                                    "npy": npy_filename,
                                    "json": f"{base_name}_C{cluster_id}_{class_name.replace(' ', '_').lower()}.json",
                                    "png": png_mask_filename 
                                }
                            }
                            
                            # Salvar JSON
                            json_filename = f"{base_name}_C{cluster_id}_{class_name.replace(' ', '_').lower()}.json"
                            json_path = os.path.join(out_dir, json_filename)
                            if _save_spectral_json(spectral_data, json_path):
                                self._log(f"[SAVE] JSON salvo: {json_filename}")
                            
                            # Preparar linha para Excel
                            row_dict = {
                                "amostra_id": base_name,
                                "classe": class_name,
                                "cluster_id": cluster_id,
                                "operacao": "segmentacao",
                                "pixels_selecionados": num_pixels,
                                "num_bandas": mean_spec.shape[0],
                            }
                            
                            for i, v in enumerate(mean_spec):
                                row_dict[f"band_{i}"] = float(v)
                                
                            rows_excel.append(row_dict)
                            self._log(f"[APPEND] Cluster {cluster_id} ('{class_name}') processado")

                except Exception as e:
                    self._log(f"[ERRO] Falha na amostra {base_name}: {e}")

        # Salvar arquivo final
        if self.processing and rows_excel:
            df = pd.DataFrame(rows_excel)
            output_path = os.path.join(out_dir, base_filename + file_format)
            
            try:
                if file_format == ".xlsx":
                    df.to_excel(output_path, index=False)
                else:
                    df.to_csv(output_path, index=False)
                    
                self._log(f"[SAVE] Arquivo final salvo em: {output_path}")
                self._log(f"[SAVE] Total de registros: {len(df)}")
                
            except Exception as e:
                self._log(f"[ERRO] Falha ao salvar arquivo {file_format}: {e}")
        else:
            self._log("[WARN] Nenhum dado processado.")

        self._log("[INFO] Pipeline concluído.")
        
        # Restaurar estado dos botões
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.processing = False


