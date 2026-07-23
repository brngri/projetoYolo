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
from scipy.ndimage import gaussian_filter, uniform_filter

import spectral_utils as su

# =========================
# CONFIGURAÇÕES E CONSTANTES
# =========================
START_ROOT_DIR = r"C:\COLOQUE\A_PASTA_RAIZ_COM_AS_CAPTURES"
START_OUT_DIR  = r"C:\COLOQUE\PASTA_DE_SAIDA_PARA_NPY_E_CSV"
DEFAULT_FILENAME = "resumo_medias"

THRESHOLD_REFLECTANCE_DEFAULT = 0.015

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

def _save_spectral_json(spectral_data: dict, file_path: str):
    """Salva dados espectrais em formato JSON."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(spectral_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Erro ao salvar JSON {file_path}: {e}")
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
        self.threshold_var = tk.StringVar(value=str(THRESHOLD_REFLECTANCE_DEFAULT))
        
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
        
        # ====== Seção de Calibração e Processamento ======
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
        
        # LINHA 2: THRESHOLD_REFLECTANCE
        ttk.Label(config_frame, text="Limiar de Reflectância:").grid(row=2, column=0, sticky="w")
        self.entry_threshold = ttk.Entry(config_frame, textvariable=self.threshold_var, width=10)
        self.entry_threshold.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        
        # LINHA 3: Configurações de Blur
        self.blur_frame = ttk.Frame(config_frame)
        self.blur_frame.grid(row=3, column=0, columnspan=3, sticky="we", pady=5)
        
        # Checkbox para aplicar blur
        self.blur_check = ttk.Checkbutton(self.blur_frame, text="Aplicar Blur antes da correção", 
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
    # Processamento da Amostra Corrigida
    # -------------------------
    def _processar_amostra_corrigida(self, cube_corr: np.ndarray, sample_hdr: str, 
                                     dark_hdr: str, white_hdr: str, base_name: str, 
                                     out_dir: str, threshold_reflectance: float):
        """Processa apenas a correção radiométrica, sem segmentação."""
        H, W, B = cube_corr.shape
        
        self._log(f"[CORRECAO] Processando {base_name}")
        
        # Calcular estatísticas da imagem corrigida
        mean_reflectance = np.mean(cube_corr, axis=2)
        std_reflectance = np.std(cube_corr, axis=2)
        
        # Calcular média espectral de toda a imagem
        mean_spectral = np.mean(cube_corr, axis=(0, 1))
        median_spectral = np.median(cube_corr, axis=(0, 1))
        
        # Salvar arquivo .npy
        npy_filename = f"{base_name}_corrigido.npy"
        npy_path = os.path.join(out_dir, npy_filename)
        np.save(npy_path, cube_corr)
        self._log(f"[SAVE] NPY salvo: {npy_filename}")

        # Salvar RGB PNG
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
        
        # Salvar JSON
        json_filename = f"{base_name}_corrigido.json"
        json_path = os.path.join(out_dir, json_filename)
        if _save_spectral_json(spectral_data, json_path):
            self._log(f"[SAVE] JSON salvo: {json_filename}")
        else:
            self._log(f"[ERRO] Falha ao salvar JSON: {json_filename}")
        
        # Preparar linha para Excel/CSV
        row_dict = {
            "amostra_id": base_name,
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
            # Validar threshold
            threshold = float(self.threshold_var.get())
            if threshold < 0:
                raise ValueError("Limiar de reflectância deve ser positivo ou zero.")
                
            # Validar parâmetros de blur se habilitado
            if self.aplicar_blur_var.get():
                try:
                    tamanho = int(self.tamanho_blur_var.get())
                    sigma = float(self.sigma_blur_var.get())
                    if tamanho < 1 or tamanho % 2 == 0:
                        raise ValueError("Tamanho do kernel deve ser ímpar e positivo")
                    if sigma < 0:
                        raise ValueError("Sigma deve ser positivo")
                except ValueError as e:
                    raise ValueError(f"Parâmetros de blur inválidos: {e}")
                    
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
    # FUNÇÃO PRINCIPAL DE PIPELINE
    # ----------------------------------------------------
    def _pipeline_full(self):
        root_dir = self.entry_root.get().strip()
        out_dir  = self.entry_out.get().strip()
        base_filename = self.entry_filename.get().strip()
        file_format = self.format_var.get()
        dark_dir_override = self.entry_dark_dir.get().strip()
        white_dir_override = self.entry_white_dir.get().strip()
        
        # Obter threshold
        try:
            threshold_reflectance = float(self.threshold_var.get()) if self.threshold_var.get() else 0.0
        except Exception as e:
            self._log(f"[ERRO] Configuração inválida: {e}")
            self.btn_run.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.processing = False
            return
        
        # Obter parâmetros de blur
        aplicar_blur = self.aplicar_blur_var.get()
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
        
        self._log(f"[INFO] MODO: Correção Radiométrica" + (" COM BLUR" if aplicar_blur else ""))
        if aplicar_blur:
            self._log(f"[INFO] Blur: {blur_params['tipo_blur']} {blur_params['tamanho_blur']}x{blur_params['tamanho_blur']} (sigma: {blur_params['sigma_blur']})")
        self._log(f"[INFO] Limiar de Reflectância: {threshold_reflectance}")
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

            # LÓGICA DE OVERRIDE PARA WHITE
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
                    
                    # Aplicar blur se configurado
                    if aplicar_blur and blur_params is not None:
                        self._log(f"[BLUR] Aplicando blur na amostra {base_name}...")
                        cube_corr = self._aplicar_blur_imagem(
                            cube_corr, 
                            blur_params["tipo_blur"], 
                            blur_params["tamanho_blur"], 
                            blur_params["sigma_blur"]
                        )
                    
                    # Processar amostra corrigida
                    row = self._processar_amostra_corrigida(
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