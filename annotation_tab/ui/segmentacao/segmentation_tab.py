# ui/segmentacao/segmentation_tab.py
"""
Aba de Segmentação.

- Formulário com TODOS os parâmetros do DEFAULT_CONFIG, agrupados em seções.
- Parâmetros habilitados/desabilitados conforme o método escolhido.
- Processamento em thread separada (não trava a UI).
- Console compacto + barra de progresso + carrossel de máscaras.
"""

import os
import re
import json
import queue
import logging
import threading
from copy import deepcopy
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# --- Imports a partir do pacote annotation_tab ---
from ...models.segmentacao.config import DEFAULT_CONFIG
from ...models.segmentacao.signatures import load_signatures_from_json
from ...services.segmentacao.preprocessing import apply_blur, apply_savgol, select_bands
from ...services.segmentacao.classifiers import (
    classify_kmeans,
    classify_kmeans_guiado,
    classify_rf,
    classify_otsu,
    prepare_training_data,
)
from ...services.segmentacao.postprocessing import postprocess_mask

from ..anotacao.console import Console


logger = logging.getLogger("SegmentationTab")


# =============================================================================
# Handler de logging que envia mensagens para a fila
# =============================================================================
class _QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[str]"):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


# =============================================================================
# Fábrica
# =============================================================================
def create_segmentation_tab(parent_frame: tk.Frame) -> "SegmentationTab":
    return SegmentationTab(parent_frame)


# =============================================================================
# Classe principal
# =============================================================================
class SegmentationTab:
    """Aba de segmentação com formulário + console + carrossel."""

    def __init__(self, parent_frame: tk.Frame):
        self.parent = parent_frame

        # --- Threading ---
        self._stop_event = threading.Event()
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._progress_queue: "queue.Queue" = queue.Queue()
        self._mask_queue: "queue.Queue[str]" = queue.Queue()   # novos .png gerados
        self._worker_thread: Optional[threading.Thread] = None
        self._alive = True

        # --- Carrossel ---
        self._masks_list: List[str] = []
        self._current_mask_idx: int = 0
        self._current_photo = None      # referência para evitar GC

        # --- Handler de log ---
        self._handler = _QueueLogHandler(self._log_queue)
        self._handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S')
        )
        logging.getLogger().addHandler(self._handler)

        # --- Variáveis ---
        self._init_variables()

        # --- UI ---
        self._build_ui()
        self._update_widget_states()

        # --- Polling ---
        self._poll_queues()

        self.console.log("Aba de segmentação pronta.")

    # =========================================================================
    # Variáveis
    # =========================================================================
    def _init_variables(self) -> None:
        C = DEFAULT_CONFIG

        self.var_root_dir = tk.StringVar(value=C["root_dir"])
        self.var_out_root = tk.StringVar(value=C["out_root"])
        self.var_method = tk.StringVar(value=C["method"])

        self.json_paths: List[str] = list(C.get("json_path") or [])
        self.json_paths_bg: List[str] = list(C.get("json_path_background") or [])

        # KMeans
        km = C["kmeans_params"]
        self.var_km_n_clusters = tk.IntVar(value=km.get("n_clusters", 2))
        self.var_km_n_init = tk.IntVar(value=km.get("n_init", 10))
        self.var_km_max_iter = tk.IntVar(value=km.get("max_iter", 300))
        self.var_km_random_state = tk.IntVar(value=km.get("random_state", 42))
        pca_default = km.get("pca_components")
        self.var_km_pca = tk.StringVar(value="" if pca_default is None else str(pca_default))

        # Random Forest
        rf = C["rf_params"]
        self.var_rf_n_estimators = tk.IntVar(value=rf.get("n_estimators", 300))
        self.var_rf_max_depth = tk.IntVar(value=rf.get("max_depth", 15))
        self.var_rf_min_samples_split = tk.IntVar(value=rf.get("min_samples_split", 5))
        self.var_rf_random_state = tk.IntVar(value=rf.get("random_state", 42))
        self.var_rf_class_weight = tk.StringVar(value=rf.get("class_weight") or "None")
        self.var_rf_n_jobs = tk.IntVar(value=rf.get("n_jobs", -1))
        self.var_rf_prob_threshold = tk.DoubleVar(value=C.get("rf_prob_threshold", 0.5))
        self.var_n_background_samples = tk.IntVar(value=C.get("n_background_samples", 5000))

        # Otsu
        ot = C["otsu_params"]
        self.var_otsu_use_ndvi = tk.BooleanVar(value=ot.get("use_ndvi", False))
        self.var_otsu_band_nm = tk.DoubleVar(value=ot.get("band_nm", 800.0))
        self.var_otsu_red_nm = tk.DoubleVar(value=ot.get("red_nm", 670.0))
        self.var_otsu_nir_nm = tk.DoubleVar(value=ot.get("nir_nm", 800.0))
        self.var_otsu_invert = tk.BooleanVar(value=ot.get("invert", False))

        # Pré-processamento
        self.var_apply_blur = tk.BooleanVar(value=C["apply_blur"])
        self.var_blur_kind = tk.StringVar(value=C["blur_kind"])
        self.var_blur_ksize = tk.IntVar(value=C["blur_ksize"])

        self.var_apply_savgol = tk.BooleanVar(value=C["apply_savgol"])
        self.var_savgol_window = tk.IntVar(value=C["savgol_window"])
        self.var_savgol_polyorder = tk.IntVar(value=C["savgol_polyorder"])

        self.var_band_selection = tk.StringVar(value=C["band_selection"] or "None")
        bsp = C["band_selection_params"]
        self.var_pca_components = tk.IntVar(value=bsp.get("pca_components", 20))
        self.var_pca_variance = tk.DoubleVar(value=bsp.get("pca_variance_retained", 0.99))
        self.var_kbest_k = tk.IntVar(value=bsp.get("kbest_k", 30))

        self.var_use_scaler = tk.BooleanVar(value=C["use_scaler"])

        # Pós-processamento
        pp = C["postprocess"]
        self.var_pp_majority = tk.BooleanVar(value=pp.get("majority_filter", True))
        self.var_pp_majority_size = tk.IntVar(value=pp.get("majority_size", 3))
        self.var_pp_morph = tk.BooleanVar(value=pp.get("morphological_closing", True))
        self.var_pp_closing_kernel = tk.IntVar(value=pp.get("closing_kernel", 11))
        self.var_pp_fill_holes = tk.BooleanVar(value=pp.get("fill_holes", False))
        self.var_pp_min_area = tk.IntVar(value=pp.get("min_area", 16000))
        self.var_pp_circ_filter = tk.BooleanVar(value=pp.get("circularity_filter", True))
        self.var_pp_circ_thresh = tk.DoubleVar(value=pp.get("circularity_threshold", 0.15))

        # Salvamento
        self.var_save_masked_cube = tk.BooleanVar(value=C["save_masked_cube"])
        self.var_save_crops = tk.BooleanVar(value=C["save_crops"])
        self.var_overwrite = tk.BooleanVar(value=C["overwrite"])

        # Status
        self.var_status = tk.StringVar(value="Pronto.")
        self.var_progress = tk.DoubleVar(value=0.0)
        self.var_current_file = tk.StringVar(value="—")
        self.var_mask_counter = tk.StringVar(value="0 / 0")

        # Listas de widgets para enable/disable dinâmico
        self._km_widgets: List[tk.Widget] = []
        self._rf_widgets: List[tk.Widget] = []
        self._otsu_widgets: List[tk.Widget] = []
        self._otsu_ndvi_widgets: List[tk.Widget] = []
        self._otsu_band_widgets: List[tk.Widget] = []
        self._sig_widgets: List[tk.Widget] = []
        self._bg_widgets: List[tk.Widget] = []
        self._band_pca_widgets: List[tk.Widget] = []
        self._band_kbest_widgets: List[tk.Widget] = []
        self._blur_widgets: List[tk.Widget] = []
        self._savgol_widgets: List[tk.Widget] = []
        self._pp_majority_widgets: List[tk.Widget] = []
        self._pp_morph_widgets: List[tk.Widget] = []
        self._pp_circ_widgets: List[tk.Widget] = []

    # =========================================================================
    # Construção da UI
    # =========================================================================
    def _build_ui(self) -> None:
        paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=4)

        params_inner = self._make_scrollable(left)
        self._build_params_panel(params_inner)
        self._build_right_panel(right)

    def _make_scrollable(self, parent: tk.Widget) -> ttk.Frame:
        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _on_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        return inner

    def _build_params_panel(self, parent: ttk.Frame) -> None:
        r = 0
        self._build_paths_section(parent, r); r += 1
        self._build_method_section(parent, r); r += 1
        self._build_signatures_section(parent, r); r += 1
        self._build_kmeans_section(parent, r); r += 1
        self._build_rf_section(parent, r); r += 1
        self._build_otsu_section(parent, r); r += 1
        self._build_preproc_section(parent, r); r += 1
        self._build_postproc_section(parent, r); r += 1
        self._build_save_section(parent, r)

    # ---- helpers de widgets -------------------------------------------------
    def _grid_entry(self, parent, row, label, var, width=12):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
        parent.columnconfigure(1, weight=1)
        return entry

    def _grid_combo(self, parent, row, label, var, values, width=12):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        cb = ttk.Combobox(parent, textvariable=var, values=values,
                          state="readonly", width=width)
        cb.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
        parent.columnconfigure(1, weight=1)
        return cb

    def _check(self, parent, row, label, var, command=None):
        cb = ttk.Checkbutton(parent, text=label, variable=var, command=command)
        cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=1)
        return cb

    # ---- Seções -------------------------------------------------------------
    def _build_paths_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Caminhos")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(1, weight=1)

        ttk.Label(lf, text="Pasta de entrada (.npy)").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(lf, textvariable=self.var_root_dir).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(lf, text="…", width=3,
                   command=self._pick_root_dir).grid(row=0, column=2, padx=2)

        ttk.Label(lf, text="Pasta de saída").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Entry(lf, textvariable=self.var_out_root).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(lf, text="…", width=3,
                   command=self._pick_out_dir).grid(row=1, column=2, padx=2)

    def _build_method_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Método")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)

        for i, (val, txt) in enumerate([
            ("kmeans", "KMeans (não supervisionado)"),
            ("kmeans_guiado", "KMeans Guiado (semi-supervisionado)"),
            ("rf", "Random Forest (supervisionado)"),
            ("otsu", "Otsu (threshold adaptativo)"),
        ]):
            ttk.Radiobutton(
                lf, text=txt, value=val, variable=self.var_method,
                command=self._on_method_change,
            ).grid(row=i, column=0, sticky="w", padx=6, pady=1)

    def _build_signatures_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Assinaturas (JSON)")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(0, weight=1)

        ttk.Label(lf, text="Assinaturas classe alvo:").grid(
            row=0, column=0, sticky="w", padx=4, pady=(4, 0))
        self.lb_json = tk.Listbox(lf, height=3, selectmode=tk.EXTENDED)
        self.lb_json.grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        self._sig_widgets.append(self.lb_json)

        bf = ttk.Frame(lf)
        bf.grid(row=2, column=0, sticky="w", padx=4, pady=(0, 6))
        btn_add_sig = ttk.Button(bf, text="+ Adicionar", command=self._add_json)
        btn_rem_sig = ttk.Button(bf, text="- Remover", command=self._remove_json)
        btn_add_sig.pack(side=tk.LEFT, padx=2)
        btn_rem_sig.pack(side=tk.LEFT, padx=2)
        self._sig_widgets += [bf, btn_add_sig, btn_rem_sig]

        ttk.Label(lf, text="Assinaturas de background:").grid(
            row=3, column=0, sticky="w", padx=4, pady=(4, 0))
        self.lb_json_bg = tk.Listbox(lf, height=3, selectmode=tk.EXTENDED)
        self.lb_json_bg.grid(row=4, column=0, sticky="ew", padx=4, pady=2)
        self._bg_widgets.append(self.lb_json_bg)

        bf2 = ttk.Frame(lf)
        bf2.grid(row=5, column=0, sticky="w", padx=4, pady=(0, 4))
        btn_add_bg = ttk.Button(bf2, text="+ Adicionar", command=self._add_json_bg)
        btn_rem_bg = ttk.Button(bf2, text="- Remover", command=self._remove_json_bg)
        btn_add_bg.pack(side=tk.LEFT, padx=2)
        btn_rem_bg.pack(side=tk.LEFT, padx=2)
        self._bg_widgets += [bf2, btn_add_bg, btn_rem_bg]

        self._refresh_json_listboxes()

    def _build_kmeans_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Parâmetros — KMeans")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(1, weight=1)

        self._km_widgets.append(self._grid_entry(lf, 0, "n_clusters", self.var_km_n_clusters))
        self._km_widgets.append(self._grid_entry(lf, 1, "n_init", self.var_km_n_init))
        self._km_widgets.append(self._grid_entry(lf, 2, "max_iter", self.var_km_max_iter))
        self._km_widgets.append(self._grid_entry(lf, 3, "random_state", self.var_km_random_state))

        ttk.Label(lf, text="pca_components (vazio = off)").grid(
            row=4, column=0, sticky="w", padx=4, pady=1)
        e_pca = ttk.Entry(lf, textvariable=self.var_km_pca, width=12)
        e_pca.grid(row=4, column=1, sticky="ew", padx=4, pady=1)
        self._km_widgets.append(e_pca)

    def _build_rf_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Parâmetros — Random Forest")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(1, weight=1)

        self._rf_widgets.append(self._grid_entry(lf, 0, "n_estimators", self.var_rf_n_estimators))
        self._rf_widgets.append(self._grid_entry(lf, 1, "max_depth", self.var_rf_max_depth))
        self._rf_widgets.append(self._grid_entry(lf, 2, "min_samples_split", self.var_rf_min_samples_split))
        self._rf_widgets.append(self._grid_entry(lf, 3, "random_state", self.var_rf_random_state))
        self._rf_widgets.append(self._grid_combo(
            lf, 4, "class_weight", self.var_rf_class_weight,
            values=["balanced", "balanced_subsample", "None"]))
        self._rf_widgets.append(self._grid_entry(lf, 5, "n_jobs (-1 = todos)", self.var_rf_n_jobs))
        self._rf_widgets.append(self._grid_entry(lf, 6, "prob_threshold", self.var_rf_prob_threshold))
        self._rf_widgets.append(self._grid_entry(lf, 7, "n_background_samples", self.var_n_background_samples))

    def _build_otsu_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Parâmetros — Otsu")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(1, weight=1)

        cb_ndvi = self._check(lf, 0, "Usar NDVI (nir-red)/(nir+red)",
                              self.var_otsu_use_ndvi,
                              command=self._update_widget_states)
        self._otsu_widgets.append(cb_ndvi)

        # Campos usados quando use_ndvi=False
        w = self._grid_entry(lf, 1, "band_nm (banda p/ Otsu)", self.var_otsu_band_nm)
        self._otsu_widgets.append(w)
        self._otsu_band_widgets.append(w)

        # Campos usados quando use_ndvi=True
        w = self._grid_entry(lf, 2, "red_nm", self.var_otsu_red_nm)
        self._otsu_widgets.append(w)
        self._otsu_ndvi_widgets.append(w)

        w = self._grid_entry(lf, 3, "nir_nm", self.var_otsu_nir_nm)
        self._otsu_widgets.append(w)
        self._otsu_ndvi_widgets.append(w)

        cb_inv = self._check(lf, 4, "Inverter máscara", self.var_otsu_invert)
        self._otsu_widgets.append(cb_inv)

    def _build_preproc_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Pré-processamento")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(1, weight=1)

        # Blur — comando para atualizar estado dos sub-campos ao alternar
        cb_blur = self._check(lf, 0, "Aplicar blur",
                              self.var_apply_blur,
                              command=self._update_widget_states)
        self._blur_widgets.append(cb_blur)
        self._blur_widgets.append(self._grid_combo(
            lf, 1, "blur_kind", self.var_blur_kind,
            values=["gaussian", "median", "box"]))
        self._blur_widgets.append(self._grid_entry(lf, 2, "blur_ksize", self.var_blur_ksize))

        cb_sg = self._check(lf, 3, "Aplicar Savitzky-Golay",
                            self.var_apply_savgol,
                            command=self._update_widget_states)
        self._savgol_widgets.append(cb_sg)
        self._savgol_widgets.append(self._grid_entry(lf, 4, "savgol_window", self.var_savgol_window))
        self._savgol_widgets.append(self._grid_entry(lf, 5, "savgol_polyorder", self.var_savgol_polyorder))

        self._grid_combo(
            lf, 6, "band_selection", self.var_band_selection,
            values=["None", "pca", "kbest"]
        ).bind("<<ComboboxSelected>>", lambda e: self._update_widget_states())

        self._band_pca_widgets.append(self._grid_entry(lf, 7, "pca_components", self.var_pca_components))
        self._band_pca_widgets.append(self._grid_entry(lf, 8, "pca_variance_retained", self.var_pca_variance))
        self._band_kbest_widgets.append(self._grid_entry(lf, 9, "kbest_k", self.var_kbest_k))

        self._check(lf, 10, "use_scaler (StandardScaler)", self.var_use_scaler)

    def _build_postproc_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Pós-processamento")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(1, weight=1)

        cb_maj = self._check(lf, 0, "Filtro de maioria",
                             self.var_pp_majority,
                             command=self._update_widget_states)
        self._pp_majority_widgets.append(cb_maj)
        self._pp_majority_widgets.append(self._grid_entry(lf, 1, "majority_size", self.var_pp_majority_size))

        cb_morph = self._check(lf, 2, "Fechamento morfológico",
                               self.var_pp_morph,
                               command=self._update_widget_states)
        self._pp_morph_widgets.append(cb_morph)
        self._pp_morph_widgets.append(self._grid_entry(lf, 3, "closing_kernel", self.var_pp_closing_kernel))

        self._check(lf, 4, "Preencher buracos", self.var_pp_fill_holes)
        self._grid_entry(lf, 5, "min_area (px)", self.var_pp_min_area)

        cb_circ = self._check(lf, 6, "Filtro de circularidade",
                              self.var_pp_circ_filter,
                              command=self._update_widget_states)
        self._pp_circ_widgets.append(cb_circ)
        self._pp_circ_widgets.append(self._grid_entry(lf, 7, "circularity_threshold", self.var_pp_circ_thresh))

    def _build_save_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Salvamento")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)

        self._check(lf, 0, "Salvar cubo segmentado (.npy)", self.var_save_masked_cube)
        self._check(lf, 1, "Salvar máscara (.png) e metadados (.json)", self.var_save_crops)
        self._check(lf, 2, "Sobrescrever resultados existentes", self.var_overwrite)

    # ---- Painel direito (progresso + preview + console) --------------------
    def _build_right_panel(self, parent: ttk.Frame) -> None:
        # ---------- Progresso ----------
        prog_frame = ttk.LabelFrame(parent, text="Progresso")
        prog_frame.pack(fill=tk.X, padx=4, pady=(4, 2))

        ttk.Label(prog_frame, text="Arquivo atual:").pack(anchor="w", padx=6, pady=(6, 0))
        ttk.Label(prog_frame, textvariable=self.var_current_file,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=6)

        self.progressbar = ttk.Progressbar(
            prog_frame, orient="horizontal", mode="determinate",
            variable=self.var_progress, maximum=100.0)
        self.progressbar.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(prog_frame, textvariable=self.var_status).pack(anchor="w", padx=6, pady=(0, 6))

        # ---------- Botões ----------
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=4, pady=2)
        self.btn_process = ttk.Button(btn_frame, text="▶ Processar", command=self._on_process)
        self.btn_process.pack(side=tk.LEFT, padx=2)
        self.btn_stop = ttk.Button(btn_frame, text="■ Parar", command=self._on_stop, state="disabled")
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        # ---------- Preview (carrossel) ----------
        preview_frame = ttk.LabelFrame(parent, text="Máscaras Geradas")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 2))

        self.lbl_preview = ttk.Label(
            preview_frame, text="Nenhuma máscara ainda.",
            anchor="center", background="#1e1e1e", foreground="lightgray")
        self.lbl_preview.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        nav_frame = ttk.Frame(preview_frame)
        nav_frame.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.btn_prev = ttk.Button(nav_frame, text="◀ Anterior",
                                   command=self._prev_mask, state="disabled")
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        self.btn_next = ttk.Button(nav_frame, text="Próxima ▶",
                                   command=self._next_mask, state="disabled")
        self.btn_next.pack(side=tk.LEFT, padx=2)
        ttk.Label(nav_frame, textvariable=self.var_mask_counter).pack(side=tk.RIGHT, padx=6)

        # ---------- Console (compacto) ----------
        console_frame = ttk.LabelFrame(parent, text="Console")
        console_frame.pack(fill=tk.X, padx=4, pady=(2, 4))
        self.console = Console(console_frame, height=6)

        # Guarda referência ao frame pra podermos redimensionar a imagem
        self._preview_frame = preview_frame

        if not _HAS_PIL:
            self.console.log("⚠️ Pillow não instalado — preview desabilitado. "
                             "Rode: pip install Pillow")

    # =========================================================================
    # File pickers
    # =========================================================================
    def _pick_root_dir(self):
        d = filedialog.askdirectory(initialdir=self.var_root_dir.get() or ".")
        if d:
            self.var_root_dir.set(d)

    def _pick_out_dir(self):
        d = filedialog.askdirectory(initialdir=self.var_out_root.get() or ".")
        if d:
            self.var_out_root.set(d)

    def _add_json(self):
        paths = filedialog.askopenfilenames(
            title="Selecione JSON(s) de assinaturas de folha",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")])
        for p in paths:
            if p not in self.json_paths:
                self.json_paths.append(p)
        self._refresh_json_listboxes()

    def _add_json_bg(self):
        paths = filedialog.askopenfilenames(
            title="Selecione JSON(s) de assinaturas de background",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")])
        for p in paths:
            if p not in self.json_paths_bg:
                self.json_paths_bg.append(p)
        self._refresh_json_listboxes()

    def _remove_json(self):
        for idx in reversed(self.lb_json.curselection()):
            del self.json_paths[idx]
        self._refresh_json_listboxes()

    def _remove_json_bg(self):
        for idx in reversed(self.lb_json_bg.curselection()):
            del self.json_paths_bg[idx]
        self._refresh_json_listboxes()

    def _refresh_json_listboxes(self):
        self.lb_json.delete(0, tk.END)
        for p in self.json_paths:
            self.lb_json.insert(tk.END, os.path.basename(p))
        self.lb_json_bg.delete(0, tk.END)
        for p in self.json_paths_bg:
            self.lb_json_bg.insert(tk.END, os.path.basename(p))

    # =========================================================================
    # Enable/disable dinâmico
    # =========================================================================
    def _on_method_change(self):
        self._update_widget_states()

    def _set_state(self, widgets, enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in widgets:
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    def _update_widget_states(self):
        method = self.var_method.get()

        # KMeans
        km_on = method in ("kmeans", "kmeans_guiado")
        self._set_state(self._km_widgets, km_on)
        if km_on and len(self._km_widgets) > 1:
            try:
                self._km_widgets[1].configure(state=("normal" if method == "kmeans" else "disabled"))
            except Exception:
                pass

        # RF
        rf_on = method == "rf"
        self._set_state(self._rf_widgets, rf_on)

        # Otsu
        otsu_on = method == "otsu"
        self._set_state(self._otsu_widgets, otsu_on)
        if otsu_on:
            use_ndvi = self.var_otsu_use_ndvi.get()
            self._set_state(self._otsu_ndvi_widgets, use_ndvi)
            self._set_state(self._otsu_band_widgets, not use_ndvi)

        # Assinaturas
        sig_on = method in ("kmeans_guiado", "rf")
        self._set_state(self._sig_widgets, sig_on)
        bg_on = method == "rf"
        self._set_state(self._bg_widgets, bg_on)

        # Blur — sempre habilitado, mas sub-campos só quando checkbox ligado
        self._set_state(self._blur_widgets, True)
        if not self.var_apply_blur.get():
            self._set_state(self._blur_widgets[1:], False)

        # Savgol — idem
        self._set_state(self._savgol_widgets, True)
        if not self.var_apply_savgol.get():
            self._set_state(self._savgol_widgets[1:], False)

        # Band selection
        bs = self.var_band_selection.get()
        self._set_state(self._band_pca_widgets, bs == "pca")
        self._set_state(self._band_kbest_widgets, bs == "kbest")

        # Pós-processamento
        self._set_state(self._pp_majority_widgets, True)
        if not self.var_pp_majority.get():
            self._set_state(self._pp_majority_widgets[1:], False)
        self._set_state(self._pp_morph_widgets, True)
        if not self.var_pp_morph.get():
            self._set_state(self._pp_morph_widgets[1:], False)
        self._set_state(self._pp_circ_widgets, True)
        if not self.var_pp_circ_filter.get():
            self._set_state(self._pp_circ_widgets[1:], False)

    # =========================================================================
    # Config
    # =========================================================================
    def _build_config(self) -> Dict[str, Any]:
        config = deepcopy(DEFAULT_CONFIG)

        config["root_dir"] = self.var_root_dir.get()
        config["out_root"] = self.var_out_root.get()
        config["json_path"] = list(self.json_paths)
        config["json_path_background"] = list(self.json_paths_bg)
        config["method"] = self.var_method.get()

        pca_str = self.var_km_pca.get().strip()
        try:
            pca_val = int(pca_str) if pca_str else None
        except ValueError:
            pca_val = None

        config["kmeans_params"] = {
            "n_clusters": self.var_km_n_clusters.get(),
            "n_init": self.var_km_n_init.get(),
            "max_iter": self.var_km_max_iter.get(),
            "random_state": self.var_km_random_state.get(),
            "pca_components": pca_val,
        }

        cw = self.var_rf_class_weight.get()
        config["rf_params"] = {
            "n_estimators": self.var_rf_n_estimators.get(),
            "max_depth": self.var_rf_max_depth.get(),
            "min_samples_split": self.var_rf_min_samples_split.get(),
            "random_state": self.var_rf_random_state.get(),
            "class_weight": None if cw == "None" else cw,
            "n_jobs": self.var_rf_n_jobs.get(),
        }
        config["rf_prob_threshold"] = self.var_rf_prob_threshold.get()
        config["n_background_samples"] = self.var_n_background_samples.get()

        config["otsu_params"] = {
            "use_ndvi": self.var_otsu_use_ndvi.get(),
            "band_nm": self.var_otsu_band_nm.get(),
            "red_nm": self.var_otsu_red_nm.get(),
            "nir_nm": self.var_otsu_nir_nm.get(),
            "invert": self.var_otsu_invert.get(),
        }

        config["apply_blur"] = self.var_apply_blur.get()
        config["blur_kind"] = self.var_blur_kind.get()
        config["blur_ksize"] = self.var_blur_ksize.get()
        config["apply_savgol"] = self.var_apply_savgol.get()
        config["savgol_window"] = self.var_savgol_window.get()
        config["savgol_polyorder"] = self.var_savgol_polyorder.get()

        bs = self.var_band_selection.get()
        config["band_selection"] = None if bs == "None" else bs
        config["band_selection_params"] = {
            "pca_components": self.var_pca_components.get(),
            "pca_variance_retained": self.var_pca_variance.get(),
            "kbest_k": self.var_kbest_k.get(),
        }
        config["use_scaler"] = self.var_use_scaler.get()

        config["postprocess"] = {
            "majority_filter": self.var_pp_majority.get(),
            "majority_size": self.var_pp_majority_size.get(),
            "morphological_closing": self.var_pp_morph.get(),
            "closing_kernel": self.var_pp_closing_kernel.get(),
            "fill_holes": self.var_pp_fill_holes.get(),
            "min_area": self.var_pp_min_area.get(),
            "circularity_filter": self.var_pp_circ_filter.get(),
            "circularity_threshold": self.var_pp_circ_thresh.get(),
        }

        config["save_masked_cube"] = self.var_save_masked_cube.get()
        config["save_crops"] = self.var_save_crops.get()
        config["overwrite"] = self.var_overwrite.get()

        return config

    # =========================================================================
    # Processamento
    # =========================================================================
    def _on_process(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self.console.log("⚠️ Já existe processamento em andamento.")
            return

        try:
            config = self._build_config()
        except Exception as e:
            self.console.log(f"❌ Config inválida: {e}")
            return

        root_dir = config["root_dir"]
        if not os.path.isdir(root_dir):
            self.console.log(f"❌ Pasta de entrada inválida: {root_dir}")
            return

        npys = [
            os.path.join(root_dir, f)
            for f in sorted(os.listdir(root_dir))
            if f.lower().endswith(".npy")
        ]
        if not npys:
            self.console.log("❌ Nenhum arquivo .npy encontrado na pasta de entrada.")
            return

        os.makedirs(config["out_root"], exist_ok=True)

        # Limpa carrossel para esta nova execução
        self._masks_list.clear()
        self._current_mask_idx = 0
        self.var_mask_counter.set("0 / 0")
        self.lbl_preview.configure(image="", text="Aguardando primeira máscara...")
        self._current_photo = None
        self.btn_prev.configure(state="disabled")
        self.btn_next.configure(state="disabled")

        self._stop_event.clear()
        self.var_progress.set(0.0)
        self.var_status.set("Iniciando...")
        self.var_current_file.set("—")
        self._set_running(True)

        self.console.log(f"▶ Iniciando: {len(npys)} imagens | método={config['method']}")

        self._worker_thread = threading.Thread(
            target=self._worker,
            args=(config, npys),
            daemon=True,
        )
        self._worker_thread.start()

    def _on_stop(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self.console.log("⏹️ Solicitando parada...")
            self._stop_event.set()

    def _set_running(self, running: bool):
        if running:
            self.btn_process.configure(state="disabled")
            self.btn_stop.configure(state="normal")
        else:
            self.btn_process.configure(state="normal")
            self.btn_stop.configure(state="disabled")

    # ---- Worker (thread separada) ------------------------------------------
    def _worker(self, config: dict, npys: List[str]):
        try:
            signatures = None
            signatures_bg = None
            method = config["method"]

            if method in ("kmeans_guiado", "rf"):
                self._log_queue.put("Carregando assinaturas de folha...")
                signatures, _ = load_signatures_from_json(config["json_path"])
                self._log_queue.put(
                    f"  → {signatures.shape[0]} assinaturas | {signatures.shape[1]} bandas")

            if method == "rf":
                bg_paths = [p for p in config.get("json_path_background", []) if os.path.exists(p)]
                if bg_paths:
                    self._log_queue.put(f"Carregando background de {len(bg_paths)} arquivo(s)...")
                    signatures_bg, _ = load_signatures_from_json(bg_paths)
                    self._log_queue.put(
                        f"  → {signatures_bg.shape[0]} assinaturas de background")
                else:
                    self._log_queue.put(
                        "⚠️ Nenhum background válido. RF usará amostragem aleatória.")

            total = len(npys)
            for i, npy in enumerate(npys, 1):
                if self._stop_event.is_set():
                    self._log_queue.put("⏹️ Processamento interrompido pelo usuário.")
                    break

                fname = os.path.basename(npy)
                self._progress_queue.put(("status", f"({i}/{total}) {fname}"))
                self._progress_queue.put(("file", fname))
                self._progress_queue.put(("progress", 100.0 * (i - 1) / total))

                try:
                    self._process_one_image(npy, signatures, signatures_bg, config)
                except Exception as e:
                    logger.exception(f"Erro em {fname}: {e}")
                    self._log_queue.put(f"❌ Erro em {fname}: {e}")

                self._progress_queue.put(("progress", 100.0 * i / total))

            if not self._stop_event.is_set():
                self._progress_queue.put(("status", "Concluído."))
                self._log_queue.put("✅ Processamento concluído.")
            else:
                self._progress_queue.put(("status", "Interrompido."))

        except Exception as e:
            logger.exception("Falha geral no worker")
            self._log_queue.put(f"❌ Falha geral: {e}")
        finally:
            self._progress_queue.put(("done", None))

    # ---- Pipeline de uma imagem --------------------------------------------
    def _process_one_image(self, npy_path, signatures, signatures_bg, config):
        base = os.path.splitext(os.path.basename(npy_path))[0]
        safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base.replace(":", ""))

        out_root = config["out_root"]
        npy_out = os.path.join(out_root, f"{safe_base}_segmentado.npy")
        mask_png = os.path.join(out_root, f"{safe_base}_mascara.png")

        if not config.get("overwrite", False) and os.path.exists(npy_out):
            logger.info(f"Já existe, pulando: {os.path.basename(npy_path)}")
            # Mesmo pulando, adiciona ao carrossel se a máscara existir
            if os.path.exists(mask_png):
                self._mask_queue.put(mask_png)
            return

        cube = np.load(npy_path).astype(np.float32)
        if cube.ndim != 3:
            logger.error(f"Formato inesperado: {cube.shape}")
            return

        h, w, b = cube.shape
        logger.info(f"Processando {os.path.basename(npy_path)} -> {h}x{w}x{b}")

        # --- Pré-processamento ---
        proc = cube.copy()
        if config.get("apply_blur", False):
            proc = apply_blur(proc, config.get("blur_kind", "gaussian"),
                              config.get("blur_ksize", 3))
        if config.get("apply_savgol", False):
            proc = apply_savgol(proc, config.get("savgol_window", 11),
                                config.get("savgol_polyorder", 2))

        # --- Seleção de bandas ---
        band_selector = None
        if config.get("band_selection") is not None:
            proc, band_selector = select_bands(
                proc, signatures, config["band_selection"],
                config.get("band_selection_params", {}))

        # --- Classificação ---
        method = config["method"]
        if method == "kmeans":
            mask = classify_kmeans(
                proc,
                n_clusters=config["kmeans_params"].get("n_clusters", 2),
                kmeans_params=config["kmeans_params"])
        elif method == "kmeans_guiado":
            if signatures is None or len(signatures) == 0:
                raise ValueError("kmeans_guiado requer assinaturas de folha")
            sig_use = band_selector.transform(signatures) if band_selector else signatures
            mask = classify_kmeans_guiado(proc, sig_use,
                                          kmeans_params=config["kmeans_params"])
        elif method == "rf":
            if signatures is None or len(signatures) == 0:
                raise ValueError("RF requer assinaturas de folha")
            X_train, y_train = prepare_training_data(
                cube=proc,
                signatures_target=signatures,
                signatures_background=signatures_bg,
                n_background=config.get("n_background_samples", 5000),
                band_selector=band_selector)
            mask = classify_rf(
                proc, X_train, y_train,
                rf_params=config["rf_params"],
                prob_threshold=config.get("rf_prob_threshold", 0.5),
                use_scaler=config.get("use_scaler", True))
        elif method == "otsu":
            mask = classify_otsu(
                proc,
                otsu_params=config.get("otsu_params", {}),
                cube_start_nm=config.get("cube_start_nm", 400.0),
                cube_end_nm=config.get("cube_end_nm", 1000.0))
        else:
            raise ValueError(f"Método desconhecido: {method}")

        # --- Pós-processamento ---
        mask = postprocess_mask(mask, config)

        # --- Salvamento ---
        if not config.get("save_crops", True):
            return

        cv2.imwrite(mask_png, (mask * 255).astype(np.uint8))
        self._mask_queue.put(mask_png)   # <-- alimenta o carrossel

        if config.get("save_masked_cube", True):
            masked_cube = cube.copy()
            masked_cube[mask == 0, :] = 0.0
            np.save(npy_out, masked_cube)

        metadata = {
            "arquivo_origem": os.path.basename(npy_path),
            "data_criacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metodo": config.get("method"),
            "shape_original": list(cube.shape),
            "pixels_selecionados": int(mask.sum()),
            "fundo_preenchido_com": 0,
            "parametros": {
                "method": config.get("method"),
                "kmeans_params": config.get("kmeans_params"),
                "rf_params": config.get("rf_params"),
                "otsu_params": config.get("otsu_params"),
                "rf_prob_threshold": config.get("rf_prob_threshold"),
                "band_selection": config.get("band_selection"),
                "use_scaler": config.get("use_scaler"),
                "postprocess": config.get("postprocess", {}),
            },
        }
        with open(os.path.join(out_root, f"{safe_base}.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Finalizado: {base} ({mask.sum()} pixels)")

    # =========================================================================
    # Carrossel
    # =========================================================================
    def _prev_mask(self):
        if self._current_mask_idx > 0:
            self._current_mask_idx -= 1
            self._show_current_mask()

    def _next_mask(self):
        if self._current_mask_idx < len(self._masks_list) - 1:
            self._current_mask_idx += 1
            self._show_current_mask()

    def _show_current_mask(self):
        if not self._masks_list:
            self.lbl_preview.configure(image="", text="Nenhuma máscara ainda.")
            self.var_mask_counter.set("0 / 0")
            self.btn_prev.configure(state="disabled")
            self.btn_next.configure(state="disabled")
            return

        path = self._masks_list[self._current_mask_idx]
        self.var_mask_counter.set(f"{self._current_mask_idx + 1} / {len(self._masks_list)}")

        self.btn_prev.configure(state=("normal" if self._current_mask_idx > 0 else "disabled"))
        self.btn_next.configure(state=("normal"
                                       if self._current_mask_idx < len(self._masks_list) - 1
                                       else "disabled"))

        if not _HAS_PIL:
            self.lbl_preview.configure(text=os.path.basename(path))
            return

        try:
            img = Image.open(path)

            # Ajusta tamanho disponível
            self._preview_frame.update_idletasks()
            avail_w = max(self._preview_frame.winfo_width() - 30, 200)
            avail_h = max(self._preview_frame.winfo_height() - 80, 150)

            img.thumbnail((avail_w, avail_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self._current_photo = photo  # evita GC
            self.lbl_preview.configure(image=photo, text="")
        except Exception as e:
            self.lbl_preview.configure(image="", text=f"Erro ao abrir: {e}")
            self._current_photo = None

    def _on_new_mask(self, path: str):
        """Chamado na thread principal quando uma nova máscara é gerada."""
        was_at_end = (self._current_mask_idx >= len(self._masks_list) - 1)
        self._masks_list.append(path)
        if was_at_end:
            self._current_mask_idx = len(self._masks_list) - 1
            self._show_current_mask()
        else:
            # Só atualiza o contador
            self.var_mask_counter.set(
                f"{self._current_mask_idx + 1} / {len(self._masks_list)}")
            self.btn_next.configure(state="normal")

    # =========================================================================
    # Polling
    # =========================================================================
    def _poll_queues(self):
        if not self._alive:
            return

        # Logs
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self.console.log(msg)
        except queue.Empty:
            pass

        # Progresso / status / arquivo atual
        try:
            while True:
                kind, value = self._progress_queue.get_nowait()
                if kind == "progress":
                    self.var_progress.set(value)
                elif kind == "status":
                    self.var_status.set(value)
                elif kind == "file":
                    self.var_current_file.set(value)
                elif kind == "done":
                    self._set_running(False)
        except queue.Empty:
            pass

        # Novas máscaras para o carrossel
        try:
            while True:
                path = self._mask_queue.get_nowait()
                self._on_new_mask(path)
        except queue.Empty:
            pass

        self.parent.after(120, self._poll_queues)

    # =========================================================================
    # Cleanup
    # =========================================================================
    def destroy(self):
        self._alive = False
        try:
            logging.getLogger().removeHandler(self._handler)
        except Exception:
            pass