"""
Aba de Embedding.

Fluxo:
  1. Modelo: treina novo (nas imagens segmentadas) OU carrega de disco (.pt)
  2. Aplica o modelo nas imagens corrigidas -> pseudo-RGB (carrossel + disco)
  3. Grafico dos filtros aparece no painel fixo ao lado do carrossel

Estrutura de saida:
  out_root/train/   -> modelo, filtros, plot (somente quando treina)
  out_root/apply/   -> pseudo_rgb/ + (opcional) filtros do modelo carregado
"""

import os
import re
import queue
import logging
import threading
from copy import deepcopy
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, filedialog

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# --- Imports do pacote annotation_tab ---
from ...models.embedding.config import DEFAULT_CONFIG
from ...models.embedding.search import automated_search
from ...services.embedding.data_loader import carregar_espectros_da_pasta
from ...services.embedding.trainer import train_embedding_system
from ...services.embedding.inference import (
    cube_to_pseudo_rgb,
    get_learned_filters,
    load_model_from_checkpoint,
)
from ...services.embedding.visualization import plot_filters, save_pseudo_rgb_png

from ..anotacao.console import Console


logger = logging.getLogger("EmbeddingTab")


# =============================================================================
# Handler de logging que envia mensagens para a fila
# =============================================================================
class _QueueLogHandler(logging.Handler):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


# =============================================================================
# Fabrica
# =============================================================================
def create_embedding_tab(parent_frame: tk.Frame) -> "EmbeddingTab":
    return EmbeddingTab(parent_frame)


# =============================================================================
# Classe principal
# =============================================================================
class EmbeddingTab:
    """Aba de embedding com treino OU carregamento + aplicacao."""

    def __init__(self, parent_frame: tk.Frame):
        self.parent = parent_frame

        # --- Threading ---
        self._stop_event = threading.Event()
        self._log_queue = queue.Queue()
        self._progress_queue = queue.Queue()
        self._pseudo_rgb_queue = queue.Queue()      # PNGs das pseudo-RGB (carrossel)
        self._filters_plot_queue = queue.Queue()    # caminho do PNG dos filtros
        self._worker_thread: Optional[threading.Thread] = None
        self._alive = True

        # --- Carrossel ---
        self._rgb_list: List[str] = []
        self._current_idx: int = 0
        self._current_rgb_photo = None
        self._filter_photo = None

        # --- Handler de log ---
        self._handler = _QueueLogHandler(self._log_queue)
        self._handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S')
        )
        logging.getLogger().addHandler(self._handler)

        # --- Variaveis ---
        self._init_variables()

        # --- UI ---
        self._build_ui()
        self._update_widget_states()

        # --- Polling ---
        self._poll_queues()

        self.console.log("Aba de embedding pronta.")

    # =========================================================================
    # Variaveis
    # =========================================================================
    def _init_variables(self) -> None:
        C = DEFAULT_CONFIG

        # --- Caminhos base ---
        self.var_out_root = tk.StringVar(value=C["out_root"])

        # --- Modo ---
        self.var_mode = tk.StringVar(value=C["mode"])            # "train" ou "load"
        self.var_segmented_dir = tk.StringVar(value=C["segmented_dir"])
        self.var_model_path = tk.StringVar(value=C["model_path"])
        self.var_corrected_dir = tk.StringVar(value=C["corrected_dir"])

        # --- Estrutura ---
        self.var_n_channels = tk.IntVar(value=C["n_channels"])
        self.var_n_gaussians = tk.IntVar(value=C["n_gaussians"])
        self.var_hidden_dim = tk.IntVar(value=C["hidden_dim"])
        self.var_width_min = tk.StringVar(
            value="" if C["width_min"] is None else str(C["width_min"]))

        # --- Loss ---
        self.var_alpha = tk.DoubleVar(value=C["alpha"])
        self.var_beta = tk.DoubleVar(value=C["beta"])

        # --- Busca ---
        self.var_use_search = tk.BooleanVar(value=C["use_auto_search"])
        self.var_search_width_fracs = tk.StringVar(
            value=", ".join(str(x) for x in C["search_width_min_fracs"]))
        self.var_search_gammas = tk.StringVar(
            value=", ".join(str(x) for x in C["search_gammas"]))
        self.var_search_epochs = tk.IntVar(value=C["search_epochs"])
        self.var_search_max_pixels = tk.IntVar(value=C["search_max_pixels"])
        self.var_save_search_candidates = tk.BooleanVar(
            value=C["save_search_candidates"])

        # --- Sem busca ---
        self.var_gamma = tk.DoubleVar(value=C["gamma"])

        # --- Treino ---
        self.var_epochs = tk.IntVar(value=C["epochs"])
        self.var_batch_size = tk.IntVar(value=C["batch_size"])
        self.var_lr = tk.DoubleVar(value=C["lr"])
        self.var_max_pixels = tk.IntVar(value=C["max_pixels_per_image"])
        self.var_seed = tk.IntVar(value=C["seed"])

        # --- Salvamento ---
        self.var_save_pseudo_npy = tk.BooleanVar(value=C["save_pseudo_rgb_npy"])
        self.var_overwrite = tk.BooleanVar(value=C["overwrite"])

        # --- Status ---
        self.var_status = tk.StringVar(value="Pronto.")
        self.var_progress = tk.DoubleVar(value=0.0)
        self.var_current_file = tk.StringVar(value="—")
        self.var_rgb_counter = tk.StringVar(value="0 / 0")

        # --- Listas de widgets para enable/disable ---
        self._train_mode_widgets: List[tk.Widget] = []
        self._load_mode_widgets: List[tk.Widget] = []
        self._search_sub_widgets: List[tk.Widget] = []
        self._fixed_gamma_widgets: List[tk.Widget] = []

    # =========================================================================
    # Construcao da UI
    # =========================================================================
    def _build_ui(self) -> None:
        paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=4)

        params_inner = self._make_scrollable(left)
        self._build_params_panel(params_inner)
        self._build_right_panel(right)

    def _make_scrollable(self, parent: tk.Widget) -> ttk.Frame:
        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(canvas_window, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _on_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        return inner

    # ---- Painel de parametros ----------------------------------------------
    def _build_params_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        r = 0
        self._build_paths_section(parent, r); r += 1
        self._build_mode_section(parent, r); r += 1
        self._build_embedding_section(parent, r); r += 1
        self._build_loss_section(parent, r); r += 1
        self._build_search_section(parent, r); r += 1
        self._build_train_section(parent, r); r += 1
        self._build_save_section(parent, r)

    # ---- helpers de widgets -------------------------------------------------
    def _grid_entry(self, parent, row, label, var, width=12):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
        parent.columnconfigure(1, weight=1)
        return e

    def _check(self, parent, row, label, var, command=None):
        cb = ttk.Checkbutton(parent, text=label, variable=var, command=command)
        cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=1)
        return cb

    # ---- Secoes -------------------------------------------------------------
    def _build_paths_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Caminhos")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(1, weight=1)

        ttk.Label(lf, text="Pasta de imagens CORRIGIDAS").grid(
            row=0, column=0, sticky="w", padx=4, pady=(6, 1))
        ttk.Entry(lf, textvariable=self.var_corrected_dir).grid(
            row=0, column=1, sticky="ew", padx=4, pady=(6, 1))
        ttk.Button(lf, text="…", width=3,
                   command=self._pick_corrected_dir).grid(row=0, column=2, padx=2, pady=(6, 1))

        ttk.Label(lf, text="Pasta de saida").grid(
            row=1, column=0, sticky="w", padx=4, pady=1)
        ttk.Entry(lf, textvariable=self.var_out_root).grid(
            row=1, column=1, sticky="ew", padx=4, pady=1)
        ttk.Button(lf, text="…", width=3,
                   command=self._pick_out_dir).grid(row=1, column=2, padx=2, pady=1)

        ttk.Label(lf, text="(corrigidas = entrada da aplicacao; saida = out_root)",
                  foreground="gray").grid(
            row=2, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 6))

    def _build_mode_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Modelo")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(1, weight=1)

        # --- Radio buttons de modo ---
        ttk.Radiobutton(
            lf, text="Treinar novo modelo (nas imagens segmentadas)",
            value="train", variable=self.var_mode,
            command=self._update_widget_states,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(6, 1))

        ttk.Radiobutton(
            lf, text="Carregar modelo de disco (.pt)",
            value="load", variable=self.var_mode,
            command=self._update_widget_states,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=6, pady=(1, 4))

        # --- Campo: pasta de segmentadas (so modo train) ---
        lbl_seg = ttk.Label(lf, text="Pasta de imagens SEGMENTADAS")
        lbl_seg.grid(row=2, column=0, sticky="w", padx=4, pady=1)
        e_seg = ttk.Entry(lf, textvariable=self.var_segmented_dir)
        e_seg.grid(row=2, column=1, sticky="ew", padx=4, pady=1)
        b_seg = ttk.Button(lf, text="…", width=3,
                           command=self._pick_segmented_dir)
        b_seg.grid(row=2, column=2, padx=2, pady=1)
        self._train_mode_widgets += [lbl_seg, e_seg, b_seg]

        # --- Campo: caminho do modelo .pt (so modo load) ---
        lbl_m = ttk.Label(lf, text="Modelo .pt")
        lbl_m.grid(row=3, column=0, sticky="w", padx=4, pady=1)
        e_m = ttk.Entry(lf, textvariable=self.var_model_path)
        e_m.grid(row=3, column=1, sticky="ew", padx=4, pady=1)
        b_m = ttk.Button(lf, text="…", width=3, command=self._pick_model_path)
        b_m.grid(row=3, column=2, padx=2, pady=1)
        self._load_mode_widgets += [lbl_m, e_m, b_m]

        ttk.Label(lf, text="Formato aceito: .pt (salvo por esta aba)",
                  foreground="gray").grid(
            row=4, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 6))

    def _build_embedding_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Estrutura do embedding")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(1, weight=1)

        w = self._grid_entry(lf, 0, "n_channels", self.var_n_channels)
        self._train_mode_widgets.append(w)
        w = self._grid_entry(lf, 1, "n_gaussians", self.var_n_gaussians)
        self._train_mode_widgets.append(w)
        w = self._grid_entry(lf, 2, "hidden_dim", self.var_hidden_dim)
        self._train_mode_widgets.append(w)

        lbl = ttk.Label(lf, text="width_min (vazio = auto)")
        lbl.grid(row=3, column=0, sticky="w", padx=4, pady=1)
        e = ttk.Entry(lf, textvariable=self.var_width_min)
        e.grid(row=3, column=1, sticky="ew", padx=4, pady=1)
        self._train_mode_widgets += [lbl, e]

    def _build_loss_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Loss")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(1, weight=1)

        w = self._grid_entry(lf, 0, "alpha (amplitude)", self.var_alpha)
        self._train_mode_widgets.append(w)
        w = self._grid_entry(lf, 1, "beta (derivada)", self.var_beta)
        self._train_mode_widgets.append(w)

    def _build_search_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Busca automatica (treino)")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(1, weight=1)

        cb = self._check(lf, 0, "Usar busca automatica",
                         self.var_use_search, command=self._update_widget_states)
        self._train_mode_widgets.append(cb)

        w = self._grid_entry(lf, 1, "width_min_fracs (csv)",
                             self.var_search_width_fracs, width=22)
        self._train_mode_widgets.append(w)
        self._search_sub_widgets.append(w)

        w = self._grid_entry(lf, 2, "gammas (csv)",
                             self.var_search_gammas, width=22)
        self._train_mode_widgets.append(w)
        self._search_sub_widgets.append(w)

        w = self._grid_entry(lf, 3, "search_epochs", self.var_search_epochs)
        self._train_mode_widgets.append(w)
        self._search_sub_widgets.append(w)

        w = self._grid_entry(lf, 4, "search_max_pixels", self.var_search_max_pixels)
        self._train_mode_widgets.append(w)
        self._search_sub_widgets.append(w)

        w = self._check(lf, 5, "Salvar candidatos em disco",
                        self.var_save_search_candidates)
        self._train_mode_widgets.append(w)
        self._search_sub_widgets.append(w)

        sep = ttk.Separator(lf, orient="horizontal")
        sep.grid(row=6, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        self._train_mode_widgets.append(sep)

        lbl = ttk.Label(lf, text="Se desligada, usa gamma fixo:")
        lbl.grid(row=7, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 2))
        self._train_mode_widgets.append(lbl)

        w = self._grid_entry(lf, 8, "gamma", self.var_gamma)
        self._train_mode_widgets.append(w)
        self._fixed_gamma_widgets.append(w)

    def _build_train_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Treino final")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        lf.columnconfigure(1, weight=1)

        for i, (lbl, var) in enumerate([
            ("epochs", self.var_epochs),
            ("batch_size", self.var_batch_size),
            ("learning_rate", self.var_lr),
            ("max_pixels_per_image", self.var_max_pixels),
            ("seed", self.var_seed),
        ]):
            w = self._grid_entry(lf, i, lbl, var)
            self._train_mode_widgets.append(w)

    def _build_save_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Salvamento")
        lf.grid(row=row, column=0, sticky="ew", padx=4, pady=4)

        self._check(lf, 0, "Salvar pseudo-RGB tambem como .npy",
                    self.var_save_pseudo_npy)
        self._check(lf, 1, "Sobrescrever resultados existentes",
                    self.var_overwrite)

    # ---- Painel direito ----------------------------------------------------
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
        ttk.Label(prog_frame, textvariable=self.var_status).pack(
            anchor="w", padx=6, pady=(0, 6))

        # ---------- Botoes ----------
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=4, pady=2)
        self.btn_process = ttk.Button(btn_frame, text="▶ Processar",
                                      command=self._on_process)
        self.btn_process.pack(side=tk.LEFT, padx=2)
        self.btn_stop = ttk.Button(btn_frame, text="■ Parar",
                                   command=self._on_stop, state="disabled")
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        # ---------- Area central: carrossel (esq) + grafico (dir) ----------
        middle = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        middle.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 2))

        # Lado esquerdo: carrossel
        preview_frame = ttk.LabelFrame(middle, text="Pseudo-RGB (aplicacao nas corrigidas)")
        middle.add(preview_frame, weight=3)

        self.lbl_preview = ttk.Label(
            preview_frame, text="Nenhuma pseudo-RGB ainda.",
            anchor="center", background="#1e1e1e", foreground="lightgray")
        self.lbl_preview.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        nav_frame = ttk.Frame(preview_frame)
        nav_frame.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.btn_prev = ttk.Button(nav_frame, text="◀ Anterior",
                                   command=self._prev_rgb, state="disabled")
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        self.btn_next = ttk.Button(nav_frame, text="Próxima ▶",
                                   command=self._next_rgb, state="disabled")
        self.btn_next.pack(side=tk.LEFT, padx=2)
        ttk.Label(nav_frame, textvariable=self.var_rgb_counter).pack(
            side=tk.RIGHT, padx=6)

        # Lado direito: grafico fixo
        filters_frame = ttk.LabelFrame(middle, text="Filtros Aprendidos")
        middle.add(filters_frame, weight=2)

        self.lbl_filters = ttk.Label(
            filters_frame, text="Aguardando treino...",
            anchor="center", background="#1e1e1e", foreground="lightgray")
        self.lbl_filters.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._preview_frame = preview_frame
        self._filters_container = filters_frame

        # ---------- Console ----------
        console_frame = ttk.LabelFrame(parent, text="Console")
        console_frame.pack(fill=tk.X, padx=4, pady=(2, 4))
        self.console = Console(console_frame, height=6)

        if not _HAS_PIL:
            self.console.log("⚠️ Pillow nao instalado -- preview desabilitado. "
                             "Rode: pip install Pillow")

    # =========================================================================
    # File pickers
    # =========================================================================
    def _pick_corrected_dir(self):
        d = filedialog.askdirectory(initialdir=self.var_corrected_dir.get() or ".")
        if d:
            self.var_corrected_dir.set(d)

    def _pick_segmented_dir(self):
        d = filedialog.askdirectory(initialdir=self.var_segmented_dir.get() or ".")
        if d:
            self.var_segmented_dir.set(d)

    def _pick_out_dir(self):
        d = filedialog.askdirectory(initialdir=self.var_out_root.get() or ".")
        if d:
            self.var_out_root.set(d)

    def _pick_model_path(self):
        p = filedialog.askopenfilename(
            title="Selecione o modelo (.pt)",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("Todos", "*.*")],
            initialdir=os.path.dirname(self.var_model_path.get() or "."),
        )
        if p:
            self.var_model_path.set(p)

    # =========================================================================
    # Enable/disable dinamico
    # =========================================================================
    def _set_state(self, widgets, enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in widgets:
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    def _update_widget_states(self):
        mode = self.var_mode.get()
        is_train = (mode == "train")

        # Widgets exclusivos de cada modo
        self._set_state(self._train_mode_widgets, is_train)
        self._set_state(self._load_mode_widgets, not is_train)

        # Sub-estado da busca (so relevante em train)
        if is_train:
            use_search = self.var_use_search.get()
            self._set_state(self._search_sub_widgets, use_search)
            self._set_state(self._fixed_gamma_widgets, not use_search)

    # =========================================================================
    # Config
    # =========================================================================
    def _parse_float_list(self, s: str):
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return [float(p) for p in parts]

    def _build_config(self) -> Dict[str, Any]:
        config = deepcopy(DEFAULT_CONFIG)

        config["out_root"] = self.var_out_root.get()
        config["mode"] = self.var_mode.get()
        config["segmented_dir"] = self.var_segmented_dir.get()
        config["model_path"] = self.var_model_path.get()
        config["corrected_dir"] = self.var_corrected_dir.get()

        config["n_channels"] = self.var_n_channels.get()
        config["n_gaussians"] = self.var_n_gaussians.get()
        config["hidden_dim"] = self.var_hidden_dim.get()

        wm_str = self.var_width_min.get().strip()
        try:
            config["width_min"] = float(wm_str) if wm_str else None
        except ValueError:
            config["width_min"] = None

        config["alpha"] = self.var_alpha.get()
        config["beta"] = self.var_beta.get()

        config["use_auto_search"] = self.var_use_search.get()
        config["search_width_min_fracs"] = self._parse_float_list(
            self.var_search_width_fracs.get())
        config["search_gammas"] = self._parse_float_list(
            self.var_search_gammas.get())
        config["search_epochs"] = self.var_search_epochs.get()
        config["search_max_pixels"] = self.var_search_max_pixels.get()
        config["save_search_candidates"] = self.var_save_search_candidates.get()

        config["gamma"] = self.var_gamma.get()

        config["epochs"] = self.var_epochs.get()
        config["batch_size"] = self.var_batch_size.get()
        config["lr"] = self.var_lr.get()
        config["max_pixels_per_image"] = self.var_max_pixels.get()
        config["seed"] = self.var_seed.get()

        config["save_pseudo_rgb_npy"] = self.var_save_pseudo_npy.get()
        config["overwrite"] = self.var_overwrite.get()

        return config

    # =========================================================================
    # Processamento
    # =========================================================================
    def _on_process(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self.console.log("⚠️ Ja existe processamento em andamento.")
            return

        try:
            config = self._build_config()
        except Exception as e:
            self.console.log(f"❌ Config invalida: {e}")
            return

        # ---- Validacoes ----
        corrected_dir = config["corrected_dir"]
        if not os.path.isdir(corrected_dir):
            self.console.log(f"❌ Pasta de corrigidas invalida: {corrected_dir}")
            return

        corrected_npys = [
            os.path.join(corrected_dir, f)
            for f in sorted(os.listdir(corrected_dir))
            if f.lower().endswith(".npy")
        ]
        if not corrected_npys:
            self.console.log("❌ Nenhum .npy encontrado na pasta de corrigidas.")
            return

        if not config["out_root"]:
            self.console.log("❌ Pasta de saida nao definida.")
            return

        if config["mode"] == "train":
            seg_dir = config["segmented_dir"]
            if not os.path.isdir(seg_dir):
                self.console.log(f"❌ Pasta de segmentadas invalida: {seg_dir}")
                return
        else:  # load
            model_path = config["model_path"]
            if not os.path.isfile(model_path):
                self.console.log(f"❌ Modelo .pt nao encontrado: {model_path}")
                return
            if not model_path.lower().endswith(".pt"):
                self.console.log(f"❌ Extensao invalida (esperado .pt): {model_path}")
                return

        os.makedirs(config["out_root"], exist_ok=True)

        # ---- Reset UI ----
        self._rgb_list.clear()
        self._current_idx = 0
        self.var_rgb_counter.set("0 / 0")
        self.lbl_preview.configure(image="", text="Aguardando processamento...")
        self._current_rgb_photo = None
        self.btn_prev.configure(state="disabled")
        self.btn_next.configure(state="disabled")

        self.lbl_filters.configure(image="", text="Aguardando treino...")
        self._filter_photo = None

        self._stop_event.clear()
        self.var_progress.set(0.0)
        self.var_status.set("Iniciando...")
        self.var_current_file.set("—")
        self._set_running(True)

        self.console.log(
            f"▶ Modo={config['mode']} | "
            f"{len(corrected_npys)} corrigidas para aplicar | "
            f"saida={config['out_root']}"
        )

        self._worker_thread = threading.Thread(
            target=self._worker,
            args=(config, corrected_npys),
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

    # ---- Worker ------------------------------------------------------------
    def _worker(self, config: dict, corrected_npys: List[str]):
        try:
            torch.manual_seed(config["seed"])
            np.random.seed(config["seed"])

            out_root = config["out_root"]
            train_dir = os.path.join(out_root, "train")
            apply_dir = os.path.join(out_root, "apply")
            pseudo_dir = os.path.join(apply_dir, "pseudo_rgb")
            os.makedirs(apply_dir, exist_ok=True)
            os.makedirs(pseudo_dir, exist_ok=True)

            # ================================================================
            # FASE 1 — obter o modelo (treinar ou carregar)
            # ================================================================
            if config["mode"] == "train":
                # ---- 1.1 Carregar espectros das segmentadas ----
                self._progress_queue.put(("status", "Carregando segmentadas..."))
                self._log_queue.put("[info] carregando espectros das segmentadas")

                spectra_all = carregar_espectros_da_pasta(
                    config["segmented_dir"],
                    max_pixels_por_amostra=config["max_pixels_per_image"],
                    seed=config["seed"],
                    log_cb=lambda m: self._log_queue.put(m),
                )

                if self._stop_event.is_set():
                    self._progress_queue.put(("status", "Interrompido."))
                    return

                # ---- 1.2 Busca automatica ou hiperparametros fixos ----
                if config["use_auto_search"]:
                    self._log_queue.put("[info] iniciando busca automatica")

                    def _cand_cb(idx, total, cand):
                        self._progress_queue.put(("progress", 100.0 * idx / total))
                        self._progress_queue.put(("status",
                            f"Busca: candidato {idx}/{total} "
                            f"(wf={cand['width_min_frac']:.3f}, "
                            f"gamma={cand['gamma']:.1f})"))

                    width_min_escolhido, gamma_escolhido, _ = automated_search(
                        spectra_all,
                        n_channels=config["n_channels"],
                        n_gaussians=config["n_gaussians"],
                        hidden_dim=config["hidden_dim"],
                        alpha=config["alpha"],
                        beta=config["beta"],
                        width_min_fracs=config["search_width_min_fracs"],
                        gammas=config["search_gammas"],
                        epochs_busca=config["search_epochs"],
                        max_pixels_busca=config["search_max_pixels"],
                        batch_size=config["batch_size"],
                        lr=config["lr"],
                        seed=config["seed"],
                        log_cb=lambda m: self._log_queue.put(m),
                        candidate_cb=_cand_cb,
                    )
                else:
                    width_min_escolhido = config["width_min"]
                    gamma_escolhido = config["gamma"]
                    self._log_queue.put(
                        f"[info] busca desligada -- gamma={gamma_escolhido}, "
                        f"width_min={width_min_escolhido}")

                if self._stop_event.is_set():
                    self._progress_queue.put(("status", "Interrompido."))
                    return

                # ---- 1.3 Treino final ----
                self._progress_queue.put(("status", "Treinando modelo..."))
                self._log_queue.put(
                    f"[info] treino final: {config['epochs']} epocas | "
                    f"gamma={gamma_escolhido} | width_min={width_min_escolhido}"
                )

                def _epoch_cb(epoch, total, metrics):
                    self._progress_queue.put(("progress", 100.0 * epoch / total))
                    self._progress_queue.put(("status",
                        f"Treino: epoca {epoch}/{total}"))

                model = train_embedding_system(
                    spectra_all,
                    n_channels=config["n_channels"],
                    n_gaussians=config["n_gaussians"],
                    hidden_dim=config["hidden_dim"],
                    alpha=config["alpha"],
                    beta=config["beta"],
                    gamma=gamma_escolhido,
                    width_min=width_min_escolhido,
                    epochs=config["epochs"],
                    batch_size=config["batch_size"],
                    lr=config["lr"],
                    verbose=False,
                    log_cb=lambda m: self._log_queue.put(m),
                    epoch_cb=_epoch_cb,
                )

                if self._stop_event.is_set():
                    self._progress_queue.put(("status", "Interrompido."))
                    return

                # ---- 1.4 Salvar em out_root/train/ ----
                os.makedirs(train_dir, exist_ok=True)
                model_path_out = os.path.join(train_dir, "spectral_embedding.pt")
                torch.save({
                    "state_dict":  model.state_dict(),
                    "n_bands":     spectra_all.shape[1],
                    "n_channels":  config["n_channels"],
                    "n_gaussians": config["n_gaussians"],
                    "hidden_dim":  config["hidden_dim"],
                    "width_min":   model.filter_bank.width_min,
                    "gamma":       gamma_escolhido,
                }, model_path_out)
                self._log_queue.put(f"[info] modelo salvo em {model_path_out}")

                filters = get_learned_filters(model)
                filters_npy = os.path.join(train_dir, "learned_filters.npy")
                np.save(filters_npy, filters)
                self._log_queue.put(f"[info] filtros salvos em {filters_npy}")

                plot_path = os.path.join(train_dir, "learned_filters_plot.png")
                plot_filters(filters, plot_path)
                self._log_queue.put(f"[info] plot dos filtros salvo em {plot_path}")

            else:
                # ============================================================
                # Modo LOAD: carregar modelo de disco
                # ============================================================
                self._progress_queue.put(("status", "Carregando modelo..."))
                self._log_queue.put(
                    f"[info] carregando modelo de disco: {config['model_path']}")

                model, meta = load_model_from_checkpoint(config["model_path"])
                self._log_queue.put(
                    f"[info] modelo carregado: n_bands={meta['n_bands']}, "
                    f"n_channels={meta['n_channels']}, "
                    f"n_gaussians={meta['n_gaussians']}, "
                    f"gamma={meta.get('gamma')}"
                )

                if self._stop_event.is_set():
                    self._progress_queue.put(("status", "Interrompido."))
                    return

                # Plota os filtros do modelo carregado -> apply/
                filters = get_learned_filters(model)
                filters_npy = os.path.join(apply_dir, "learned_filters.npy")
                np.save(filters_npy, filters)

                plot_path = os.path.join(apply_dir, "learned_filters_plot.png")
                plot_filters(filters, plot_path)
                self._log_queue.put(f"[info] plot dos filtros salvo em {plot_path}")

            # Manda o plot para a UI (em ambos os modos)
            self._filters_plot_queue.put(plot_path)

            if self._stop_event.is_set():
                self._progress_queue.put(("status", "Interrompido."))
                return

            # ================================================================
            # FASE 2 — aplicar modelo nas corrigidas
            # ================================================================
            self._progress_queue.put(("status", "Aplicando nas corrigidas..."))
            self._log_queue.put("[info] aplicando modelo nas imagens corrigidas")

            total = len(corrected_npys)
            n_bands_model = model.filter_bank.n_bands

            for i, npy in enumerate(corrected_npys, 1):
                if self._stop_event.is_set():
                    self._log_queue.put("⏹️ Aplicacao interrompida.")
                    break

                fname = os.path.basename(npy)
                self._progress_queue.put(("file", fname))
                self._progress_queue.put(("status", f"Aplicando: {i}/{total}"))
                self._progress_queue.put(("progress", 100.0 * (i - 1) / total))

                base = os.path.splitext(fname)[0]
                safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)

                try:
                    cube = np.load(npy)
                    if cube.ndim != 3:
                        self._log_queue.put(
                            f"[aviso] {fname}: shape {cube.shape} nao e (H,W,n_bands) "
                            f"-- pulando")
                        continue

                    if cube.shape[-1] != n_bands_model:
                        self._log_queue.put(
                            f"[aviso] {fname}: {cube.shape[-1]} bandas != "
                            f"{n_bands_model} esperadas pelo modelo -- pulando")
                        continue

                    pseudo_rgb = cube_to_pseudo_rgb(model, cube)

                    out_png = os.path.join(pseudo_dir, f"{safe_base}_pseudo_rgb.png")
                    save_pseudo_rgb_png(pseudo_rgb, out_png)

                    if config["save_pseudo_rgb_npy"]:
                        out_npy = os.path.join(pseudo_dir, f"{safe_base}_pseudo_rgb.npy")
                        np.save(out_npy, pseudo_rgb)

                    self._pseudo_rgb_queue.put(out_png)
                    self._log_queue.put(f"[info] pseudo-RGB: {safe_base}")

                except Exception as e:
                    logger.exception(f"Erro em {fname}: {e}")
                    self._log_queue.put(f"❌ Erro em {fname}: {e}")

                self._progress_queue.put(("progress", 100.0 * i / total))

            if not self._stop_event.is_set():
                self._progress_queue.put(("status", "Concluido."))
                self._log_queue.put("✅ Processamento concluido.")
            else:
                self._progress_queue.put(("status", "Interrompido."))

        except Exception as e:
            logger.exception("Falha geral no worker")
            self._log_queue.put(f"❌ Falha geral: {e}")
        finally:
            self._progress_queue.put(("done", None))

    # =========================================================================
    # Carrossel
    # =========================================================================
    def _prev_rgb(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._show_current_rgb()

    def _next_rgb(self):
        if self._current_idx < len(self._rgb_list) - 1:
            self._current_idx += 1
            self._show_current_rgb()

    def _show_current_rgb(self):
        if not self._rgb_list:
            self.lbl_preview.configure(image="", text="Nenhuma pseudo-RGB ainda.")
            self.var_rgb_counter.set("0 / 0")
            self.btn_prev.configure(state="disabled")
            self.btn_next.configure(state="disabled")
            return

        path = self._rgb_list[self._current_idx]
        self.var_rgb_counter.set(f"{self._current_idx + 1} / {len(self._rgb_list)}")

        self.btn_prev.configure(
            state=("normal" if self._current_idx > 0 else "disabled"))
        self.btn_next.configure(
            state=("normal" if self._current_idx < len(self._rgb_list) - 1
                   else "disabled"))

        if not _HAS_PIL:
            self.lbl_preview.configure(text=os.path.basename(path))
            return

        try:
            img = Image.open(path)

            self._preview_frame.update_idletasks()
            avail_w = max(self._preview_frame.winfo_width() - 30, 200)
            avail_h = max(self._preview_frame.winfo_height() - 80, 150)

            img.thumbnail((avail_w, avail_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self._current_rgb_photo = photo
            self.lbl_preview.configure(image=photo, text="")
        except Exception as e:
            self.lbl_preview.configure(image="", text=f"Erro ao abrir: {e}")
            self._current_rgb_photo = None

    def _on_new_rgb(self, path: str):
        was_at_end = (self._current_idx >= len(self._rgb_list) - 1)
        self._rgb_list.append(path)
        if was_at_end:
            self._current_idx = len(self._rgb_list) - 1
            self._show_current_rgb()
        else:
            self.var_rgb_counter.set(
                f"{self._current_idx + 1} / {len(self._rgb_list)}")
            self.btn_next.configure(state="normal")

    def _on_filters_plot(self, path: str):
        if not _HAS_PIL:
            self.lbl_filters.configure(text=os.path.basename(path))
            return

        try:
            img = Image.open(path)

            self._filters_container.update_idletasks()
            avail_w = max(self._filters_container.winfo_width() - 20, 200)
            avail_h = max(self._filters_container.winfo_height() - 20, 100)

            img.thumbnail((avail_w, avail_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self._filter_photo = photo
            self.lbl_filters.configure(image=photo, text="")
        except Exception as e:
            self.lbl_filters.configure(image="",
                                       text=f"Erro ao abrir grafico: {e}")
            self._filter_photo = None

    # =========================================================================
    # Polling
    # =========================================================================
    def _poll_queues(self):
        if not self._alive:
            return

        try:
            while True:
                msg = self._log_queue.get_nowait()
                self.console.log(msg)
        except queue.Empty:
            pass

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

        try:
            while True:
                path = self._pseudo_rgb_queue.get_nowait()
                self._on_new_rgb(path)
        except queue.Empty:
            pass

        try:
            while True:
                path = self._filters_plot_queue.get_nowait()
                self._on_filters_plot(path)
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