# annotation_tab/main.py
"""
Aba de anotação - versão refatorada com estrutura modular.
"""

import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional

import numpy as np  
from PIL import Image

from .models.class_manager import ClassManager
from .services.persistence import AnnotationPersistence
from .services.folder_loader import FolderLoader
from .services.yolo_loader import YoloLoader
from .ui.annotator import Annotator
from .ui.class_ui import ClassUI
from .ui.console import Console


class AnnotationTab:
    """Aba de anotação principal."""

    def __init__(self, parent: tk.Frame):
        self.parent = parent
        self._after_id: Optional[str] = None

        # Core components
        self.class_manager = ClassManager()
        self.persistence = AnnotationPersistence(self.class_manager, self._log)
        self.yolo_loader = YoloLoader(self.class_manager, self._log)

        # State
        self.pasta_origem = tk.StringVar()
        self.pasta_destino = tk.StringVar()
        self.modo_inicial = tk.StringVar(value="bbox")
        self.action_var = tk.StringVar()

        self.annotator: Optional[Annotator] = None
        self._arquivos_pendentes: List[str] = []
        self._indice_atual: int = 0
        self._total_imagens: int = 0
        self._classe_para_ativar: Optional[int] = None

        self._criar_widgets()
        self._carregar_configuracao_auto()
        self._after_id = self.parent.after(100, self._poll_log_queue)
        self.parent.bind("<Destroy>", self._on_destroy)

    # ------------------------------------------------------------------------
    # Inicialização da UI
    # ------------------------------------------------------------------------
    def _criar_widgets(self) -> None:
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Painel esquerdo - imagem
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        canvas_img = tk.Canvas(left_frame, borderwidth=0, highlightthickness=0)
        v_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=canvas_img.yview)
        h_scroll = ttk.Scrollbar(left_frame, orient=tk.HORIZONTAL, command=canvas_img.xview)
        canvas_img.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        canvas_img.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        self.fig_container = ttk.Frame(canvas_img)
        canvas_img.create_window((0, 0), window=self.fig_container, anchor="nw")
        self.fig_container.bind("<Configure>", lambda e: canvas_img.configure(scrollregion=canvas_img.bbox("all")))

        # Painel direito - controles
        right_frame = ttk.Frame(main_frame, width=450)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 5), pady=5)
        right_frame.pack_propagate(False)

        canvas_ctrl = tk.Canvas(right_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=canvas_ctrl.yview)
        canvas_ctrl.configure(yscrollcommand=scrollbar.set)
        canvas_ctrl.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        controls_inner = ttk.Frame(canvas_ctrl)
        canvas_ctrl.create_window((0, 0), window=controls_inner, anchor="nw")

        def _config_canvas(event):
            canvas_ctrl.configure(scrollregion=canvas_ctrl.bbox("all"))
            canvas_ctrl.itemconfig(1, width=event.width)

        controls_inner.bind("<Configure>", lambda e: canvas_ctrl.configure(scrollregion=canvas_ctrl.bbox("all")))
        canvas_ctrl.bind("<Configure>", _config_canvas)
        controls_inner.bind("<MouseWheel>", lambda e: canvas_ctrl.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # ----- Seções -----
        self._criar_secao_pastas(controls_inner)
        self._criar_secao_config_classes(controls_inner)
        self._criar_secao_modo(controls_inner)
        self._criar_secao_classes(controls_inner)
        self._criar_secao_acoes(controls_inner)
        self._criar_secao_log(controls_inner)
        self._criar_secao_atalhos(controls_inner)

        # Status bar
        status_frame = ttk.Frame(self.parent)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Pronto")
        self.progress_var = tk.StringVar(value="")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        progress_label = ttk.Label(status_frame, textvariable=self.progress_var, relief=tk.SUNKEN, width=20)
        progress_label.pack(side=tk.RIGHT)

        self._log("🟢 Aba de Anotação iniciada. Configure as pastas e classes.")

    def _criar_secao_pastas(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Pastas")
        frame.pack(fill=tk.X, pady=5)

        ttk.Label(frame, text="Origem:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.pasta_origem, width=20).grid(row=0, column=1, padx=2)
        ttk.Button(frame, text="Browse", command=self._selecionar_pasta_origem, width=8).grid(row=0, column=2)
        self.btn_carregar = ttk.Button(frame, text="Carregar",
                                       command=self._iniciar_anotacao, width=8)
        self.btn_carregar.grid(row=0, column=3, padx=(5, 0))

        ttk.Label(frame, text="Destino:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.pasta_destino, width=20).grid(row=1, column=1, padx=2)
        ttk.Button(frame, text="Browse", command=self._selecionar_pasta_destino, width=8).grid(row=1, column=2)

    def _criar_secao_config_classes(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Configuração de Classes")
        frame.pack(fill=tk.X, pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="💾 Salvar Classes",
                   command=self._salvar_configuracao_classes, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📂 Carregar Classes",
                   command=self._carregar_configuracao_classes, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🔄 Resetar Classes",
                   command=self._resetar_classes, width=15).pack(side=tk.LEFT, padx=2)

    def _criar_secao_modo(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Modo de Anotação")
        frame.pack(fill=tk.X, pady=5)

        self.rb_bbox = ttk.Radiobutton(frame, text="Bounding Box", variable=self.modo_inicial,
                                       value="bbox", command=self._radio_mode_changed)
        self.rb_bbox.pack(anchor=tk.W)
        self.rb_polygon = ttk.Radiobutton(frame, text="Polígono", variable=self.modo_inicial,
                                          value="polygon", command=self._radio_mode_changed)
        self.rb_polygon.pack(anchor=tk.W)

    def _criar_secao_classes(self, parent) -> None:
        self.class_ui = ClassUI(parent, self.class_manager, self._log)
        self.class_ui.set_activate_callback(self._ativar_classe_ui)

    def _criar_secao_acoes(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Ações durante anotação")
        frame.pack(fill=tk.X, pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=2)
        self.btn_salvar = ttk.Button(btn_frame, text="Salvar e Próximo (N)",
                                     command=self._acao_salvar, width=18)
        self.btn_salvar.pack(side=tk.LEFT, padx=2)
        self.btn_pular = ttk.Button(btn_frame, text="Pular (P)",
                                    command=self._acao_pular, width=12)
        self.btn_pular.pack(side=tk.LEFT, padx=2)
        self.btn_sair = ttk.Button(btn_frame, text="Sair (Q)",
                                   command=self._acao_sair, width=12)
        self.btn_sair.pack(side=tk.LEFT, padx=2)
        ttk.Button(frame, text="Desfazer (Z)", command=self._acao_desfazer, width=20).pack(pady=2)

    def _criar_secao_log(self, parent) -> None:
        self.console = Console(parent, height=8)

    def _criar_secao_atalhos(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Atalhos")
        frame.pack(fill=tk.X, pady=5)
        txt = ("1-9,0: selecionar classe\n"
               "Z: desfazer último ponto (polígono) ou última anotação\n"
               "M: alternar modo (BBox/Polígono)\n"
               "N/Enter: salvar e próxima\n"
               "P: pular imagem\n"
               "Q: sair\n"
               "ESC: cancelar desenho do polígono")
        ttk.Label(frame, text=txt, justify=tk.LEFT).pack(anchor=tk.W)

    # ------------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if hasattr(self, 'console'):
            self.console.log(msg)
        else:
            print(msg)
        self.status_var.set(msg[:60])

    def _poll_log_queue(self) -> None:
        self._after_id = self.parent.after(100, self._poll_log_queue)

    def _on_destroy(self, event) -> None:
        if self._after_id:
            try:
                self.parent.after_cancel(self._after_id)
            except:
                pass
            self._after_id = None

    # ------------------------------------------------------------------------
    # Seleção de pastas
    # ------------------------------------------------------------------------
    def _selecionar_pasta_origem(self) -> None:
        pasta = filedialog.askdirectory()
        if pasta:
            self.pasta_origem.set(pasta)
            self._log(f"📂 Pasta de origem: {pasta}")

    def _selecionar_pasta_destino(self) -> None:
        pasta = filedialog.askdirectory()
        if pasta:
            self.pasta_destino.set(pasta)
            self._log(f"📂 Pasta de destino: {pasta}")
            # Tenta carregar configuração automaticamente
            if self.persistence.load_classes(pasta):
                self.class_ui.update()

    # ------------------------------------------------------------------------
    # Gerenciamento de classes (UI)
    # ------------------------------------------------------------------------
    def _ativar_classe_ui(self, cid: int) -> None:
        """Callback do ClassUI quando uma classe é ativada."""
        if self.annotator is not None:
            self.annotator.set_class(cid)
        else:
            self._classe_para_ativar = cid
        self._log(f"🔵 Classe {cid} ativada")

    # ------------------------------------------------------------------------
    # Configuração de classes (persistência)
    # ------------------------------------------------------------------------
    def _carregar_configuracao_auto(self) -> None:
        destino = self.pasta_destino.get()
        if destino and os.path.exists(destino):
            if self.persistence.load_classes(destino):
                self.class_ui.update()
                return
        if self.persistence.load_classes(None):
            self.class_ui.update()
            return
        self._log("ℹ️ Nenhuma configuração de classes encontrada. Adicione classes manualmente.")

    def _salvar_configuracao_classes(self) -> None:
        destino = self.pasta_destino.get()
        if destino:
            if self.persistence.save_classes(destino):
                messagebox.showinfo("Sucesso", f"Configuração salva em:\n{os.path.join(destino, 'classes_config.json')}")
        else:
            if self.persistence.save_classes(None):
                messagebox.showinfo("Sucesso", f"Configuração salva em:\nclasses_config.json")
        self.class_ui.update()

    def _carregar_configuracao_classes(self) -> None:
        arquivo = filedialog.askopenfilename(
            title="Selecionar arquivo de configuração de classes",
            filetypes=[("Arquivos JSON", "*.json"), ("Todos os arquivos", "*.*")],
            initialdir=self.pasta_destino.get() if self.pasta_destino.get() else os.getcwd()
        )
        if not arquivo:
            return
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'classes' not in data:
                self._log("❌ Arquivo inválido: não contém 'classes'")
                messagebox.showerror("Erro", "Arquivo inválido!")
                return
            self.class_manager.from_dict(data)
            self.class_ui.update()
            self._log(f"✅ Configuração carregada de: {os.path.basename(arquivo)}")
            messagebox.showinfo("Sucesso", f"Configuração carregada com sucesso!\nClasses: {len(data['classes'])}")
        except Exception as e:
            self._log(f"❌ Erro ao carregar arquivo: {e}")
            messagebox.showerror("Erro", f"Erro ao carregar arquivo:\n{e}")

    def _resetar_classes(self) -> None:
        if not self.class_manager.classes:
            self._log("ℹ️ Não há classes para resetar.")
            return
        resposta = messagebox.askyesno("Confirmar", "Remover TODAS as classes?")
        if resposta:
            for cid in list(self.class_manager.classes.keys()):
                self.class_manager.remove_class(cid)
            self.class_ui.update()
            self._log("🔄 Todas as classes foram removidas.")

    # ------------------------------------------------------------------------
    # Modo
    # ------------------------------------------------------------------------
    def _radio_mode_changed(self) -> None:
        modo = self.modo_inicial.get()
        if self.annotator is not None:
            self.annotator.set_mode(modo)
            self._log(f"Modo alterado via radiobutton para: {modo.upper()}")

    def _on_mode_changed(self, modo: str) -> None:
        if modo in ('bbox', 'polygon'):
            self.modo_inicial.set(modo)

    # ------------------------------------------------------------------------
    # Ações do usuário
    # ------------------------------------------------------------------------
    def _acao_salvar(self) -> None:
        if self.annotator is not None:
            self.annotator.set_acao('salvar')
        else:
            self._log("⚠️ Nenhuma anotação em andamento.")

    def _acao_pular(self) -> None:
        if self.annotator is not None:
            self.annotator.set_acao('pular')
        else:
            self._log("⚠️ Nenhuma anotação em andamento.")

    def _acao_sair(self) -> None:
        if self.annotator is not None:
            self.annotator.set_acao('sair')
        else:
            self._log("⚠️ Nenhuma anotação em andamento.")

    def _acao_desfazer(self) -> None:
        if self.annotator is not None:
            self.annotator.undo()
        else:
            self._log("⚠️ Nenhuma anotação em andamento.")

    # ------------------------------------------------------------------------
    # Iniciar anotação (com threading)
    # ------------------------------------------------------------------------
    def _iniciar_anotacao(self) -> None:
        origem = self.pasta_origem.get()
        destino = self.pasta_destino.get()
        if not origem or not destino:
            messagebox.showerror("Erro", "Selecione as pastas de origem e destino.")
            return
        if not self.class_manager.classes:
            messagebox.showwarning("Aviso", "Adicione pelo menos uma classe.")
            return

        self.btn_carregar.config(state=tk.DISABLED)
        self.status_var.set("Carregando imagens...")

        loader = FolderLoader(origem, destino, self._on_imagens_carregadas, self._log)
        loader.start()

    def _on_imagens_carregadas(self, arquivos: List[str]) -> None:
        self.btn_carregar.config(state=tk.NORMAL)
        self._arquivos_pendentes = arquivos
        self._total_imagens = len(arquivos)
        self._indice_atual = 0

        if not arquivos:
            self._log("ℹ️ Todas as imagens já foram anotadas!")
            messagebox.showinfo("Info", "Todas as imagens já foram anotadas!")
            self.status_var.set("Pronto")
            self.progress_var.set("")
            return

        self._log(f"📷 {len(arquivos)} imagens pendentes encontradas.")
        self._processar_proxima_imagem()

    # ------------------------------------------------------------------------
    # Processamento sequencial de imagens
    # ------------------------------------------------------------------------
    def _processar_proxima_imagem(self):
        if self._indice_atual >= len(self._arquivos_pendentes):
            self._finalizar_sessao()
            return

        imagem_path = self._arquivos_pendentes[self._indice_atual]
        nome_arquivo = os.path.basename(imagem_path)
        self.progress_var.set(f"Imagem {self._indice_atual+1}/{self._total_imagens}")

        self._log(f"📷 Anotando: {nome_arquivo} ({self._indice_atual+1}/{self._total_imagens})")

        # Limpa anotações anteriores
        self.class_manager.clear_annotations()

        try:
            from PIL import Image
            img = Image.open(imagem_path).convert('RGB')
            img_array = np.array(img)
        except Exception as e:
            self._log(f"❌ Erro ao abrir {nome_arquivo}: {e}")
            self._indice_atual += 1
            self._processar_proxima_imagem()
            return

        for widget in self.fig_container.winfo_children():
            widget.destroy()

        self.class_manager.set_image_shape(img_array.shape)

        # Tenta carregar anotações existentes
        destino = self.pasta_destino.get()
        if self.persistence.load_annotations(imagem_path, destino):
            self._log(f"   📂 Anotações carregadas do JSON.")
        elif self.yolo_loader.load_annotations(imagem_path, destino):
            self._log(f"   📂 Anotações carregadas de arquivos YOLO.")
        else:
            self._log(f"   ℹ️ Nenhuma anotação existente encontrada.")

        # Cria Annotator
        self.annotator = Annotator(
            self.fig_container,
            imagem_path,
            img_array,
            self.class_manager,
            self._log,
            self.action_var,
            on_update_callback=self.class_ui.update,
            on_mode_change_callback=self._on_mode_changed,
            mode=self.modo_inicial.get()
        )

        if self._classe_para_ativar is not None:
            self.annotator.set_class(self._classe_para_ativar)
            self._classe_para_ativar = None

        self.annotator.fig.canvas.draw_idle()

        # Aguarda ação do usuário
        self.action_var.set("")
        self.parent.wait_variable(self.action_var)
        acao = self.action_var.get()

        if acao == 'salvar':
            self._salvar_anotacao(imagem_path)
            self._indice_atual += 1
        elif acao == 'pular':
            self._log(f"⏭️ Pulada: {nome_arquivo}")
            self._indice_atual += 1
        elif acao == 'sair':
            self._log("🚪 Sessão encerrada pelo usuário")
            self.annotator = None
            self._finalizar_sessao()
            return

        self.annotator = None
        self.class_ui.update()
        # Processa próxima
        self._processar_proxima_imagem()

    def _salvar_anotacao(self, imagem_path: str) -> None:
        destino = self.pasta_destino.get()
        counts = self.class_manager.get_annotation_count()
        total_objetos = sum(c['total'] for c in counts.values())
        self._log(f"   📊 Resumo: {total_objetos} objetos")
        for cid, info in counts.items():
            self._log(f"      {info['name']}: {info['bboxes']} bboxes, {info['polygons']} polygons")

        if self.persistence.save_annotations(imagem_path, destino):
            self._log(f"   ✅ JSON salvo em annotations/")
        else:
            self._log(f"   ⚠️ Erro ao salvar JSON")

        n_bbox, n_poly = self.persistence.export_yolo(imagem_path, destino)
        self._log(f"💾 Salva concluída: {os.path.basename(imagem_path)} (BBox: {n_bbox}, Polygon: {n_poly})")

    def _finalizar_sessao(self) -> None:
        self._log("✅ Anotação finalizada.")
        messagebox.showinfo("Concluído", "Anotação finalizada!")
        self.status_var.set("Pronto")
        self.progress_var.set("")
        self.btn_carregar.config(state=tk.NORMAL)


# ------------------------------------------------------------------------
# Função de criação para integração com o notebook
# ------------------------------------------------------------------------
def create_annotation_tab(parent: tk.Frame) -> AnnotationTab:
    """Cria e retorna uma instância da aba de anotação."""
    return AnnotationTab(parent)


# ------------------------------------------------------------------------
# Teste independente
# ------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Teste - Aba de Anotação")
    root.geometry("1300x800")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    tab = create_annotation_tab(notebook)
    notebook.add(tab, text="Anotação")

    root.mainloop()