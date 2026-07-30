# annotation_tab/ui/class_ui.py
"""
Componente UI para gerenciamento de classes.
"""

import tkinter as tk
from tkinter import ttk, simpledialog
from typing import Optional, Callable
from ..models.class_manager import ClassManager


class ClassUI:
    """UI para gerenciar classes de anotação."""

    def __init__(self, parent: ttk.Frame, class_manager: ClassManager,
                 log_callback: Optional[Callable] = None):
        self.parent = parent
        self.class_manager = class_manager
        self.log = log_callback or (lambda msg: None)
        self._activate_callback: Optional[Callable[[int], None]] = None

        self._create_widgets()
        self._update_list()

    def _create_widgets(self) -> None:
        frame = ttk.LabelFrame(self.parent, text="Gerenciar Classes")
        frame.pack(fill=tk.X, pady=5)

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.X, pady=2)

        self.lista_classes = tk.Listbox(list_frame, height=6, selectmode=tk.SINGLE)
        self.lista_classes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.lista_classes.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.lista_classes.config(yscrollcommand=scroll.set)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(btn_frame, text="Adicionar", command=self._adicionar, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remover", command=self._remover, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Renomear", command=self._renomear, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Ativar", command=self._ativar, width=10).pack(side=tk.LEFT, padx=2)

    def _update_list(self) -> None:
        self.lista_classes.delete(0, tk.END)
        for cid, info in sorted(self.class_manager.classes.items()):
            if info['is_background']:
                continue
            total = len(info['bboxes']) + len(info['polygons'])
            self.lista_classes.insert(tk.END, f"{cid}: {info['name']} ({info['color']}) [{total}]")

    def _adicionar(self) -> None:
        nome = simpledialog.askstring("Nova Classe", "Nome da classe:")
        if not nome:
            return
        cor = simpledialog.askstring("Cor", "Cor (ex: red, blue) ou vazio:")
        if cor:
            cores_validas = ['red','blue','green','orange','purple','cyan','magenta','lime',
                             'brown','pink','olive','teal','gold','coral','indigo','violet','darkgreen']
            if cor not in cores_validas:
                self.log(f"⚠️ Cor '{cor}' inválida, usando automática.")
                cor = None
        cid = self.class_manager.add_class(nome, color=cor)
        self._update_list()
        self.log(f"✅ Classe '{nome}' adicionada (ID {cid})")

    def _remover(self) -> None:
        selecao = self.lista_classes.curselection()
        if not selecao:
            self.log("⚠️ Selecione uma classe para remover.")
            return
        item = self.lista_classes.get(selecao[0])
        cid = int(item.split(':')[0])
        nome = self.class_manager.classes[cid]['name']
        self.class_manager.remove_class(cid)
        self._update_list()
        self.log(f"🗑️ Classe '{nome}' removida.")

    def _renomear(self) -> None:
        selecao = self.lista_classes.curselection()
        if not selecao:
            self.log("⚠️ Selecione uma classe para renomear.")
            return
        item = self.lista_classes.get(selecao[0])
        cid = int(item.split(':')[0])
        novo_nome = simpledialog.askstring("Renomear", "Novo nome:")
        if novo_nome:
            self.class_manager.rename_class(cid, novo_nome)
            self._update_list()
            self.log(f"✏️ Classe renomeada para '{novo_nome}'")

    def _ativar(self) -> None:
        selecao = self.lista_classes.curselection()
        if not selecao:
            self.log("⚠️ Selecione uma classe para ativar.")
            return
        item = self.lista_classes.get(selecao[0])
        cid = int(item.split(':')[0])
        self.log(f"🔵 Classe {cid} ativada")
        if self._activate_callback:
            self._activate_callback(cid)

    def set_activate_callback(self, callback: Callable[[int], None]) -> None:
        """Define callback para ativação de classe."""
        self._activate_callback = callback

    def update(self) -> None:
        """Atualiza a lista de classes."""
        self._update_list()