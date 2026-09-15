# annotation_tab/ui/console.py
"""
Componente de console de log.
"""
import time
import tkinter as tk
from tkinter import ttk
from typing import Optional


class Console:
    """Console de log com texto colorido e scroll."""

    def __init__(self, parent: tk.Widget, height: int = 8, width: int = 80):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Botão limpar
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill=tk.X)
        self.btn_clear = ttk.Button(btn_frame, text="Limpar Log", command=self.clear, width=12)
        self.btn_clear.pack(side=tk.RIGHT, padx=2)

        # Text widget
        self.text = tk.Text(self.frame, height=height, state='normal', wrap=tk.WORD,
                            bg='black', fg='lightgreen', font=('Consolas', 8))
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.config(yscrollcommand=scroll.set)

    def log(self, msg: str) -> None:
        """Adiciona mensagem ao console."""
        import time
        self.text.config(state='normal')
        self.text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.text.see(tk.END)
        self.text.config(state='disabled')

    def clear(self) -> None:
        """Limpa o console."""
        self.text.config(state='normal')
        self.text.delete(1.0, tk.END)
        self.text.config(state='disabled')