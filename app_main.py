import tkinter as tk
from tkinter import ttk

from tab_radiometria import RadiometricTab



class AppMain:
    def __init__(self, root):
        self.root = root
        root.title("SpectralImage")
        root.geometry("1400x900")

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # ========== ABA 1 ==========
        frame_rad = ttk.Frame(notebook)
        notebook.add(frame_rad, text="1. Pré-processamento FX10")
        self.tab_rad = RadiometricTab(frame_rad)


        self._center_window()

    def _center_window(self):
        """Centraliza a janela na tela"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')


if __name__ == "__main__":
    root = tk.Tk()
    app = AppMain(root)
    root.mainloop()