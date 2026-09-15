# annotation_tab/services/folder_loader.py
"""
Carrega lista de imagens em thread separada.
"""

import os
import threading
import logging
from typing import List, Callable, Optional, Set

EXTENSOES_VALIDAS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif', '.webp', '.heic')


class FolderLoader:
    """Carrega lista de imagens em thread separada e notifica quando pronto."""

    def __init__(self, origem: str, destino: str, callback: Callable[[List[str]], None],
                 log_callback: Optional[Callable] = None):
        self.origem = origem
        self.destino = destino
        self.callback = callback
        self.log = log_callback or (lambda msg: None)
        self.thread: Optional[threading.Thread] = None
        self.running: bool = False

    def start(self) -> None:
        if not os.path.isdir(self.origem):
            self.log("❌ Pasta de origem inválida.")
            return
        if not os.path.isdir(self.destino):
            self.log("❌ Pasta de destino inválida.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._load, daemon=True)
        self.thread.start()

    def _load(self) -> None:
        try:
            arquivos = []
            for root, dirs, files in os.walk(self.origem):
                for f in files:
                    if f.lower().endswith(EXTENSOES_VALIDAS):
                        arquivos.append(os.path.join(root, f))
            ja_feitos = self._get_annotated_images()
            pendentes = [f for f in arquivos if os.path.splitext(os.path.basename(f))[0] not in ja_feitos]
            pendentes.sort()
            self.callback(pendentes)
        except Exception as e:
            logger.error(f"Erro ao carregar arquivos: {e}")
            self.log(f"❌ Erro ao carregar arquivos: {e}")
        finally:
            self.running = False

    def _get_annotated_images(self) -> Set[str]:
        ja_feitas = set()
        for sub in ['annotations', 'labels/bbox', 'labels/polygon']:
            path = os.path.join(self.destino, sub)
            if os.path.isdir(path):
                for f in os.listdir(path):
                    if f.endswith(('.json', '.txt')):
                        ja_feitas.add(os.path.splitext(f)[0])
        return ja_feitas