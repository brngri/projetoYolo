# annotation_tab/services/yolo_loader.py
"""
Carregamento de anotações a partir de arquivos YOLO.
"""

import os
import logging
from typing import Optional, Callable, Dict
from ..models.class_manager import ClassManager
from .persistence import YoloExporter

logger = logging.getLogger("AnnotationTab.YoloLoader")


class YoloLoader:
    """Carrega anotações de arquivos YOLO (bbox e polygon)."""

    def __init__(self, class_manager: ClassManager, log_callback: Optional[Callable] = None):
        self.class_manager = class_manager
        self.log = log_callback or (lambda msg: None)

    def load_annotations(self, image_path: str, destino: str) -> bool:
        """Tenta carregar anotações dos arquivos YOLO."""
        nome_base = os.path.splitext(os.path.basename(image_path))[0]

        # Carrega mapeamento YOLO -> nome
        mapping_path = os.path.join(destino, 'yolo_classes.txt')
        yolo_to_name = YoloExporter.load_mapping(mapping_path)
        if not yolo_to_name:
            return False

        # Garante que as classes existam no class_manager
        yolo_to_cid = {}
        for yolo_id, nome in yolo_to_name.items():
            found = None
            for cid, info in self.class_manager.classes.items():
                if info['name'] == nome and not info['is_background']:
                    found = cid
                    break
            if found is None:
                cid = self.class_manager.add_class(nome)
            else:
                cid = found
            yolo_to_cid[yolo_id] = cid

        # Lê arquivos bbox e polygon
        bbox_path = os.path.join(destino, 'labels', 'bbox', nome_base + '.txt')
        poly_path = os.path.join(destino, 'labels', 'polygon', nome_base + '.txt')

        loaded = False
        h, w = self.class_manager._ny, self.class_manager._nx
        if h is None or w is None:
            return False

        # Carrega bboxes
        if os.path.exists(bbox_path):
            with open(bbox_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    yolo_id = int(parts[0])
                    if yolo_id not in yolo_to_cid:
                        continue
                    cid = yolo_to_cid[yolo_id]
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = (cx - bw/2) * w
                    x2 = (cx + bw/2) * w
                    y1 = (cy - bh/2) * h
                    y2 = (cy + bh/2) * h
                    self.class_manager.add_bbox(cid, x1, y1, x2, y2)
                    loaded = True

        # Carrega polígonos
        if os.path.exists(poly_path):
            with open(poly_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    yolo_id = int(parts[0])
                    if yolo_id not in yolo_to_cid:
                        continue
                    cid = yolo_to_cid[yolo_id]
                    coords = list(map(float, parts[1:]))
                    if len(coords) % 2 != 0:
                        continue
                    verts = []
                    for i in range(0, len(coords), 2):
                        x = coords[i] * w
                        y = coords[i+1] * h
                        verts.append((x, y))
                    if len(verts) >= 3:
                        self.class_manager.add_polygon_mask(cid, verts)
                        loaded = True

        if loaded:
            self.log(f"   📂 Anotações carregadas de arquivos YOLO para {nome_base}")
        return loaded