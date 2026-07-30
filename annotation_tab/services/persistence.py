# annotation_tab/services/persistence.py
"""
Serviço de persistência para anotações e configurações.
Inclui exportação YOLO.
"""

import os
import json
import shutil
import logging
from typing import Optional, Callable, Dict, List, Tuple
from ..models.class_manager import ClassManager

logger = logging.getLogger("AnnotationTab.Persistence")

# Constantes
CONFIG_FILE = 'classes_config.json'
YOLO_MAPPING_FILE = 'yolo_classes.txt'


class AnnotationPersistence:
    """Gerencia persistência de anotações e configurações."""

    def __init__(self, class_manager: ClassManager, log_callback: Optional[Callable] = None):
        self.class_manager = class_manager
        self.log = log_callback or (lambda msg: None)

    # --- Configuração de classes ---

    def save_classes(self, destino: Optional[str] = None) -> bool:
        """Salva configuração de classes em JSON e mapeamento."""
        config_path = self._get_config_path(destino)
        try:
            data = self.class_manager.to_dict()
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


            self.log(f"✅ Configuração de classes salva em: {config_path}")
            self.log(f"   {len(data['classes'])} classes salvas")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar configuração: {e}")
            self.log(f"❌ Erro ao salvar configuração: {e}")
            return False

    def load_classes(self, destino: Optional[str] = None) -> bool:
        """Carrega configuração de classes de JSON."""
        config_path = self._get_config_path(destino)
        if not os.path.exists(config_path):
            self.log(f"ℹ️ Arquivo de configuração não encontrado: {config_path}")
            return False

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.class_manager.from_dict(data)
            self.log(f"✅ Configuração de classes carregada de: {config_path}")
            self.log(f"   {len(data['classes'])} classes carregadas")
            for cid_str, info in data['classes'].items():
                self.log(f"      ID {cid_str}: {info['name']} ({info['color']})")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            self.log(f"❌ Erro ao carregar configuração: {e}")
            return False

    # --- Anotações JSON ---

    def save_annotations(self, image_path: str, destino: str) -> bool:
        """Salva anotações de uma imagem em JSON."""
        removed = self.class_manager.remove_empty_classes()
        if removed > 0:
            self.log(f"   🗑️ {removed} classe(s) vazia(s) removidas")

        json_path = self._get_annotation_path(image_path, destino)
        data = self._prepare_annotation_data(image_path)

        try:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.log(f"   📝 JSON salvo com {len(data['classes'])} classes")
            for cid_str, cls_data in data['classes'].items():
                self.log(f"      Classe {cid_str} ({cls_data['name']}): "
                         f"{len(cls_data['bboxes'])} bboxes, {len(cls_data['polygons'])} polygons")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar JSON {json_path}: {e}")
            self.log(f"❌ Erro ao salvar JSON: {e}")
            return False

    def load_annotations(self, image_path: str, destino: str) -> bool:
        """Carrega anotações de JSON."""
        json_path = self._get_annotation_path(image_path, destino)
        if not os.path.exists(json_path):
            return False

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if self.class_manager._ny is None or self.class_manager._nx is None:
                logger.error("Dimensões da imagem não definidas")
                return False

            if data['shape'][0] != self.class_manager._ny or data['shape'][1] != self.class_manager._nx:
                logger.warning(f"Shape da imagem mudou, ignorando JSON antigo para {os.path.basename(image_path)}")
                return False

            self.class_manager.clear_annotations()
            self._restore_annotations(data)

            self.log(f"   📂 JSON carregado com {len(data['classes'])} classes")
            for cid_str, cls_data in data['classes'].items():
                self.log(f"      Classe {cid_str} ({cls_data['name']}): "
                         f"{len(cls_data.get('bboxes', []))} bboxes, {len(cls_data.get('polygons', []))} polygons")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar JSON {json_path}: {e}")
            self.log(f"❌ Erro ao carregar JSON: {e}")
            return False

    # --- Exportação YOLO ---

    def export_yolo(self, image_path: str, destino: str) -> Tuple[int, int]:
        """Exporta anotações para YOLO (bbox e polygon)."""
        exporter = YoloExporter(self.class_manager, self.log)
        return exporter.export(image_path, destino)

    # --- Métodos privados ---

    def _get_config_path(self, destino: Optional[str]) -> str:
        if destino:
            return os.path.join(destino, CONFIG_FILE)
        return CONFIG_FILE


    def _get_annotation_path(self, image_path: str, destino: str) -> str:
        nome_base = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(destino, 'annotations', nome_base + '.json')

    def _prepare_annotation_data(self, image_path: str) -> dict:
        data = {
            'image_path': image_path,
            'shape': [self.class_manager._ny, self.class_manager._nx],
            'classes': {}
        }
        for cid, info in self.class_manager.classes.items():
            if info['is_background']:
                continue
            if len(info['bboxes']) > 0 or len(info['polygons']) > 0:
                data['classes'][str(cid)] = {
                    'name': info['name'],
                    'color': info['color'],
                    'bboxes': info['bboxes'],
                    'polygons': info['polygons']
                }
        return data

    def _restore_annotations(self, data: dict) -> None:
        for cid_str, info in data['classes'].items():
            cid = int(cid_str)
            if cid not in self.class_manager.classes:
                self.class_manager.classes[cid] = {
                    'name': info['name'],
                    'color': info['color'],
                    'mask': None,
                    'polygons': [],
                    'bboxes': [],
                    'desc': '',
                    'is_background': False,
                    'artists': []
                }
                if cid >= self.class_manager.next_id:
                    self.class_manager.next_id = cid + 1
                if cid in self.class_manager.available_ids:
                    self.class_manager.available_ids.remove(cid)

            for (x1, y1, x2, y2) in info.get('bboxes', []):
                self.class_manager.add_bbox(cid, x1, y1, x2, y2)
            for verts in info.get('polygons', []):
                if len(verts) >= 3:
                    self.class_manager.add_polygon_mask(cid, verts)


# ============================================================================
# YOLO Exporter (internal class)
# ============================================================================
class YoloExporter:
    """Exporta anotações para formato YOLO (detecção e segmentação)."""

    def __init__(self, class_manager: ClassManager, log_callback: Optional[Callable] = None):
        self.class_manager = class_manager
        self.log = log_callback or (lambda msg: None)

    def export(self, image_path: str, destino: str) -> Tuple[int, int]:
        pasta_imgs = os.path.join(destino, 'images')
        pasta_labels_bbox = os.path.join(destino, 'labels', 'bbox')
        pasta_labels_poly = os.path.join(destino, 'labels', 'polygon')
        os.makedirs(pasta_imgs, exist_ok=True)
        os.makedirs(pasta_labels_bbox, exist_ok=True)
        os.makedirs(pasta_labels_poly, exist_ok=True)

        nome_base = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(image_path)[1]
        destino_img = os.path.join(pasta_imgs, nome_base + ext)
        shutil.copy2(image_path, destino_img)

        h, w = self.class_manager._ny, self.class_manager._nx
        if h is None or w is None:
            self.log("❌ Dimensões da imagem não definidas.")
            return 0, 0

        sorted_ids = self.class_manager.get_sorted_non_background_ids()
        id_to_yolo = {cid: idx for idx, cid in enumerate(sorted_ids)}

        num_bbox = self._export_bbox(nome_base, destino, h, w, id_to_yolo)
        num_poly = self._export_polygon(nome_base, destino, h, w, id_to_yolo)

        self._save_mapping(destino, sorted_ids, id_to_yolo)
        return num_bbox, num_poly

    def _export_bbox(self, nome_base: str, destino: str, h: int, w: int,
                     id_to_yolo: Dict[int, int]) -> int:
        destino_txt = os.path.join(destino, 'labels', 'bbox', nome_base + '.txt')
        linhas = []
        for cid, info in self.class_manager.classes.items():
            if info['is_background'] or cid not in id_to_yolo:
                continue
            yolo_idx = id_to_yolo[cid]
            for (x1, y1, x2, y2) in info['bboxes']:
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                cx = max(0, min(cx, 1))
                cy = max(0, min(cy, 1))
                bw = max(0, min(bw, 1))
                bh = max(0, min(bh, 1))
                linhas.append(f"{yolo_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(destino_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(linhas))
        return len(linhas)

    def _export_polygon(self, nome_base: str, destino: str, h: int, w: int,
                        id_to_yolo: Dict[int, int]) -> int:
        destino_txt = os.path.join(destino, 'labels', 'polygon', nome_base + '.txt')
        linhas = []
        for cid, info in self.class_manager.classes.items():
            if info['is_background'] or cid not in id_to_yolo:
                continue
            yolo_idx = id_to_yolo[cid]
            for verts in info['polygons']:
                if len(verts) < 3:
                    continue
                norm_pts = []
                for (x, y) in verts:
                    x = max(0, min(x, w))
                    y = max(0, min(y, h))
                    norm_pts.append(f"{x/w:.6f}")
                    norm_pts.append(f"{y/h:.6f}")
                linhas.append(f"{yolo_idx} " + " ".join(norm_pts))
        with open(destino_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(linhas))
        return len(linhas)

    def _save_mapping(self, destino: str, sorted_ids: List[int],
                      id_to_yolo: Dict[int, int]) -> None:
        mapping_path = os.path.join(destino, YOLO_MAPPING_FILE)
        with open(mapping_path, 'w', encoding='utf-8') as f:
            for cid in sorted_ids:
                yolo_id = id_to_yolo[cid]
                nome = self.class_manager.get_class_name(cid)
                f.write(f"{yolo_id}: {nome} (ID original {cid})\n")
        self.log(f"📝 Mapeamento YOLO salvo em {mapping_path}")

    @staticmethod
    def load_mapping(mapping_path: str) -> Dict[int, str]:
        mapping = {}
        if not os.path.exists(mapping_path):
            return mapping
        with open(mapping_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                parts = line.split(':', 1)
                yolo_id = int(parts[0].strip())
                nome_part = parts[1].split('(')[0].strip()
                mapping[yolo_id] = nome_part
        return mapping