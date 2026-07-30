"""
Aba de Anotação para o SpectralImage
Integra a ferramenta de anotação YOLO como uma aba no aplicativo principal
Com melhorias de performance, threading, exportação YOLO e mapeamento de IDs.
"""

import os
import sys
import shutil
import json
import time
import random
import threading
import logging
import traceback
from pathlib import Path as PathLib
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path as MPath
from matplotlib.widgets import RectangleSelector
from skimage.draw import polygon
from PIL import Image

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================
EXTENSOES_VALIDAS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif', '.webp', '.heic')
BACKGROUND_KEYWORDS = ('bg', 'background', 'fundo', 'esteira')
CONFIG_FILE = 'classes_config.json'
MAPPING_FILE = 'classes_mapping.txt'
YOLO_MAPPING_FILE = 'yolo_classes.txt'

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AnnotationTab")

# ============================================================================
# CLASS MANAGER (OTIMIZADO)
# ============================================================================
class ClassManager:
    """Gerencia classes, anotações e máscaras com otimização via skimage.draw.polygon."""

    def __init__(self):
        self.classes: Dict[int, Dict] = {}
        self.next_id: int = 1
        self.available_ids: Set[int] = set()
        self._ny: Optional[int] = None
        self._nx: Optional[int] = None
        self._background_id: Optional[int] = None
        self._mask_cache: Dict[int, np.ndarray] = {}  # cache opcional

    def set_image_shape(self, shape: Tuple[int, int, int]) -> None:
        self._ny, self._nx = shape[:2]
        self._mask_cache.clear()

    def add_class(self, name: str, color: Optional[str] = None, desc: str = "") -> int:
        if color is None:
            # Gera cor a partir de colormap
            from matplotlib import colormaps
            cmap = colormaps['tab20']
            idx = (self.next_id - 1) % 20
            color = cmap(idx)
            # Converte para string hex
            import matplotlib.colors as mcolors
            color = mcolors.to_hex(color)

        if self.available_ids:
            class_id = min(self.available_ids)
            self.available_ids.remove(class_id)
        else:
            class_id = self.next_id
            self.next_id += 1

        is_bg = any(kw in name.lower() for kw in BACKGROUND_KEYWORDS)
        if is_bg and self._background_id is not None:
            logger.warning(f"Aviso: '{name}' não será fundo, já existe classe {self._background_id}.")
            is_bg = False
        if is_bg:
            self._background_id = class_id

        self.classes[class_id] = {
            'name': name,
            'color': color,
            'mask': None,
            'polygons': [],
            'bboxes': [],
            'desc': desc,
            'is_background': is_bg,
            'artists': []   # para desfazer incremental
        }
        self._mask_cache.pop(class_id, None)
        return class_id

    def remove_class(self, class_id: int) -> None:
        if class_id in self.classes:
            if self._background_id == class_id:
                self._background_id = None
            del self.classes[class_id]
            self.available_ids.add(class_id)
            self._mask_cache.pop(class_id, None)

    def rename_class(self, class_id: int, new_name: str) -> None:
        if class_id not in self.classes:
            return
        self.classes[class_id]['name'] = new_name
        is_bg = any(kw in new_name.lower() for kw in BACKGROUND_KEYWORDS)
        if is_bg and self._background_id is not None and self._background_id != class_id:
            logger.warning(f"Aviso: '{new_name}' não será fundo, já existe classe {self._background_id}.")
            is_bg = False
        self.classes[class_id]['is_background'] = is_bg
        if is_bg:
            self._background_id = class_id
        elif self._background_id == class_id:
            self._background_id = None
        self._mask_cache.pop(class_id, None)

    def get_class(self, class_id: int) -> Optional[Dict]:
        return self.classes.get(class_id)

    def get_class_name(self, class_id: int) -> Optional[str]:
        info = self.classes.get(class_id)
        return info['name'] if info else None

    def get_all_classes(self) -> Dict[int, Dict]:
        return self.classes

    def get_class_list(self) -> List[Tuple[int, str, str, int]]:
        result = []
        for cid, info in self.classes.items():
            if info['is_background']:
                continue
            total = len(info['bboxes']) + len(info['polygons'])
            result.append((cid, info['name'], info['color'], total))
        return result

    def get_background_id(self) -> Optional[int]:
        return self._background_id

    def get_sorted_non_background_ids(self) -> List[int]:
        return sorted([cid for cid, info in self.classes.items() if not info['is_background']])

    def add_bbox(self, class_id: int, x1: float, y1: float, x2: float, y2: float) -> bool:
        if class_id not in self.classes or self._ny is None:
            return False
        x1 = max(0, min(x1, self._nx))
        x2 = max(0, min(x2, self._nx))
        y1 = max(0, min(y1, self._ny))
        y2 = max(0, min(y2, self._ny))
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return False
        self.classes[class_id]['bboxes'].append((x1, y1, x2, y2))
        self._recalc_mask(class_id)
        return True

    def add_polygon_mask(self, class_id: int, polygon_verts: List[Tuple[float, float]]) -> bool:
        if class_id not in self.classes or self._ny is None:
            return False
        if len(polygon_verts) < 3:
            return False
        # Fecha polígono se necessário
        if polygon_verts[0] != polygon_verts[-1]:
            polygon_verts = polygon_verts + [polygon_verts[0]]
        # Verifica se tem área
        if len(polygon_verts) < 4:
            return False
        self.classes[class_id]['polygons'].append(polygon_verts)
        self._recalc_mask(class_id)
        return True

    def _recalc_mask(self, class_id: int) -> None:
        """Recalcula a máscara usando skimage.draw.polygon (otimizado)."""
        if class_id not in self.classes or self._ny is None:
            return
        info = self.classes[class_id]
        mask = np.zeros((self._ny, self._nx), dtype=bool)

        for verts in info['polygons']:
            if len(verts) < 3:
                continue
            rr, cc = polygon([p[1] for p in verts], [p[0] for p in verts], mask.shape)
            mask[rr, cc] = True

        for (x1, y1, x2, y2) in info['bboxes']:
            mask[int(y1):int(y2), int(x1):int(x2)] = True

        info['mask'] = mask if mask.any() else None
        self._mask_cache[class_id] = info['mask']

    def undo_last(self, class_id: int) -> bool:
        """Desfaz a última anotação (bbox ou polígono) e retorna True se bem sucedido."""
        if class_id not in self.classes:
            return False
        info = self.classes[class_id]

        if info['polygons']:
            # Remove o último polígono
            removed = info['polygons'].pop()
            # Remove também o artista associado (se existir)
            if info.get('artists') and len(info['artists']) > 0:
                last_artist = info['artists'].pop()
                try:
                    last_artist.remove()
                except:
                    pass
        elif info['bboxes']:
            removed = info['bboxes'].pop()
            if info.get('artists') and len(info['artists']) > 0:
                last_artist = info['artists'].pop()
                try:
                    last_artist.remove()
                except:
                    pass
        else:
            return False

        self._recalc_mask(class_id)
        return True

    def clear_annotations(self) -> None:
        for cid in self.classes:
            self.classes[cid]['bboxes'] = []
            self.classes[cid]['polygons'] = []
            self.classes[cid]['mask'] = None
            self.classes[cid]['artists'] = []
        self._mask_cache.clear()

    def get_annotation_count(self) -> Dict[int, Dict]:
        counts = {}
        for cid, info in self.classes.items():
            if not info['is_background']:
                counts[cid] = {
                    'name': info['name'],
                    'bboxes': len(info['bboxes']),
                    'polygons': len(info['polygons']),
                    'total': len(info['bboxes']) + len(info['polygons'])
                }
        return counts

    def get_mask(self, class_id: int) -> Optional[np.ndarray]:
        if class_id in self._mask_cache:
            return self._mask_cache[class_id]
        info = self.classes.get(class_id)
        if info is None or self._ny is None:
            return None
        if not info['is_background']:
            return info['mask']
        # Para fundo, retorna complemento da união das outras classes
        union_others = None
        for cid, other in self.classes.items():
            if cid == class_id:
                continue
            other_mask = self.get_mask(cid)
            if other_mask is not None and other_mask.any():
                if union_others is None:
                    union_others = other_mask.copy()
                else:
                    union_others |= other_mask
        if union_others is None:
            return np.ones((self._ny, self._nx), dtype=bool)
        return ~union_others

    def get_pixel_count(self, class_id: int) -> int:
        mask = self.get_mask(class_id)
        return mask.sum() if mask is not None else 0

    def remove_empty_classes(self) -> int:
        to_remove = []
        for cid, info in self.classes.items():
            if not info['is_background'] and len(info['bboxes']) == 0 and len(info['polygons']) == 0:
                to_remove.append(cid)
        for cid in to_remove:
            self.remove_class(cid)
        return len(to_remove)

    def to_dict(self) -> Dict:
        data = {
            'classes': {},
            'next_id': self.next_id,
            'background_id': self._background_id
        }
        for cid, info in self.classes.items():
            data['classes'][str(cid)] = {
                'name': info['name'],
                'color': info['color'],
                'desc': info.get('desc', ''),
                'is_background': info['is_background']
            }
        return data

    def from_dict(self, data: Dict) -> None:
        self.classes.clear()
        self.available_ids.clear()
        self.next_id = data.get('next_id', 1)
        self._background_id = data.get('background_id', None)
        for cid_str, info in data['classes'].items():
            cid = int(cid_str)
            self.classes[cid] = {
                'name': info['name'],
                'color': info['color'],
                'mask': None,
                'polygons': [],
                'bboxes': [],
                'desc': info.get('desc', ''),
                'is_background': info.get('is_background', False),
                'artists': []
            }
            if info.get('is_background', False):
                self._background_id = cid
        self._mask_cache.clear()

    def save_mapping_file(self, filepath: str) -> bool:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for cid, info in sorted(self.classes.items()):
                    if not info['is_background']:
                        f.write(f"{cid}: {info['name']}\n")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar mapeamento: {e}")
            return False

# ============================================================================
# YOLO EXPORTER (6.4)
# ============================================================================
class YoloExporter:
    """Gerencia a exportação de anotações para o formato YOLO (detecção e segmentação)."""

    def __init__(self, class_manager: ClassManager, log_callback: Optional[Callable] = None):
        self.class_manager = class_manager
        self.log_callback = log_callback or (lambda msg: None)

    def export(self, image_path: str, destino: str) -> Tuple[int, int]:
        """Exporta anotações para YOLO (bbox e polygon) e copia a imagem. Retorna (num_bbox, num_poly)."""
        # Cria estrutura de pastas
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
            logger.error("Dimensões da imagem não definidas")
            self.log_callback("❌ Dimensões da imagem não definidas.")
            return 0, 0

        # Mapeamento ID original -> YOLO ID (0-based)
        sorted_ids = self.class_manager.get_sorted_non_background_ids()
        id_to_yolo = {cid: idx for idx, cid in enumerate(sorted_ids)}

        num_bbox = self._export_bbox(nome_base, destino, h, w, id_to_yolo)
        num_poly = self._export_polygon(nome_base, destino, h, w, id_to_yolo)

        # Salva o mapeamento YOLO ID -> nome da classe
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
                # Clamping
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
        """Salva um arquivo de mapeamento YOLO ID -> nome da classe (e ID original)."""
        mapping_path = os.path.join(destino, YOLO_MAPPING_FILE)
        with open(mapping_path, 'w', encoding='utf-8') as f:
            for cid in sorted_ids:
                yolo_id = id_to_yolo[cid]
                nome = self.class_manager.get_class_name(cid)
                f.write(f"{yolo_id}: {nome} (ID original {cid})\n")
        self.log_callback(f"📝 Mapeamento YOLO salvo em {mapping_path}")

    @staticmethod
    def load_mapping(mapping_path: str) -> Dict[int, str]:
        """Lê o mapeamento de YOLO ID -> nome da classe."""
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
                # Extrai nome antes do parêntese
                nome_part = parts[1].split('(')[0].strip()
                mapping[yolo_id] = nome_part
        return mapping

# ============================================================================
# FUNÇÕES DE PERSISTÊNCIA (JSON e YOLO)
# ============================================================================
def salvar_classes_config(class_manager: ClassManager, pasta_destino: Optional[str] = None,
                          log_callback: Optional[Callable] = None) -> bool:
    if pasta_destino is None:
        config_path = CONFIG_FILE
    else:
        config_path = os.path.join(pasta_destino, CONFIG_FILE)

    data = class_manager.to_dict()
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Também salva o arquivo de mapeamento
        mapping_path = os.path.join(os.path.dirname(config_path), MAPPING_FILE)
        class_manager.save_mapping_file(mapping_path)

        if log_callback:
            log_callback(f"✅ Configuração de classes salva em: {config_path}")
            log_callback(f"   Mapeamento salvo em: {mapping_path}")
            log_callback(f"   {len(data['classes'])} classes salvas")
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar configuração: {e}")
        if log_callback:
            log_callback(f"❌ Erro ao salvar configuração: {e}")
        traceback.print_exc()
        return False

def carregar_classes_config(class_manager: ClassManager, pasta_destino: Optional[str] = None,
                            log_callback: Optional[Callable] = None) -> bool:
    if pasta_destino is None:
        config_path = CONFIG_FILE
    else:
        config_path = os.path.join(pasta_destino, CONFIG_FILE)

    if not os.path.exists(config_path):
        if log_callback:
            log_callback(f"ℹ️ Arquivo de configuração não encontrado: {config_path}")
        return False

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        class_manager.from_dict(data)
        if log_callback:
            log_callback(f"✅ Configuração de classes carregada de: {config_path}")
            log_callback(f"   {len(data['classes'])} classes carregadas")
            for cid_str, info in data['classes'].items():
                log_callback(f"      ID {cid_str}: {info['name']} ({info['color']})")
        return True
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {e}")
        if log_callback:
            log_callback(f"❌ Erro ao carregar configuração: {e}")
        traceback.print_exc()
        return False

def salvar_anotacoes_json(image_path: str, class_manager: ClassManager,
                          pasta_destino: str, log_callback: Optional[Callable] = None) -> bool:
    removed = class_manager.remove_empty_classes()
    if removed > 0 and log_callback:
        log_callback(f"   🗑️ {removed} classe(s) vazia(s) removidas")

    pasta_annot = os.path.join(pasta_destino, 'annotations')
    os.makedirs(pasta_annot, exist_ok=True)

    nome_base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(pasta_annot, nome_base + '.json')

    data = {
        'image_path': image_path,
        'shape': [class_manager._ny, class_manager._nx],
        'classes': {}
    }

    for cid, info in class_manager.classes.items():
        if info['is_background']:
            continue
        if len(info['bboxes']) > 0 or len(info['polygons']) > 0:
            data['classes'][str(cid)] = {
                'name': info['name'],
                'color': info['color'],
                'bboxes': info['bboxes'],
                'polygons': info['polygons']
            }

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        if log_callback:
            log_callback(f"   📝 JSON salvo com {len(data['classes'])} classes")
            for cid_str, cls_data in data['classes'].items():
                log_callback(f"      Classe {cid_str} ({cls_data['name']}): "
                             f"{len(cls_data['bboxes'])} bboxes, {len(cls_data['polygons'])} polygons")
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar JSON {json_path}: {e}")
        if log_callback:
            log_callback(f"❌ Erro ao salvar JSON: {e}")
        traceback.print_exc()
        return False

def carregar_anotacoes_json(image_path: str, class_manager: ClassManager,
                            pasta_destino: str, log_callback: Optional[Callable] = None) -> bool:
    nome_base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(pasta_destino, 'annotations', nome_base + '.json')

    if not os.path.exists(json_path):
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if class_manager._ny is None or class_manager._nx is None:
            logger.error("Dimensões da imagem não definidas")
            return False

        if data['shape'][0] != class_manager._ny or data['shape'][1] != class_manager._nx:
            logger.warning(f"Shape da imagem mudou, ignorando JSON antigo para {nome_base}")
            return False

        class_manager.clear_annotations()

        for cid_str, info in data['classes'].items():
            cid = int(cid_str)
            if cid not in class_manager.classes:
                class_manager.classes[cid] = {
                    'name': info['name'],
                    'color': info['color'],
                    'mask': None,
                    'polygons': [],
                    'bboxes': [],
                    'desc': '',
                    'is_background': False,
                    'artists': []
                }
                if cid >= class_manager.next_id:
                    class_manager.next_id = cid + 1
                if cid in class_manager.available_ids:
                    class_manager.available_ids.remove(cid)

            for (x1, y1, x2, y2) in info.get('bboxes', []):
                class_manager.add_bbox(cid, x1, y1, x2, y2)
            for verts in info.get('polygons', []):
                if len(verts) >= 3:
                    class_manager.add_polygon_mask(cid, verts)

        if log_callback:
            log_callback(f"   📂 JSON carregado com {len(data['classes'])} classes")
            for cid_str, cls_data in data['classes'].items():
                log_callback(f"      Classe {cid_str} ({cls_data['name']}): "
                             f"{len(cls_data.get('bboxes', []))} bboxes, {len(cls_data.get('polygons', []))} polygons")
        return True
    except Exception as e:
        logger.error(f"Erro ao carregar JSON {json_path}: {e}")
        if log_callback:
            log_callback(f"❌ Erro ao carregar JSON: {e}")
        traceback.print_exc()
        return False

def carregar_anotacoes_de_yolo(image_path: str, class_manager: ClassManager,
                               pasta_destino: str, log_callback: Optional[Callable] = None) -> bool:
    """Tenta carregar anotações dos arquivos YOLO (bbox e polygon) se existirem."""
    nome_base = os.path.splitext(os.path.basename(image_path))[0]

    # Carrega mapeamento YOLO -> nome
    mapping_path = os.path.join(pasta_destino, YOLO_MAPPING_FILE)
    yolo_to_name = YoloExporter.load_mapping(mapping_path)
    if not yolo_to_name:
        return False

    # Garante que as classes existam no class_manager
    yolo_to_cid = {}
    for yolo_id, nome in yolo_to_name.items():
        found = None
        for cid, info in class_manager.classes.items():
            if info['name'] == nome and not info['is_background']:
                found = cid
                break
        if found is None:
            cid = class_manager.add_class(nome)
        else:
            cid = found
        yolo_to_cid[yolo_id] = cid

    # Lê arquivos bbox e polygon
    bbox_path = os.path.join(pasta_destino, 'labels', 'bbox', nome_base + '.txt')
    poly_path = os.path.join(pasta_destino, 'labels', 'polygon', nome_base + '.txt')

    loaded = False
    h, w = class_manager._ny, class_manager._nx
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
                class_manager.add_bbox(cid, x1, y1, x2, y2)
                loaded = True

    # Carrega polígonos
    if os.path.exists(poly_path):
        with open(poly_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 7:  # mínimo: id + 3 pontos (6 coordenadas)
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
                    class_manager.add_polygon_mask(cid, verts)
                    loaded = True

    if loaded and log_callback:
        log_callback(f"   📂 Anotações carregadas de arquivos YOLO para {nome_base}")
    return loaded

# ============================================================================
# FOLDER LOADER (THREADING)
# ============================================================================
class FolderLoader:
    """Carrega lista de imagens em thread separada e notifica quando pronto."""

    def __init__(self, origem: str, destino: str, callback: Callable,
                 log_callback: Optional[Callable] = None):
        self.origem = origem
        self.destino = destino
        self.callback = callback   # função a chamar com a lista de arquivos
        self.log_callback = log_callback or (lambda msg: None)
        self.thread: Optional[threading.Thread] = None
        self.running: bool = False

    def start(self) -> None:
        if not os.path.isdir(self.origem):
            self.log_callback("❌ Pasta de origem inválida.")
            return
        if not os.path.isdir(self.destino):
            self.log_callback("❌ Pasta de destino inválida.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._load, daemon=True)
        self.thread.start()

    def _load(self) -> None:
        try:
            # Lista arquivos de imagem recursivamente
            arquivos = []
            for root, dirs, files in os.walk(self.origem):
                for f in files:
                    if f.lower().endswith(EXTENSOES_VALIDAS):
                        arquivos.append(os.path.join(root, f))
            # Filtra os já anotados (verificação rápida)
            ja_feitos = self._get_annotated_images(self.destino)
            pendentes = [f for f in arquivos if os.path.splitext(os.path.basename(f))[0] not in ja_feitos]
            pendentes.sort()
            self.callback(pendentes)
        except Exception as e:
            logger.error(f"Erro ao carregar arquivos: {e}")
            self.log_callback(f"❌ Erro ao carregar arquivos: {e}")
        finally:
            self.running = False

    def _get_annotated_images(self, destino: str) -> Set[str]:
        ja_feitas = set()
        for sub in ['annotations', 'labels/bbox', 'labels/polygon']:
            path = os.path.join(destino, sub)
            if os.path.isdir(path):
                for f in os.listdir(path):
                    if f.endswith(('.json', '.txt')):
                        ja_feitas.add(os.path.splitext(f)[0])
        return ja_feitas

# ============================================================================
# ANOTATOR (MATPLOTLIB)
# ============================================================================
class Annotator:
    """Classe responsável pela exibição e interação de anotação (bbox/polygon)."""

    def __init__(self, parent_frame: tk.Frame, image_path: str, img_array: np.ndarray,
                 class_manager: ClassManager, log_callback: Callable,
                 action_var: tk.StringVar, on_update_callback: Callable,
                 on_mode_change_callback: Callable, mode: str = 'bbox'):
        self.parent_frame = parent_frame
        self.image_path = image_path
        self.img_array = img_array
        self.class_manager = class_manager
        self.log_callback = log_callback
        self.action_var = action_var
        self.on_update_callback = on_update_callback
        self.on_mode_change_callback = on_mode_change_callback
        self.mode = mode
        self.h, self.w = img_array.shape[0], img_array.shape[1]
        self.class_manager.set_image_shape((self.h, self.w))

        self.current_class_id: Optional[int] = None
        self.acao: Optional[str] = None

        self.polygon_points: List[Tuple[float, float]] = []
        self.polygon_artists: List = []
        self.is_drawing_polygon: bool = False

        self.fig = plt.Figure(figsize=(10, 8), facecolor='lightgray')
        self.ax_image = self.fig.add_axes([0.05, 0.05, 0.90, 0.90])
        self.ax_image.imshow(self.img_array)
        self.ax_image.axis('on')
        self.atualizar_titulo()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.parent_frame)
        self.toolbar.update()

        self.rect_selector: Optional[RectangleSelector] = None
        self.inicializar_selector()

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.fig.canvas.draw_idle()

    def inicializar_selector(self) -> None:
        if self.rect_selector:
            self.rect_selector.disconnect_events()
            self.rect_selector = None

        self.polygon_points = []
        self.is_drawing_polygon = False
        self._clear_polygon_preview()

        cor = 'gray' if self.current_class_id is None else self.class_manager.classes[self.current_class_id]['color']

        if self.mode == 'bbox':
            self.rect_selector = RectangleSelector(
                self.ax_image,
                self.onselect_bbox,
                useblit=True,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords='pixels',
                interactive=False,
                props=dict(facecolor=cor, edgecolor=cor, alpha=0.25, fill=True)
            )
        else:
            self.rect_selector = None
            self.is_drawing_polygon = False
            self.polygon_points = []
            self._clear_polygon_preview()
            self.log("🖊️ Clique para adicionar pontos. Clique no primeiro ponto para fechar o polígono.")
            self.log("   Use Z para desfazer o último ponto.")

    def _clear_polygon_preview(self) -> None:
        for artist in self.polygon_artists:
            try:
                artist.remove()
            except:
                pass
        self.polygon_artists = []
        self.fig.canvas.draw_idle()

    def _update_polygon_preview(self) -> None:
        self._clear_polygon_preview()
        if len(self.polygon_points) < 1:
            self.atualizar_titulo()
            return

        points = np.array(self.polygon_points)
        scatter = self.ax_image.scatter(points[:, 0], points[:, 1],
                                        color='yellow', s=50, zorder=5)
        self.polygon_artists.append(scatter)

        if len(self.polygon_points) > 1:
            line = plt.Line2D(points[:, 0], points[:, 1],
                              color='yellow', linewidth=2, zorder=4)
            self.ax_image.add_line(line)
            self.polygon_artists.append(line)

        self.atualizar_titulo()
        self.fig.canvas.draw_idle()

    def _undo_polygon_point(self) -> bool:
        if not self.is_drawing_polygon or len(self.polygon_points) == 0:
            self.log("ℹ️ Nenhum ponto para desfazer.")
            return False

        self.polygon_points.pop()
        self.log(f"↩️ Ponto removido - {len(self.polygon_points)} pontos restantes")

        if len(self.polygon_points) == 0:
            self.is_drawing_polygon = False
            self._clear_polygon_preview()
            self.log("   Desenho do polígono cancelado.")
        else:
            self._update_polygon_preview()
        return True

    def on_mouse_click(self, event) -> None:
        if self.mode != 'polygon' or event.inaxes != self.ax_image:
            return

        if self.current_class_id is None:
            self.log("⚠️ Selecione uma classe primeiro!")
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        if x < 0 or x > self.w or y < 0 or y > self.h:
            return

        if not self.is_drawing_polygon:
            self.is_drawing_polygon = True
            self.polygon_points = [(x, y)]
            self._update_polygon_preview()
            self.log(f"   🖊️ Iniciando polígono em ({x:.1f}, {y:.1f})")
            return

        if len(self.polygon_points) >= 3:
            first_point = self.polygon_points[0]
            dist = np.sqrt((x - first_point[0])**2 + (y - first_point[1])**2)
            if dist < 15:
                self._finalize_polygon()
                return

        self.polygon_points.append((x, y))
        self._update_polygon_preview()
        self.log(f"   🖊️ Ponto adicionado: ({x:.1f}, {y:.1f}) - {len(self.polygon_points)} pontos")

    def _finalize_polygon(self) -> None:
        if len(self.polygon_points) < 3:
            self.log("⚠️ Polígono precisa de pelo menos 3 pontos.")
            self.is_drawing_polygon = False
            self.polygon_points = []
            self._clear_polygon_preview()
            return

        if self.polygon_points[0] != self.polygon_points[-1]:
            self.polygon_points.append(self.polygon_points[0])

        if self.class_manager.add_polygon_mask(self.current_class_id, self.polygon_points):
            cor = self.class_manager.classes[self.current_class_id]['color']
            poly = Polygon(self.polygon_points, linewidth=2, edgecolor=cor, facecolor='none')
            self.ax_image.add_patch(poly)
            # Armazena o artista para desfazer
            self.class_manager.classes[self.current_class_id]['artists'].append(poly)

            counts = self.class_manager.get_annotation_count()
            total = sum(c['total'] for c in counts.values())
            self.log(f"✅ Polígono finalizado e salvo (classe {self.class_manager.classes[self.current_class_id]['name']}) - Total: {total}")
            self.atualizar_titulo()
            self._notify_update()
        else:
            self.log("❌ Falha ao adicionar polígono (máscara vazia?).")

        self.is_drawing_polygon = False
        self.polygon_points = []
        self._clear_polygon_preview()
        self.fig.canvas.draw_idle()

    def set_mode(self, mode: str) -> None:
        if mode not in ('bbox', 'polygon'):
            return
        self.mode = mode
        self.is_drawing_polygon = False
        self.polygon_points = []
        self._clear_polygon_preview()
        self.inicializar_selector()
        self.atualizar_titulo()
        self.log(f"Modo alterado para: {self.mode.upper()}")
        if self.on_mode_change_callback:
            self.on_mode_change_callback(mode)

    def toggle_mode(self) -> None:
        new_mode = 'polygon' if self.mode == 'bbox' else 'bbox'
        self.set_mode(new_mode)

    def set_class(self, class_id: int) -> None:
        if class_id in self.class_manager.classes:
            self.current_class_id = class_id
            self.is_drawing_polygon = False
            self.polygon_points = []
            self._clear_polygon_preview()
            self.inicializar_selector()
            self.atualizar_titulo()
            self.fig.canvas.draw_idle()
            self.log(f"Classe ativada: {self.class_manager.classes[class_id]['name']} (ID {class_id})")

    def atualizar_titulo(self) -> None:
        nome_arquivo = os.path.basename(self.image_path)
        nome_classe = "Nenhuma" if self.current_class_id is None else self.class_manager.get_class_name(self.current_class_id)
        cor = "gray" if self.current_class_id is None else self.class_manager.classes[self.current_class_id]['color']

        counts = self.class_manager.get_annotation_count()
        info_text = ""
        if counts:
            total = sum(c['total'] for c in counts.values())
            info_text = f" | Total: {total} objetos"

        if self.mode == 'polygon' and self.is_drawing_polygon and len(self.polygon_points) > 0:
            info_text += f" | Desenhando: {len(self.polygon_points)} pontos (Z para desfazer)"

        self.ax_image.set_title(
            f"{nome_arquivo}\nClasse: {nome_classe}  |  Modo: {self.mode.upper()}{info_text}",
            fontsize=11, color=cor
        )

    def log(self, msg: str) -> None:
        if self.log_callback:
            self.log_callback(msg)

    def onselect_bbox(self, eclick, erelease) -> None:
        if self.current_class_id is None:
            self.log("⚠️ Selecione uma classe primeiro!")
            return
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 is None or x2 is None:
            return
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        x1 = max(0, min(x1, self.w))
        x2 = max(0, min(x2, self.w))
        y1 = max(0, min(y1, self.h))
        y2 = max(0, min(y2, self.h))
        if (x2 - x1) < 3 or (y2 - y1) < 3:
            self.log("⚠️ Caixa muito pequena, ignorada.")
            return

        if self.class_manager.add_bbox(self.current_class_id, x1, y1, x2, y2):
            cor = self.class_manager.classes[self.current_class_id]['color']
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=cor, facecolor='none')
            self.ax_image.add_patch(rect)
            # Armazena o artista para desfazer
            self.class_manager.classes[self.current_class_id]['artists'].append(rect)
            self.fig.canvas.draw_idle()

            counts = self.class_manager.get_annotation_count()
            total = sum(c['total'] for c in counts.values())
            self.log(f"✅ BBox adicionada (classe {self.class_manager.classes[self.current_class_id]['name']}) - Total: {total}")
            self.atualizar_titulo()
            self._notify_update()
        else:
            self.log("❌ Falha ao adicionar BBox.")

    def _notify_update(self) -> None:
        if self.on_update_callback:
            self.on_update_callback()

    def undo(self, event=None) -> None:
        if self.is_drawing_polygon and len(self.polygon_points) > 0:
            self._undo_polygon_point()
            return

        if self.current_class_id is None:
            self.log("⚠️ Nenhuma classe ativa para desfazer.")
            return

        info = self.class_manager.classes.get(self.current_class_id)
        if not info or (not info['polygons'] and not info['bboxes']):
            self.log("ℹ️ Nada para desfazer.")
            return

        if self.class_manager.undo_last(self.current_class_id):
            # O artista já foi removido pelo undo_last, apenas atualizamos a figura
            self.fig.canvas.draw_idle()
            self._notify_update()
            counts = self.class_manager.get_annotation_count()
            total = sum(c['total'] for c in counts.values())
            self.log(f"↩️ Último objeto desfeito (classe {self.class_manager.classes[self.current_class_id]['name']}) - Total: {total}")
            self.atualizar_titulo()
        else:
            self.log("❌ Erro ao desfazer.")

    def on_key(self, event) -> None:
        classes = [cid for cid, info in sorted(self.class_manager.classes.items()) if not info['is_background']]

        if event.key in '1234567890':
            idx = int(event.key) - 1 if event.key != '0' else 9
            if idx < len(classes):
                self.set_class(classes[idx])
        elif event.key == 'z':
            self.undo()
        elif event.key == 'm':
            self.toggle_mode()
        elif event.key in ('n', 'enter'):
            if self.is_drawing_polygon and len(self.polygon_points) >= 3:
                self._finalize_polygon()
            else:
                self.set_acao('salvar')
        elif event.key == 'p':
            self.set_acao('pular')
        elif event.key == 'q':
            self.set_acao('sair')
        elif event.key == 'escape':
            if self.is_drawing_polygon:
                self.is_drawing_polygon = False
                self.polygon_points = []
                self._clear_polygon_preview()
                self.log("↩️ Desenho do polígono cancelado.")
                self.atualizar_titulo()

    def set_acao(self, acao: str) -> None:
        self.acao = acao
        self.log(f"⏳ Finalizando com ação: {acao}")
        if self.action_var:
            self.action_var.set(acao)
        try:
            self.canvas.get_tk_widget().destroy()
        except Exception as e:
            logger.error(f"Erro ao destruir canvas: {e}")
        try:
            plt.close(self.fig)
        except Exception as e:
            logger.error(f"Erro ao fechar figura: {e}")

# ============================================================================
# CLASSE PRINCIPAL DA ABA DE ANOTAÇÃO
# ============================================================================
class AnnotationTab:
    """Aba de anotação para integrar no notebook do aplicativo principal."""

    def __init__(self, parent: tk.Frame):
        self.parent = parent
        self._after_id: Optional[str] = None

        self.pasta_origem = tk.StringVar()
        self.pasta_destino = tk.StringVar()
        self.modo_inicial = tk.StringVar(value="bbox")
        self.class_manager = ClassManager()
        self.annotator: Optional[Annotator] = None
        self.action_var = tk.StringVar()
        self._arquivos_pendentes: List[str] = []
        self._indice_atual: int = 0
        self._total_imagens: int = 0
        self._classe_para_ativar: Optional[int] = None

        self._criar_widgets()
        self._carregar_configuracao_auto()
        self._after_id = self.log_text.after(100, self._poll_log_queue)
        self.parent.bind("<Destroy>", self._on_destroy)

    def _on_destroy(self, event) -> None:
        if self._after_id:
            try:
                self.log_text.after_cancel(self._after_id)
            except:
                pass
            self._after_id = None

    def _criar_widgets(self) -> None:
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ===== PAINEL ESQUERDO (imagem) =====
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

        # ===== PAINEL DIREITO (controles) =====
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
        # Pastas
        folder_frame = ttk.LabelFrame(controls_inner, text="Pastas")
        folder_frame.pack(fill=tk.X, pady=5)

        ttk.Label(folder_frame, text="Origem:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(folder_frame, textvariable=self.pasta_origem, width=20).grid(row=0, column=1, padx=2)
        ttk.Button(folder_frame, text="Browse", command=self._selecionar_pasta_origem, width=8).grid(row=0, column=2)
        self.btn_carregar = ttk.Button(folder_frame, text="Carregar",
                                       command=self._iniciar_anotacao, width=8)
        self.btn_carregar.grid(row=0, column=3, padx=(5, 0))

        ttk.Label(folder_frame, text="Destino:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(folder_frame, textvariable=self.pasta_destino, width=20).grid(row=1, column=1, padx=2)
        ttk.Button(folder_frame, text="Browse", command=self._selecionar_pasta_destino, width=8).grid(row=1, column=2)

        # Configuração de Classes
        config_frame = ttk.LabelFrame(controls_inner, text="Configuração de Classes")
        config_frame.pack(fill=tk.X, pady=5)

        btn_config_frame = ttk.Frame(config_frame)
        btn_config_frame.pack(fill=tk.X, pady=2)
        ttk.Button(btn_config_frame, text="💾 Salvar Classes",
                   command=self._salvar_configuracao_classes, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_config_frame, text="📂 Carregar Classes",
                   command=self._carregar_configuracao_classes, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_config_frame, text="🔄 Resetar Classes",
                   command=self._resetar_classes, width=15).pack(side=tk.LEFT, padx=2)

        # Modo inicial
        mode_frame = ttk.LabelFrame(controls_inner, text="Modo de Anotação")
        mode_frame.pack(fill=tk.X, pady=5)
        self.rb_bbox = ttk.Radiobutton(mode_frame, text="Bounding Box", variable=self.modo_inicial,
                                       value="bbox", command=self._radio_mode_changed)
        self.rb_bbox.pack(anchor=tk.W)
        self.rb_polygon = ttk.Radiobutton(mode_frame, text="Polígono", variable=self.modo_inicial,
                                          value="polygon", command=self._radio_mode_changed)
        self.rb_polygon.pack(anchor=tk.W)

        # Classes
        class_frame = ttk.LabelFrame(controls_inner, text="Gerenciar Classes")
        class_frame.pack(fill=tk.X, pady=5)
        list_frame = ttk.Frame(class_frame)
        list_frame.pack(fill=tk.X, pady=2)
        self.lista_classes = tk.Listbox(list_frame, height=6, selectmode=tk.SINGLE)
        self.lista_classes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_list = ttk.Scrollbar(list_frame, orient="vertical", command=self.lista_classes.yview)
        scroll_list.pack(side=tk.RIGHT, fill=tk.Y)
        self.lista_classes.config(yscrollcommand=scroll_list.set)

        btn_ger = ttk.Frame(class_frame)
        btn_ger.pack(fill=tk.X, pady=2)
        ttk.Button(btn_ger, text="Adicionar", command=self._adicionar_classe, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_ger, text="Remover", command=self._remover_classe, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_ger, text="Renomear", command=self._renomear_classe, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_ger, text="Ativar", command=self._ativar_classe, width=10).pack(side=tk.LEFT, padx=2)

        # Ações durante anotação
        action_frame = ttk.LabelFrame(controls_inner, text="Ações durante anotação")
        action_frame.pack(fill=tk.X, pady=5)

        btn_action_frame = ttk.Frame(action_frame)
        btn_action_frame.pack(fill=tk.X, pady=2)
        self.btn_salvar = ttk.Button(btn_action_frame, text="Salvar e Próximo (N)",
                                     command=self._acao_salvar, width=18)
        self.btn_salvar.pack(side=tk.LEFT, padx=2)
        self.btn_pular = ttk.Button(btn_action_frame, text="Pular (P)",
                                    command=self._acao_pular, width=12)
        self.btn_pular.pack(side=tk.LEFT, padx=2)
        self.btn_sair = ttk.Button(btn_action_frame, text="Sair (Q)",
                                   command=self._acao_sair, width=12)
        self.btn_sair.pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Desfazer (Z)", command=self._acao_desfazer, width=20).pack(pady=2)

        # Console de Logs
        log_frame = ttk.LabelFrame(controls_inner, text="Console de Logs")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        log_buttons = ttk.Frame(log_frame)
        log_buttons.pack(fill=tk.X)
        ttk.Button(log_buttons, text="Limpar Log", command=self._limpar_log, width=12).pack(side=tk.RIGHT, padx=2)

        self.log_text = tk.Text(log_frame, height=8, state='normal', wrap=tk.WORD,
                                bg='black', fg='lightgreen', font=('Consolas', 8))
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_log = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll_log.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scroll_log.set)

        # Atalhos
        shortcuts_frame = ttk.LabelFrame(controls_inner, text="Atalhos")
        shortcuts_frame.pack(fill=tk.X, pady=5)
        txt = ("1-9,0: selecionar classe\n"
               "Z: desfazer último ponto (polígono) ou última anotação\n"
               "M: alternar modo (BBox/Polígono)\n"
               "N/Enter: salvar e próxima\n"
               "P: pular imagem\n"
               "Q: sair\n"
               "ESC: cancelar desenho do polígono")
        ttk.Label(shortcuts_frame, text=txt, justify=tk.LEFT).pack(anchor=tk.W)

        # Status bar com progresso
        status_frame = ttk.Frame(self.parent)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Pronto")
        self.progress_var = tk.StringVar(value="")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress_label = ttk.Label(status_frame, textvariable=self.progress_var, relief=tk.SUNKEN, width=20)
        self.progress_label.pack(side=tk.RIGHT)

        self._atualizar_lista_classes()
        self._log("🟢 Aba de Anotação iniciada. Configure as pastas e classes.")

    # ====== LOG ======
    def _log(self, msg: str) -> None:
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.status_var.set(msg[:60])

    def _limpar_log(self) -> None:
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

    def _poll_log_queue(self) -> None:
        self._after_id = self.log_text.after(100, self._poll_log_queue)

    # ====== SELEÇÃO DE PASTAS ======
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
            config_path = os.path.join(pasta, CONFIG_FILE)
            if os.path.exists(config_path):
                if carregar_classes_config(self.class_manager, pasta, self._log):
                    self._atualizar_lista_classes()

    # ====== GERENCIAMENTO DE CLASSES ======
    def _atualizar_lista_classes(self) -> None:
        self.lista_classes.delete(0, tk.END)
        for cid, info in sorted(self.class_manager.classes.items()):
            if info['is_background']:
                continue
            total = len(info['bboxes']) + len(info['polygons'])
            self.lista_classes.insert(tk.END, f"{cid}: {info['name']} ({info['color']}) [{total}]")

    def _adicionar_classe(self) -> None:
        nome = simpledialog.askstring("Nova Classe", "Nome da classe:")
        if nome:
            cor = simpledialog.askstring("Cor", "Cor (ex: red, blue) ou vazio:")
            if cor and cor not in ['red','blue','green','orange','purple','cyan','magenta','lime','brown','pink','olive','teal','gold','coral','indigo','violet','darkgreen']:
                self._log(f"⚠️ Cor '{cor}' inválida, usando automática.")
                cor = None
            cid = self.class_manager.add_class(nome, color=cor)
            self._atualizar_lista_classes()
            self._log(f"✅ Classe '{nome}' adicionada (ID {cid})")

    def _remover_classe(self) -> None:
        selecao = self.lista_classes.curselection()
        if not selecao:
            self._log("⚠️ Selecione uma classe para remover.")
            return
        item = self.lista_classes.get(selecao[0])
        cid = int(item.split(':')[0])
        nome = self.class_manager.classes[cid]['name']
        self.class_manager.remove_class(cid)
        self._atualizar_lista_classes()
        self._log(f"🗑️ Classe '{nome}' removida.")

    def _renomear_classe(self) -> None:
        selecao = self.lista_classes.curselection()
        if not selecao:
            self._log("⚠️ Selecione uma classe para renomear.")
            return
        item = self.lista_classes.get(selecao[0])
        cid = int(item.split(':')[0])
        novo_nome = simpledialog.askstring("Renomear", "Novo nome:")
        if novo_nome:
            self.class_manager.rename_class(cid, novo_nome)
            self._atualizar_lista_classes()
            self._log(f"✏️ Classe renomeada para '{novo_nome}'")

    def _ativar_classe(self) -> None:
        selecao = self.lista_classes.curselection()
        if not selecao:
            self._log("⚠️ Selecione uma classe para ativar.")
            return
        item = self.lista_classes.get(selecao[0])
        cid = int(item.split(':')[0])
        if self.annotator is not None:
            self.annotator.set_class(cid)
        else:
            self._classe_para_ativar = cid
        self._log(f"🔵 Classe {cid} ativada")

    # ====== CONFIGURAÇÃO DE CLASSES ======
    def _carregar_configuracao_auto(self) -> None:
        destino = self.pasta_destino.get()
        if destino and os.path.exists(destino):
            if carregar_classes_config(self.class_manager, destino, self._log):
                self._atualizar_lista_classes()
                return
        if carregar_classes_config(self.class_manager, None, self._log):
            self._atualizar_lista_classes()
            return
        self._log("ℹ️ Nenhuma configuração de classes encontrada. Adicione classes manualmente.")

    def _salvar_configuracao_classes(self) -> None:
        destino = self.pasta_destino.get()
        if destino:
            if salvar_classes_config(self.class_manager, destino, self._log):
                messagebox.showinfo("Sucesso", f"Configuração salva em:\n{os.path.join(destino, CONFIG_FILE)}")
        else:
            if salvar_classes_config(self.class_manager, None, self._log):
                messagebox.showinfo("Sucesso", f"Configuração salva em:\n{CONFIG_FILE}")
        self._atualizar_lista_classes()

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
            self._atualizar_lista_classes()
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
            self._atualizar_lista_classes()
            self._log("🔄 Todas as classes foram removidas.")

    # ====== RADIO BUTTON ======
    def _radio_mode_changed(self) -> None:
        modo = self.modo_inicial.get()
        if self.annotator is not None:
            self.annotator.set_mode(modo)
            self._log(f"Modo alterado via radiobutton para: {modo.upper()}")

    def _on_mode_changed(self, modo: str) -> None:
        if modo in ('bbox', 'polygon'):
            self.modo_inicial.set(modo)

    # ====== AÇÕES ======
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

    # ====== INICIAR ANOTAÇÃO (COM THREADING) ======
    def _iniciar_anotacao(self) -> None:
        origem = self.pasta_origem.get()
        destino = self.pasta_destino.get()
        if not origem or not destino:
            messagebox.showerror("Erro", "Selecione as pastas de origem e destino.")
            return
        if not self.class_manager.classes:
            messagebox.showwarning("Aviso", "Adicione pelo menos uma classe.")
            return

        # Desabilita botão durante carregamento
        self.btn_carregar.config(state=tk.DISABLED)
        self.status_var.set("Carregando imagens...")

        # Inicia loader em thread
        loader = FolderLoader(origem, destino, self._on_imagens_carregadas, self._log)
        loader.start()

    def _on_imagens_carregadas(self, arquivos: List[str]) -> None:
        """Callback chamado quando o loader terminar."""
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

    def _processar_proxima_imagem(self):
        if self._indice_atual >= len(self._arquivos_pendentes):
            self._finalizar_sessao()
            return

        imagem_path = self._arquivos_pendentes[self._indice_atual]
        nome_arquivo = os.path.basename(imagem_path)
        self.progress_var.set(f"Imagem {self._indice_atual+1}/{self._total_imagens}")

        self._log(f"📷 Anotando: {nome_arquivo} ({self._indice_atual+1}/{self._total_imagens})")

        # === CORREÇÃO: Limpa anotações anteriores ===
        self.class_manager.clear_annotations()

        try:
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

        # Tenta carregar JSON, se não, YOLO
        if carregar_anotacoes_json(imagem_path, self.class_manager, self.pasta_destino.get(), self._log):
            self._log(f"   📂 Anotações carregadas do JSON.")
        elif carregar_anotacoes_de_yolo(imagem_path, self.class_manager, self.pasta_destino.get(), self._log):
            self._log(f"   📂 Anotações carregadas de arquivos YOLO.")
        else:
            self._log(f"   ℹ️ Nenhuma anotação existente encontrada.")

        self.annotator = Annotator(
            self.fig_container,
            imagem_path,
            img_array,
            self.class_manager,
            self._log,
            self.action_var,
            on_update_callback=self._atualizar_lista_classes,
            on_mode_change_callback=self._on_mode_changed,
            mode=self.modo_inicial.get()
        )

        if self._classe_para_ativar is not None:
            self.annotator.set_class(self._classe_para_ativar)
            self._classe_para_ativar = None

        self.annotator.fig.canvas.draw_idle()

        # Aguarda ação do usuário (wait_variable)
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
        self._atualizar_lista_classes()
        # Processa próxima imagem
        self._processar_proxima_imagem()

    def _salvar_anotacao(self, imagem_path: str) -> None:
        """Salva anotações em JSON e YOLO."""
        destino = self.pasta_destino.get()
        counts = self.class_manager.get_annotation_count()
        total_objetos = sum(c['total'] for c in counts.values())
        self._log(f"   📊 Resumo: {total_objetos} objetos")
        for cid, info in counts.items():
            self._log(f"      {info['name']}: {info['bboxes']} bboxes, {info['polygons']} polygons")

        if salvar_anotacoes_json(imagem_path, self.class_manager, destino, self._log):
            self._log(f"   ✅ JSON salvo em annotations/")
        else:
            self._log(f"   ⚠️ Erro ao salvar JSON")

        exporter = YoloExporter(self.class_manager, self._log)
        n_bbox, n_poly = exporter.export(imagem_path, destino)
        self._log(f"💾 Salva concluída: {os.path.basename(imagem_path)} (BBox: {n_bbox}, Polygon: {n_poly})")

    def _finalizar_sessao(self) -> None:
        """Finaliza a sessão de anotação."""
        self._log("✅ Anotação finalizada.")
        messagebox.showinfo("Concluído", "Anotação finalizada!")
        self.status_var.set("Pronto")
        self.progress_var.set("")
        self.btn_carregar.config(state=tk.NORMAL)

# ============================================================================
# FUNÇÃO PARA CRIAR A ABA
# ============================================================================
def create_annotation_tab(parent: tk.Frame) -> AnnotationTab:
    """Cria a aba de anotação para o notebook."""
    return AnnotationTab(parent)

# ============================================================================
# TESTE
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Teste - Aba de Anotação")
    root.geometry("1300x800")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    tab = create_annotation_tab(notebook)
    notebook.add(tab, text="Anotação")

    root.mainloop()