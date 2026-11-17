#!/usr/bin/env python3
"""
Ferramenta de anotação com suporte a Bounding Box e Polígono,
com classes dinâmicas e interface tkinter.
Console de logs, botões de ação e persistência em JSON.
"""

import os
import sys
import shutil
import json
import time
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path as PathLib
import traceback

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path as MPath
from matplotlib.widgets import RectangleSelector, PolygonSelector
from PIL import Image

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================
EXTENSOES_VALIDAS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
PALETA_CORES = [
    'red', 'blue', 'green', 'orange', 'purple', 'cyan',
    'magenta', 'lime', 'brown', 'pink', 'olive', 'teal',
    'gold', 'coral', 'indigo', 'violet', 'darkgreen'
]
BACKGROUND_KEYWORDS = ('bg', 'background', 'fundo', 'esteira')
CONFIG_FILE = 'classes_config.json'  # Nome do arquivo de configuração

# ============================================================================
# CLASS MANAGER
# ============================================================================
class ClassManager:
    def __init__(self):
        self.classes = {}
        self.next_id = 1
        self.available_ids = set()
        self._ny = None
        self._nx = None
        self._background_id = None

    def set_image_shape(self, shape):
        self._ny, self._nx = shape[:2]

    def add_class(self, name, color=None, desc=""):
        if color is None:
            color = PALETA_CORES[(self.next_id - 1) % len(PALETA_CORES)]
        if self.available_ids:
            class_id = min(self.available_ids)
            self.available_ids.remove(class_id)
        else:
            class_id = self.next_id
            self.next_id += 1

        is_bg = any(kw in name.lower() for kw in BACKGROUND_KEYWORDS)
        if is_bg and self._background_id is not None:
            print(f"Aviso: '{name}' não será fundo, já existe classe {self._background_id}.")
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
            'is_background': is_bg
        }
        return class_id

    def remove_class(self, class_id):
        if class_id in self.classes:
            if self._background_id == class_id:
                self._background_id = None
            del self.classes[class_id]
            self.available_ids.add(class_id)

    def rename_class(self, class_id, new_name):
        if class_id not in self.classes:
            return
        self.classes[class_id]['name'] = new_name
        is_bg = any(kw in new_name.lower() for kw in BACKGROUND_KEYWORDS)
        if is_bg and self._background_id is not None and self._background_id != class_id:
            print(f"Aviso: '{new_name}' não será fundo, já existe classe {self._background_id}.")
            is_bg = False
        self.classes[class_id]['is_background'] = is_bg
        if is_bg:
            self._background_id = class_id
        elif self._background_id == class_id:
            self._background_id = None

    def get_class(self, class_id):
        return self.classes.get(class_id)

    def get_class_name(self, class_id):
        info = self.classes.get(class_id)
        return info['name'] if info else None

    def get_all_classes(self):
        return self.classes

    def get_class_list(self):
        result = []
        for cid, info in self.classes.items():
            total = len(info['bboxes']) + len(info['polygons'])
            result.append((cid, info['name'], info['color'], total))
        return result

    def get_background_id(self):
        return self._background_id

    def get_sorted_non_background_ids(self):
        """Retorna os IDs das classes que não são fundo, ordenados."""
        return sorted([cid for cid, info in self.classes.items() if not info['is_background']])

    def add_bbox(self, class_id, x1, y1, x2, y2):
        if class_id not in self.classes or self._ny is None:
            return False
        x1 = max(0, min(x1, self._nx))
        x2 = max(0, min(x2, self._nx))
        y1 = max(0, min(y1, self._ny))
        y2 = max(0, min(y2, self._ny))
        if (x2-x1) < 2 or (y2-y1) < 2:
            return False
        self.classes[class_id]['bboxes'].append((x1, y1, x2, y2))
        # Recalcula a máscara completa
        self._recalc_mask(class_id)
        return True

    def add_polygon_mask(self, class_id, polygon_verts):
        if class_id not in self.classes or self._ny is None:
            return False
        path = MPath(polygon_verts)
        x, y = np.meshgrid(np.arange(self._nx), np.arange(self._ny))
        coords = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
        mask = path.contains_points(coords).reshape(self._ny, self._nx)
        if mask.sum() == 0:
            return False
        self.classes[class_id]['polygons'].append(polygon_verts)
        # Recalcula a máscara completa
        self._recalc_mask(class_id)
        return True

    def _recalc_mask(self, class_id):
        """Recalcula a máscara completa para uma classe a partir das anotações atuais."""
        if class_id not in self.classes or self._ny is None:
            return
        info = self.classes[class_id]
        total_mask = np.zeros((self._ny, self._nx), dtype=bool)
        
        # Adiciona polígonos
        for verts in info['polygons']:
            path = MPath(verts)
            x, y = np.meshgrid(np.arange(self._nx), np.arange(self._ny))
            coords = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
            total_mask |= path.contains_points(coords).reshape(self._ny, self._nx)
        
        # Adiciona bounding boxes
        for (x1,y1,x2,y2) in info['bboxes']:
            total_mask[int(y1):int(y2), int(x1):int(x2)] = True
        
        info['mask'] = total_mask if total_mask.any() else None

    def undo_last(self, class_id):
        """Remove a última anotação (bbox ou polygon) de uma classe."""
        if class_id not in self.classes:
            return False
        info = self.classes[class_id]
        
        # Remove a última anotação (polygon ou bbox)
        if info['polygons']:
            removed = info['polygons'].pop()
            removed_type = 'polygon'
        elif info['bboxes']:
            removed = info['bboxes'].pop()
            removed_type = 'bbox'
        else:
            return False
        
        # Recalcula a máscara
        self._recalc_mask(class_id)
        return True

    def clear_annotations(self):
        """Limpa todas as anotações (bboxes, polígonos e máscaras) de todas as classes."""
        for cid in self.classes:
            self.classes[cid]['bboxes'] = []
            self.classes[cid]['polygons'] = []
            self.classes[cid]['mask'] = None

    def get_annotation_count(self):
        """Retorna o total de anotações por classe."""
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

    def get_mask(self, class_id):
        info = self.classes.get(class_id)
        if info is None or self._ny is None:
            return None
        if not info['is_background']:
            return info['mask']
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

    def get_pixel_count(self, class_id):
        mask = self.get_mask(class_id)
        return mask.sum() if mask is not None else 0
    
    def remove_empty_classes(self):
        """Remove classes que não têm anotações (bboxes nem polygons)."""
        to_remove = []
        for cid, info in self.classes.items():
            if not info['is_background']:
                if len(info['bboxes']) == 0 and len(info['polygons']) == 0:
                    to_remove.append(cid)
        
        for cid in to_remove:
            self.remove_class(cid)
        
        return len(to_remove)

    def to_dict(self):
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

    def from_dict(self, data):
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
                'is_background': info.get('is_background', False)
            }
            if info.get('is_background', False):
                self._background_id = cid

# ============================================================================
# FUNÇÕES DE PERSISTÊNCIA DE CLASSES
# ============================================================================
def salvar_classes_config(class_manager, pasta_destino=None, log_callback=None):
    """Salva a configuração das classes em um arquivo JSON."""
    if pasta_destino is None:
        # Se não especificado, salva no diretório atual
        config_path = CONFIG_FILE
    else:
        config_path = os.path.join(pasta_destino, CONFIG_FILE)
    
    data = class_manager.to_dict()
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        if log_callback:
            log_callback(f"✅ Configuração de classes salva em: {config_path}")
            log_callback(f"   {len(data['classes'])} classes salvas")
        
        return True
    except Exception as e:
        if log_callback:
            log_callback(f"❌ Erro ao salvar configuração: {e}")
        traceback.print_exc()
        return False

def carregar_classes_config(class_manager, pasta_destino=None, log_callback=None):
    """Carrega a configuração das classes de um arquivo JSON."""
    if pasta_destino is None:
        # Se não especificado, tenta carregar do diretório atual
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
        if log_callback:
            log_callback(f"❌ Erro ao carregar configuração: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# FUNÇÕES DE PERSISTÊNCIA DE ANOTAÇÕES (JSON)
# ============================================================================
def salvar_anotacoes_json(image_path, class_manager, pasta_destino, log_callback=None):
    """Salva todas as anotações (bboxes e polígonos) em um arquivo JSON."""
    # Remove classes vazias antes de salvar
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
        # Só salva classes que têm anotações
        if len(info['bboxes']) > 0 or len(info['polygons']) > 0:
            data['classes'][str(cid)] = {
                'name': info['name'],
                'color': info['color'],
                'bboxes': info['bboxes'],
                'polygons': info['polygons']
            }
    
    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Log detalhado do que foi salvo
        if log_callback:
            log_callback(f"   📝 JSON salvo com {len(data['classes'])} classes")
            for cid_str, cls_data in data['classes'].items():
                log_callback(f"      Classe {cid_str} ({cls_data['name']}): "
                           f"{len(cls_data['bboxes'])} bboxes, {len(cls_data['polygons'])} polygons")
        return True
    except Exception as e:
        print(f"Erro ao salvar JSON {json_path}: {e}")
        traceback.print_exc()
        return False

def carregar_anotacoes_json(image_path, class_manager, pasta_destino, log_callback=None):
    """Carrega anotações de um arquivo JSON, se existir."""
    nome_base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(pasta_destino, 'annotations', nome_base + '.json')
    
    if not os.path.exists(json_path):
        return False
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Verifica se a shape bate
        if class_manager._ny is None or class_manager._nx is None:
            print("Erro: dimensões da imagem não definidas")
            return False
            
        if data['shape'][0] != class_manager._ny or data['shape'][1] != class_manager._nx:
            print(f"Aviso: shape da imagem mudou, ignorando JSON antigo para {nome_base}")
            return False
        
        # Limpa anotações atuais antes de carregar
        class_manager.clear_annotations()
        
        for cid_str, info in data['classes'].items():
            cid = int(cid_str)
            
            # Garante que a classe existe
            if cid not in class_manager.classes:
                # Cria a classe com o ID correto
                class_manager.classes[cid] = {
                    'name': info['name'],
                    'color': info['color'],
                    'mask': None,
                    'polygons': [],
                    'bboxes': [],
                    'desc': '',
                    'is_background': False
                }
                if cid >= class_manager.next_id:
                    class_manager.next_id = cid + 1
                if cid in class_manager.available_ids:
                    class_manager.available_ids.remove(cid)
            
            # Adiciona as anotações (bboxes)
            for (x1,y1,x2,y2) in info.get('bboxes', []):
                class_manager.add_bbox(cid, x1, y1, x2, y2)
            
            # Adiciona as anotações (polygons)
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
        print(f"Erro ao carregar JSON {json_path}: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# FUNÇÕES DE EXPORTAÇÃO YOLO (melhoradas com logs)
# ============================================================================
def export_yolo_detection(image_path, class_manager, pasta_destino, log_callback=None):
    """Exporta bounding boxes para pasta labels/bbox/"""
    pasta_imgs = os.path.join(pasta_destino, 'images')
    pasta_labels = os.path.join(pasta_destino, 'labels', 'bbox')
    os.makedirs(pasta_imgs, exist_ok=True)
    os.makedirs(pasta_labels, exist_ok=True)

    nome_base = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    destino_img = os.path.join(pasta_imgs, nome_base + ext)
    destino_txt = os.path.join(pasta_labels, nome_base + '.txt')
    
    # Copia a imagem (sobrescreve se existir)
    try:
        shutil.copy2(image_path, destino_img)
    except Exception as e:
        print(f"Erro ao copiar imagem para {destino_img}: {e}")
        return 0

    h, w = class_manager._ny, class_manager._nx
    if h is None or w is None:
        print("Erro: dimensões da imagem não definidas")
        return 0
        
    sorted_ids = class_manager.get_sorted_non_background_ids()
    
    # Cria um dicionário para mapear ID -> índice YOLO
    id_to_yolo = {cid: idx for idx, cid in enumerate(sorted_ids)}
    
    linhas = []
    objetos_por_classe = {}
    
    for cid, info in class_manager.classes.items():
        if info['is_background']:
            continue
        if cid not in id_to_yolo:
            continue
        
        yolo_idx = id_to_yolo[cid]
        nome_classe = info['name']
        objetos_por_classe[nome_classe] = 0
        
        for (x1,y1,x2,y2) in info['bboxes']:
            # Garante coordenadas dentro dos limites
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            
            # Garante que os valores estejam entre 0 e 1
            cx = max(0, min(cx, 1))
            cy = max(0, min(cy, 1))
            bw = max(0, min(bw, 1))
            bh = max(0, min(bh, 1))
            
            linhas.append(f"{yolo_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            objetos_por_classe[nome_classe] += 1
    
    try:
        with open(destino_txt, 'w') as f:
            f.write('\n'.join(linhas))
        
        if log_callback:
            log_callback(f"   📦 BBox exportado: {destino_txt}")
            for nome_classe, count in objetos_por_classe.items():
                if count > 0:
                    log_callback(f"      {nome_classe}: {count} objetos")
        
        return len(linhas)
    except Exception as e:
        print(f"Erro ao salvar {destino_txt}: {e}")
        return 0

def export_yolo_segmentation(image_path, class_manager, pasta_destino, log_callback=None):
    """Exporta polígonos para pasta labels/polygon/"""
    pasta_imgs = os.path.join(pasta_destino, 'images')
    pasta_labels = os.path.join(pasta_destino, 'labels', 'polygon')
    os.makedirs(pasta_imgs, exist_ok=True)
    os.makedirs(pasta_labels, exist_ok=True)

    nome_base = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    destino_img = os.path.join(pasta_imgs, nome_base + ext)
    destino_txt = os.path.join(pasta_labels, nome_base + '.txt')
    
    try:
        shutil.copy2(image_path, destino_img)
    except Exception as e:
        print(f"Erro ao copiar imagem para {destino_img}: {e}")
        return 0

    h, w = class_manager._ny, class_manager._nx
    if h is None or w is None:
        print("Erro: dimensões da imagem não definidas")
        return 0
        
    sorted_ids = class_manager.get_sorted_non_background_ids()
    id_to_yolo = {cid: idx for idx, cid in enumerate(sorted_ids)}
    
    linhas = []
    objetos_por_classe = {}
    
    for cid, info in class_manager.classes.items():
        if info['is_background']:
            continue
        if cid not in id_to_yolo:
            continue
        
        yolo_idx = id_to_yolo[cid]
        nome_classe = info['name']
        objetos_por_classe[nome_classe] = 0
        
        for verts in info['polygons']:
            if len(verts) < 3:
                continue
            norm_pts = []
            for (x, y) in verts:
                # Garante coordenadas dentro dos limites
                x = max(0, min(x, w))
                y = max(0, min(y, h))
                norm_pts.append(f"{x/w:.6f}")
                norm_pts.append(f"{y/h:.6f}")
            linha = f"{yolo_idx} " + " ".join(norm_pts)
            linhas.append(linha)
            objetos_por_classe[nome_classe] += 1
    
    try:
        with open(destino_txt, 'w') as f:
            f.write('\n'.join(linhas))
        
        if log_callback:
            log_callback(f"   🔷 Polygon exportado: {destino_txt}")
            for nome_classe, count in objetos_por_classe.items():
                if count > 0:
                    log_callback(f"      {nome_classe}: {count} objetos")
        
        return len(linhas)
    except Exception as e:
        print(f"Erro ao salvar {destino_txt}: {e}")
        return 0

def export_all_formats(image_path, class_manager, pasta_destino, log_callback=None):
    """Exporta anotações em todos os formatos suportados"""
    if log_callback:
        log_callback(f"   📤 Exportando anotações...")
    
    n_bbox = export_yolo_detection(image_path, class_manager, pasta_destino, log_callback)
    n_poly = export_yolo_segmentation(image_path, class_manager, pasta_destino, log_callback)
    
    if log_callback:
        log_callback(f"   ✅ Exportação concluída: {n_bbox} bboxes, {n_poly} polygons")
    
    return n_bbox, n_poly


# ============================================================================
# ANOTADOR (MATPLOTLIB) - COM UNDO DE PONTOS DO POLÍGONO 
# ============================================================================
class Annotator:
    def __init__(self, parent_frame, image_path, img_array, class_manager, 
                 log_callback, action_var, on_update_callback, on_mode_change_callback, mode='bbox'):
        self.parent_frame = parent_frame
        self.image_path = image_path
        self.img_array = img_array
        self.class_manager = class_manager
        self.log_callback = log_callback
        self.action_var = action_var  # tk.StringVar para controle de fluxo
        self.on_update_callback = on_update_callback
        self.on_mode_change_callback = on_mode_change_callback
        self.mode = mode  # 'bbox' ou 'polygon'
        self.h, self.w = img_array.shape[0], img_array.shape[1]
        self.class_manager.set_image_shape((self.h, self.w))

        self.current_class_id = None
        self.acao = None  # 'salvar', 'pular', 'sair'
        
        # Para controle do polígono
        self.polygon_points = []  # Pontos temporários do polígono atual
        self.polygon_artists = []  # Para desenhar o polígono em andamento
        self.is_drawing_polygon = False

        self.fig = plt.Figure(figsize=(10, 8), facecolor='lightgray')
        self.ax_image = self.fig.add_axes([0.05, 0.05, 0.90, 0.90])
        self.ax_image.imshow(self.img_array)
        self.ax_image.axis('on')
        self.atualizar_titulo()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.parent_frame)
        self.toolbar.update()

        self.rect_selector = None
        self.poly_selector = None
        self.inicializar_selector()

        # Conecta eventos para controle manual do polígono
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.fig.canvas.draw_idle()

    def inicializar_selector(self):
        if self.rect_selector:
            self.rect_selector.disconnect_events()
            self.rect_selector = None
        if self.poly_selector:
            self.poly_selector.disconnect_events()
            self.poly_selector = None
            self.poly_selector = None

        # Limpa pontos temporários do polígono
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
            # Não usamos mais o PolygonSelector padrão, pois ele salva a cada clique
            # Vamos usar eventos de mouse manualmente
            self.poly_selector = None
            self.is_drawing_polygon = False
            self.polygon_points = []
            self._clear_polygon_preview()
            
            # Instrução para o usuário
            self.log("🖊️ Clique para adicionar pontos. Clique no primeiro ponto ou dê duplo clique para fechar o polígono.")
            self.log("   Use Z para desfazer o último ponto.")

    def _clear_polygon_preview(self):
        """Remove a visualização temporária do polígono."""
        for artist in self.polygon_artists:
            try:
                artist.remove()
            except:
                pass
        self.polygon_artists = []
        self.fig.canvas.draw_idle()

    def _update_polygon_preview(self):
        """Atualiza a visualização do polígono em andamento."""
        self._clear_polygon_preview()
        
        if len(self.polygon_points) < 1:
            self.atualizar_titulo()
            return
        
        # Desenha os pontos
        points = np.array(self.polygon_points)
        scatter = self.ax_image.scatter(points[:, 0], points[:, 1], 
                                       color='yellow', s=50, zorder=5)
        self.polygon_artists.append(scatter)
        
        # Desenha as linhas entre os pontos
        if len(self.polygon_points) > 1:
            line = plt.Line2D(points[:, 0], points[:, 1], 
                             color='yellow', linewidth=2, zorder=4)
            self.ax_image.add_line(line)
            self.polygon_artists.append(line)
        
        self.atualizar_titulo()
        self.fig.canvas.draw_idle()

    def _undo_polygon_point(self):
        """Desfaz o último ponto adicionado ao polígono em andamento."""
        if not self.is_drawing_polygon or len(self.polygon_points) == 0:
            self.log("ℹ️ Nenhum ponto para desfazer.")
            return False
        
        # Remove o último ponto
        removed_point = self.polygon_points.pop()
        self.log(f"↩️ Ponto removido: ({removed_point[0]:.1f}, {removed_point[1]:.1f}) - {len(self.polygon_points)} pontos restantes")
        
        # Se não tiver mais pontos, cancela o desenho
        if len(self.polygon_points) == 0:
            self.is_drawing_polygon = False
            self._clear_polygon_preview()
            self.log("   Desenho do polígono cancelado.")
        else:
            self._update_polygon_preview()
        
        return True

    def on_mouse_click(self, event):
        """Manipula cliques do mouse para desenho de polígonos."""
        # Só processa se estiver no modo polygon e dentro da imagem
        if self.mode != 'polygon' or event.inaxes != self.ax_image:
            return
        
        if self.current_class_id is None:
            self.log("⚠️ Selecione uma classe primeiro!")
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # Verifica se está dentro dos limites da imagem
        if x < 0 or x > self.w or y < 0 or y > self.h:
            return
        
        # Se não está desenhando, inicia um novo polígono
        if not self.is_drawing_polygon:
            self.is_drawing_polygon = True
            self.polygon_points = [(x, y)]
            self._update_polygon_preview()
            self.log(f"   🖊️ Iniciando polígono em ({x:.1f}, {y:.1f})")
            return
        
        # Verifica se clicou no primeiro ponto (fechar polígono)
        if len(self.polygon_points) >= 3:
            first_point = self.polygon_points[0]
            # Distância do clique ao primeiro ponto
            dist = np.sqrt((x - first_point[0])**2 + (y - first_point[1])**2)
            if dist < 15:  # Tolerância de 15 pixels
                # Fecha o polígono
                self._finalize_polygon()
                return
        
        # Adiciona o ponto ao polígono
        self.polygon_points.append((x, y))
        self._update_polygon_preview()
        self.log(f"   🖊️ Ponto adicionado: ({x:.1f}, {y:.1f}) - {len(self.polygon_points)} pontos")

    def _finalize_polygon(self):
        """Finaliza o polígono atual e salva."""
        if len(self.polygon_points) < 3:
            self.log("⚠️ Polígono precisa de pelo menos 3 pontos.")
            self.is_drawing_polygon = False
            self.polygon_points = []
            self._clear_polygon_preview()
            return
        
        # Fecha o polígono (garante que o último ponto seja o primeiro)
        if self.polygon_points[0] != self.polygon_points[-1]:
            self.polygon_points.append(self.polygon_points[0])
        
        # Salva o polígono
        if self.class_manager.add_polygon_mask(self.current_class_id, self.polygon_points):
            cor = self.class_manager.classes[self.current_class_id]['color']
            poly = Polygon(self.polygon_points, linewidth=2, edgecolor=cor, facecolor='none')
            self.ax_image.add_patch(poly)
            
            # Mostra contagem atualizada
            counts = self.class_manager.get_annotation_count()
            total = sum(c['total'] for c in counts.values())
            self.log(f"✅ Polígono finalizado e salvo (classe {self.class_manager.classes[self.current_class_id]['name']}) - Total: {total}")
            self.atualizar_titulo()
            self._notify_update()
        else:
            self.log("❌ Falha ao adicionar polígono (máscara vazia?).")
        
        # Limpa o estado do polígono
        self.is_drawing_polygon = False
        self.polygon_points = []
        self._clear_polygon_preview()
        self.fig.canvas.draw_idle()

    def set_mode(self, mode):
        """Define o modo de anotação (bbox ou polygon) e recria o seletor."""
        if mode not in ('bbox', 'polygon'):
            return
        self.mode = mode
        # Limpa qualquer desenho de polígono em andamento
        self.is_drawing_polygon = False
        self.polygon_points = []
        self._clear_polygon_preview()
        
        self.inicializar_selector()
        self.atualizar_titulo()
        self.log(f"Modo alterado para: {self.mode.upper()}")
        if self.on_mode_change_callback:
            self.on_mode_change_callback(mode)

    def toggle_mode(self):
        """Alterna entre bbox e polygon."""
        new_mode = 'polygon' if self.mode == 'bbox' else 'bbox'
        self.set_mode(new_mode)

    def set_class(self, class_id):
        if class_id in self.class_manager.classes:
            self.current_class_id = class_id
            # Limpa qualquer desenho de polígono em andamento
            self.is_drawing_polygon = False
            self.polygon_points = []
            self._clear_polygon_preview()
            self.inicializar_selector()
            self.atualizar_titulo()
            self.fig.canvas.draw_idle()
            self.log(f"Classe ativada: {self.class_manager.classes[class_id]['name']} (ID {class_id})")

    def atualizar_titulo(self):
        nome_arquivo = os.path.basename(self.image_path)
        nome_classe = "Nenhuma" if self.current_class_id is None else self.class_manager.get_class_name(self.current_class_id)
        cor = "gray" if self.current_class_id is None else self.class_manager.classes[self.current_class_id]['color']
        
        # Adiciona contagem de anotações
        counts = self.class_manager.get_annotation_count()
        info_text = ""
        if counts:
            total = sum(c['total'] for c in counts.values())
            info_text = f" | Total: {total} objetos"
        
        # Se estiver desenhando polígono, mostra quantos pontos
        if self.mode == 'polygon' and self.is_drawing_polygon and len(self.polygon_points) > 0:
            info_text += f" | Desenhando: {len(self.polygon_points)} pontos (Z para desfazer)"
        
        self.ax_image.set_title(
            f"{nome_arquivo}\nClasse: {nome_classe}  |  Modo: {self.mode.upper()}{info_text}",
            fontsize=11, color=cor
        )

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)

    def onselect_bbox(self, eclick, erelease):
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
        if (x2-x1) < 3 or (y2-y1) < 3:
            self.log("⚠️ Caixa muito pequena, ignorada.")
            return
        if self.class_manager.add_bbox(self.current_class_id, x1, y1, x2, y2):
            cor = self.class_manager.classes[self.current_class_id]['color']
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=cor, facecolor='none')
            self.ax_image.add_patch(rect)
            self.fig.canvas.draw_idle()
            
            # Mostra contagem atualizada
            counts = self.class_manager.get_annotation_count()
            total = sum(c['total'] for c in counts.values())
            self.log(f"✅ BBox adicionada (classe {self.class_manager.classes[self.current_class_id]['name']}) - Total: {total}")
            self.atualizar_titulo()
            self._notify_update()
        else:
            self.log("❌ Falha ao adicionar BBox.")

    def _notify_update(self):
        if self.on_update_callback:
            self.on_update_callback()

    def undo(self, event=None):
        """Desfaz a última ação: se estiver desenhando polígono, remove o último ponto."""
        # Se estiver desenhando um polígono, desfaz o último ponto
        if self.is_drawing_polygon and len(self.polygon_points) > 0:
            self._undo_polygon_point()
            return
        
        # Se não estiver desenhando, desfaz a última anotação salva
        if self.current_class_id is None:
            self.log("⚠️ Nenhuma classe ativa para desfazer.")
            return
        
        # Verifica se há algo para desfazer
        info = self.class_manager.classes.get(self.current_class_id)
        if not info or (not info['polygons'] and not info['bboxes']):
            self.log("ℹ️ Nada para desfazer.")
            return
        
        # Remove a última anotação
        if self.class_manager.undo_last(self.current_class_id):
            self.redesenhar_objetos()
            self._notify_update()
            
            # Mostra contagem atualizada
            counts = self.class_manager.get_annotation_count()
            total = sum(c['total'] for c in counts.values())
            self.log(f"↩️ Último objeto desfeito (classe {self.class_manager.classes[self.current_class_id]['name']}) - Total: {total}")
            self.atualizar_titulo()
        else:
            self.log("❌ Erro ao desfazer.")

    def redesenhar_objetos(self):
        self.ax_image.clear()
        self.ax_image.imshow(self.img_array)
        self.ax_image.axis('on')
        self.atualizar_titulo()
        for cid, info in self.class_manager.classes.items():
            if info['is_background']:
                continue
            cor = info['color']
            for (x1,y1,x2,y2) in info['bboxes']:
                rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=cor, facecolor='none')
                self.ax_image.add_patch(rect)
            for verts in info['polygons']:
                poly = Polygon(verts, linewidth=2, edgecolor=cor, facecolor='none')
                self.ax_image.add_patch(poly)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        """Manipula eventos de teclado."""
        classes = [cid for cid, info in sorted(self.class_manager.classes.items()) if not info['is_background']]
        
        # Teclas numéricas para selecionar classes
        if event.key in '1234567890':
            idx = int(event.key) - 1 if event.key != '0' else 9
            if idx < len(classes):
                self.set_class(classes[idx])
        
        # Z: desfaz ponto do polígono ou última anotação
        elif event.key == 'z':
            self.undo()
        
        # M: alternar modo
        elif event.key == 'm':
            self.toggle_mode()
        
        # N ou Enter: salvar ou finalizar polígono
        elif event.key in ('n', 'enter'):
            # Se estiver desenhando polígono, finaliza
            if self.is_drawing_polygon and len(self.polygon_points) >= 3:
                self._finalize_polygon()
            else:
                self.set_acao('salvar')
        
        # P: pular imagem
        elif event.key == 'p':
            self.set_acao('pular')
        
        # Q: sair
        elif event.key == 'q':
            self.set_acao('sair')
        
        # ESC: cancelar desenho do polígono
        elif event.key == 'escape':
            if self.is_drawing_polygon:
                self.is_drawing_polygon = False
                self.polygon_points = []
                self._clear_polygon_preview()
                self.log("↩️ Desenho do polígono cancelado.")
                self.atualizar_titulo()

    def set_acao(self, acao):
        self.acao = acao
        self.log(f"⏳ Finalizando com ação: {acao}")
        # Define a variável para desbloquear o wait_variable na main
        if self.action_var:
            self.action_var.set(acao)
        # Fecha a figura de forma segura
        try:
            self.canvas.get_tk_widget().destroy()
        except Exception as e:
            self.log(f"Erro ao destruir canvas: {e}")
        try:
            plt.close(self.fig)
        except Exception as e:
            self.log(f"Erro ao fechar figura: {e}")

# ============================================================================
# JANELA PRINCIPAL TKINTER - COM PERSISTÊNCIA DE CLASSES
# ============================================================================
class AnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ferramenta de Anotação - YOLO (BBox/Polígono)")
        self.root.geometry("1300x750")

        self.pasta_origem = tk.StringVar()
        self.pasta_destino = tk.StringVar()
        self.modo_inicial = tk.StringVar(value="bbox")
        self.class_manager = ClassManager()
        self.annotator = None
        self.action_var = tk.StringVar()  # Para controle do loop

        self.criar_widgets()
        
        # Tenta carregar a configuração de classes automaticamente
        self.carregar_configuracao_auto()

    def criar_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ===== PAINEL ESQUERDO =====
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

        # ===== PAINEL DIREITO =====
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

        controls_inner.bind("<MouseWheel>", lambda e: canvas_ctrl.yview_scroll(int(-1*(e.delta/120)), "units"))

        # ===== SEÇÕES DE CONTROLE =====
        # Pastas
        folder_frame = ttk.LabelFrame(controls_inner, text="Pastas")
        folder_frame.pack(fill=tk.X, pady=5)
        ttk.Label(folder_frame, text="Origem:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(folder_frame, textvariable=self.pasta_origem, width=20).grid(row=0, column=1, padx=2)
        ttk.Button(folder_frame, text="Browse", command=self.selecionar_pasta_origem, width=8).grid(row=0, column=2)
        ttk.Label(folder_frame, text="Destino:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(folder_frame, textvariable=self.pasta_destino, width=20).grid(row=1, column=1, padx=2)
        ttk.Button(folder_frame, text="Browse", command=self.selecionar_pasta_destino, width=8).grid(row=1, column=2)

        # ===== NOVO: Gerenciamento de Configuração de Classes =====
        config_frame = ttk.LabelFrame(controls_inner, text="Configuração de Classes")
        config_frame.pack(fill=tk.X, pady=5)
        
        btn_config_frame = ttk.Frame(config_frame)
        btn_config_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_config_frame, text="💾 Salvar Classes", 
                  command=self.salvar_configuracao_classes, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_config_frame, text="📂 Carregar Classes", 
                  command=self.carregar_configuracao_classes, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_config_frame, text="🔄 Resetar Classes", 
                  command=self.resetar_classes, width=15).pack(side=tk.LEFT, padx=2)

        # Modo inicial (radiobuttons)
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
        self.lista_classes = tk.Listbox(list_frame, height=5, selectmode=tk.SINGLE)
        self.lista_classes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_list = ttk.Scrollbar(list_frame, orient="vertical", command=self.lista_classes.yview)
        scroll_list.pack(side=tk.RIGHT, fill=tk.Y)
        self.lista_classes.config(yscrollcommand=scroll_list.set)

        btn_ger = ttk.Frame(class_frame)
        btn_ger.pack(fill=tk.X, pady=2)
        ttk.Button(btn_ger, text="Adicionar", command=self.adicionar_classe, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_ger, text="Remover", command=self.remover_classe, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_ger, text="Renomear", command=self.renomear_classe, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_ger, text="Ativar", command=self.ativar_classe, width=10).pack(side=tk.LEFT, padx=2)

        # Ações
        action_frame = ttk.LabelFrame(controls_inner, text="Ações durante anotação")
        action_frame.pack(fill=tk.X, pady=5)

        btn_action_frame = ttk.Frame(action_frame)
        btn_action_frame.pack(fill=tk.X, pady=2)

        self.btn_salvar = ttk.Button(btn_action_frame, text="Salvar e Próximo (N)",
                                     command=self.acao_salvar, width=18)
        self.btn_salvar.pack(side=tk.LEFT, padx=2)

        self.btn_pular = ttk.Button(btn_action_frame, text="Pular (P)",
                                    command=self.acao_pular, width=12)
        self.btn_pular.pack(side=tk.LEFT, padx=2)

        self.btn_sair = ttk.Button(btn_action_frame, text="Sair (Q)",
                                   command=self.acao_sair, width=12)
        self.btn_sair.pack(side=tk.LEFT, padx=2)

        ttk.Button(action_frame, text="Desfazer (Z)", command=self.acao_desfazer, width=20).pack(pady=2)

        # Botão Iniciar
        ttk.Button(controls_inner, text="Iniciar Anotação", command=self.iniciar_anotacao).pack(fill=tk.X, pady=5)

        # Console de Logs (maior)
        log_frame = ttk.LabelFrame(controls_inner, text="Console de Logs")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = tk.Text(log_frame, height=8, state='disabled', wrap=tk.WORD,
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

        # Status
        self.status_var = tk.StringVar(value="Pronto")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.atualizar_lista_classes()
        self.log("🟢 Ferramenta iniciada. Configure as pastas e classes.")

    # ===== MÉTODOS DE PERSISTÊNCIA DE CLASSES =====
    def carregar_configuracao_auto(self):
        """Tenta carregar a configuração automaticamente ao iniciar."""
        # Primeiro tenta carregar da pasta de destino (se definida)
        destino = self.pasta_destino.get()
        if destino and os.path.exists(destino):
            if carregar_classes_config(self.class_manager, destino, self.log):
                self.atualizar_lista_classes()
                return
        
        # Se não encontrou na pasta de destino, tenta no diretório atual
        if carregar_classes_config(self.class_manager, None, self.log):
            self.atualizar_lista_classes()
            return
        
        self.log("ℹ️ Nenhuma configuração de classes encontrada. Adicione classes manualmente.")

    def salvar_configuracao_classes(self):
        """Salva a configuração atual das classes."""
        destino = self.pasta_destino.get()
        if destino:
            # Salva na pasta de destino
            if salvar_classes_config(self.class_manager, destino, self.log):
                messagebox.showinfo("Sucesso", f"Configuração salva em:\n{os.path.join(destino, CONFIG_FILE)}")
        else:
            # Salva no diretório atual
            if salvar_classes_config(self.class_manager, None, self.log):
                messagebox.showinfo("Sucesso", f"Configuração salva em:\n{CONFIG_FILE}")
        
        self.atualizar_lista_classes()

    def carregar_configuracao_classes(self):
        """Carrega uma configuração de classes de um arquivo."""
        # Pergunta onde carregar
        destino = self.pasta_destino.get()
        if destino and os.path.exists(destino):
            config_path = os.path.join(destino, CONFIG_FILE)
            if os.path.exists(config_path):
                if carregar_classes_config(self.class_manager, destino, self.log):
                    self.atualizar_lista_classes()
                    messagebox.showinfo("Sucesso", f"Configuração carregada de:\n{config_path}")
                    return
        
        # Tenta carregar do diretório atual
        if carregar_classes_config(self.class_manager, None, self.log):
            self.atualizar_lista_classes()
            messagebox.showinfo("Sucesso", f"Configuração carregada de:\n{CONFIG_FILE}")
        else:
            # Pergunta se quer selecionar um arquivo manualmente
            resposta = messagebox.askyesno("Arquivo não encontrado", 
                                          "Deseja selecionar um arquivo de configuração manualmente?")
            if resposta:
                arquivo = filedialog.askopenfilename(
                    title="Selecionar arquivo de configuração",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
                )
                if arquivo:
                    try:
                        with open(arquivo, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        self.class_manager.from_dict(data)
                        self.atualizar_lista_classes()
                        self.log(f"✅ Configuração carregada de: {arquivo}")
                        messagebox.showinfo("Sucesso", f"Configuração carregada de:\n{arquivo}")
                    except Exception as e:
                        self.log(f"❌ Erro ao carregar arquivo: {e}")
                        messagebox.showerror("Erro", f"Erro ao carregar arquivo:\n{e}")

    def resetar_classes(self):
        """Remove todas as classes atuais."""
        if not self.class_manager.classes:
            self.log("ℹ️ Não há classes para resetar.")
            return
        
        resposta = messagebox.askyesno("Confirmar", 
                                      "Tem certeza que deseja remover TODAS as classes?")
        if resposta:
            # Limpa todas as classes
            for cid in list(self.class_manager.classes.keys()):
                self.class_manager.remove_class(cid)
            self.atualizar_lista_classes()
            self.log("🔄 Todas as classes foram removidas.")

    # ===== MÉTODO PARA SINCRONIZAR RADIOBUTTON COM MODO =====
    def _radio_mode_changed(self):
        """Quando o usuário clica no radiobutton, atualiza o modo do anotador se estiver ativo."""
        modo = self.modo_inicial.get()
        if self.annotator is not None:
            self.annotator.set_mode(modo)
            self.log(f"Modo alterado via radiobutton para: {modo.upper()}")

    def _on_mode_changed(self, modo):
        """Callback chamado pelo Annotator quando o modo muda via tecla (M)."""
        if modo in ('bbox', 'polygon'):
            self.modo_inicial.set(modo)
            # Não precisa logar novamente, o Annotator já loga

    # ===== MÉTODOS DE LOG =====
    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.status_var.set(msg[:60])

    # ===== MÉTODOS DE SELEÇÃO DE PASTAS =====
    def selecionar_pasta_origem(self):
        pasta = filedialog.askdirectory()
        if pasta:
            self.pasta_origem.set(pasta)
            self.log(f"📂 Pasta de origem: {pasta}")

    def selecionar_pasta_destino(self):
        pasta = filedialog.askdirectory()
        if pasta:
            self.pasta_destino.set(pasta)
            self.log(f"📂 Pasta de destino: {pasta}")
            # Tenta carregar configuração da nova pasta de destino
            config_path = os.path.join(pasta, CONFIG_FILE)
            if os.path.exists(config_path):
                if carregar_classes_config(self.class_manager, pasta, self.log):
                    self.atualizar_lista_classes()

    # ===== GERENCIAMENTO DE CLASSES =====
    def atualizar_lista_classes(self):
        self.lista_classes.delete(0, tk.END)
        for cid, info in sorted(self.class_manager.classes.items()):
            if info['is_background']:
                continue
            total = len(info['bboxes']) + len(info['polygons'])
            self.lista_classes.insert(tk.END, f"{cid}: {info['name']} ({info['color']}) [{total}]")

    def adicionar_classe(self):
        nome = simpledialog.askstring("Nova Classe", "Nome da classe:")
        if nome:
            cor = simpledialog.askstring("Cor", "Cor (ex: red, blue) ou vazio:")
            if cor and cor not in PALETA_CORES:
                self.log(f"⚠️ Cor '{cor}' inválida, usando automática.")
                cor = None
            cid = self.class_manager.add_class(nome, color=cor)
            self.atualizar_lista_classes()
            self.log(f"✅ Classe '{nome}' adicionada (ID {cid})")

    def remover_classe(self):
        selecao = self.lista_classes.curselection()
        if not selecao:
            self.log("⚠️ Selecione uma classe para remover.")
            return
        item = self.lista_classes.get(selecao[0])
        cid = int(item.split(':')[0])
        nome = self.class_manager.classes[cid]['name']
        self.class_manager.remove_class(cid)
        self.atualizar_lista_classes()
        self.log(f"🗑️ Classe '{nome}' removida.")

    def renomear_classe(self):
        selecao = self.lista_classes.curselection()
        if not selecao:
            self.log("⚠️ Selecione uma classe para renomear.")
            return
        item = self.lista_classes.get(selecao[0])
        cid = int(item.split(':')[0])
        novo_nome = simpledialog.askstring("Renomear", "Novo nome:")
        if novo_nome:
            self.class_manager.rename_class(cid, novo_nome)
            self.atualizar_lista_classes()
            self.log(f"✏️ Classe renomeada para '{novo_nome}'")

    def ativar_classe(self):
        selecao = self.lista_classes.curselection()
        if not selecao:
            self.log("⚠️ Selecione uma classe para ativar.")
            return
        item = self.lista_classes.get(selecao[0])
        cid = int(item.split(':')[0])
        if self.annotator is not None:
            self.annotator.set_class(cid)
        else:
            # Salva para quando o anotador iniciar
            self._classe_para_ativar = cid
        self.log(f"🔵 Classe {cid} ativada")

    # ===== AÇÕES DURANTE ANOTAÇÃO =====
    def acao_salvar(self):
        if self.annotator is not None:
            self.annotator.set_acao('salvar')
        else:
            self.log("⚠️ Nenhuma anotação em andamento.")

    def acao_pular(self):
        if self.annotator is not None:
            self.annotator.set_acao('pular')
        else:
            self.log("⚠️ Nenhuma anotação em andamento.")

    def acao_sair(self):
        if self.annotator is not None:
            self.annotator.set_acao('sair')
        else:
            self.log("⚠️ Nenhuma anotação em andamento.")

    def acao_desfazer(self):
        if self.annotator is not None:
            self.annotator.undo()
        else:
            self.log("⚠️ Nenhuma anotação em andamento.")

    # ===== MÉTODO PARA VERIFICAR IMAGENS JÁ ANOTADAS =====
    def _get_annotated_images(self, destino):
        """Retorna um set com os nomes base das imagens já anotadas"""
        ja_feitas = set()
        
        # Verifica JSON
        annot_path = os.path.join(destino, 'annotations')
        if os.path.exists(annot_path):
            for f in os.listdir(annot_path):
                if f.endswith('.json'):
                    ja_feitas.add(os.path.splitext(f)[0])
        
        # Verifica BBox TXT
        bbox_path = os.path.join(destino, 'labels', 'bbox')
        if os.path.exists(bbox_path):
            for f in os.listdir(bbox_path):
                if f.endswith('.txt'):
                    ja_feitas.add(os.path.splitext(f)[0])
        
        # Verifica Polygon TXT
        poly_path = os.path.join(destino, 'labels', 'polygon')
        if os.path.exists(poly_path):
            for f in os.listdir(poly_path):
                if f.endswith('.txt'):
                    ja_feitas.add(os.path.splitext(f)[0])
        
        return ja_feitas

    # ===== INICIAR ANOTAÇÃO =====
    def iniciar_anotacao(self):
        origem = self.pasta_origem.get()
        destino = self.pasta_destino.get()
        modo = self.modo_inicial.get()
        
        if not origem or not destino:
            messagebox.showerror("Erro", "Selecione as pastas de origem e destino.")
            return
        if not self.class_manager.classes:
            messagebox.showwarning("Aviso", "Adicione pelo menos uma classe.")
            return

        # Cria pastas necessárias
        os.makedirs(os.path.join(destino, 'images'), exist_ok=True)
        os.makedirs(os.path.join(destino, 'labels', 'bbox'), exist_ok=True)
        os.makedirs(os.path.join(destino, 'labels', 'polygon'), exist_ok=True)
        os.makedirs(os.path.join(destino, 'annotations'), exist_ok=True)

        # Verifica imagens já anotadas
        ja_feitas = self._get_annotated_images(destino)

        arquivos = sorted([
            f for f in os.listdir(origem)
            if f.lower().endswith(EXTENSOES_VALIDAS)
            and os.path.splitext(f)[0] not in ja_feitas
        ])

        if not arquivos:
            self.log("ℹ️ Todas as imagens já foram anotadas!")
            messagebox.showinfo("Info", "Todas as imagens já foram anotadas!")
            return

        total_salvas = 0
        total_pulei = 0
        self._classe_para_ativar = None

        for i, nome_arquivo in enumerate(arquivos, 1):
            caminho = os.path.join(origem, nome_arquivo)
            try:
                img = Image.open(caminho).convert('RGB')
                img_array = np.array(img)
            except Exception as e:
                self.log(f"❌ Erro ao abrir {nome_arquivo}: {e}")
                traceback.print_exc()
                continue

            self.log(f"📷 Anotando: {nome_arquivo} ({i}/{len(arquivos)})")

            # Limpa container
            for widget in self.fig_container.winfo_children():
                widget.destroy()

            # Define a shape no ClassManager
            self.class_manager.set_image_shape(img_array.shape)

            # Tenta carregar anotações existentes (JSON)
            if carregar_anotacoes_json(caminho, self.class_manager, destino, self.log):
                self.log(f"   📂 Anotações carregadas do JSON.")

            self.annotator = Annotator(
                self.fig_container,
                caminho,
                img_array,
                self.class_manager,
                self.log,
                self.action_var,
                on_update_callback=self.atualizar_lista_classes,
                on_mode_change_callback=self._on_mode_changed,
                mode=modo
            )

            # Ativa classe se houver alguma pendente
            if hasattr(self, '_classe_para_ativar') and self._classe_para_ativar is not None:
                self.annotator.set_class(self._classe_para_ativar)
                self._classe_para_ativar = None

            self.annotator.fig.canvas.draw_idle()

            # Aguarda a ação do usuário de forma robusta (wait_variable)
            self.action_var.set("")  # Reseta
            self.root.wait_variable(self.action_var)
            acao = self.action_var.get()

            # Verifica qual ação foi tomada
            if acao == 'salvar':
                # Mostra resumo antes de salvar
                counts = self.class_manager.get_annotation_count()
                total_objetos = sum(c['total'] for c in counts.values())
                self.log(f"   📊 Resumo das anotações: {total_objetos} objetos no total")
                for cid, info in counts.items():
                    nome = info['name']
                    self.log(f"      {nome}: {info['bboxes']} bboxes, {info['polygons']} polygons")
                
                # 1. Salva JSON com todas as anotações (removendo classes vazias)
                if salvar_anotacoes_json(caminho, self.class_manager, destino, self.log):
                    self.log(f"   ✅ JSON salvo em annotations/")
                else:
                    self.log(f"   ⚠️ Erro ao salvar JSON")
                
                # 2. Exporta para YOLO em AMBOS os formatos (bbox e polygon)
                n_bbox, n_poly = export_all_formats(caminho, self.class_manager, destino, self.log)
                
                total_salvas += 1
                self.log(f"💾 Salva concluída: {nome_arquivo} (BBox: {n_bbox}, Polygon: {n_poly})")
                
            elif acao == 'pular':
                total_pulei += 1
                self.log(f"⏭️ Pulada: {nome_arquivo}")
            elif acao == 'sair':
                self.log("🚪 Sessão encerrada pelo usuário")
                # Limpa o anotador antes de sair
                self.annotator = None
                break

            self.annotator = None
            self.atualizar_lista_classes()

        self.log(f"✅ Anotação finalizada. Salvas: {total_salvas}, Puladas: {total_pulei}")
        messagebox.showinfo("Concluído", f"Anotação finalizada.\nSalvas: {total_salvas}\nPuladas: {total_pulei}")
        self.status_var.set("Pronto")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationApp(root)
    root.mainloop()