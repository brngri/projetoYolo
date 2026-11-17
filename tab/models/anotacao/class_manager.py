# annotation_tab/models/class_manager.py
"""
Gerencia classes, anotações e máscaras com otimização via skimage.draw.polygon.
"""

import logging
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np
from skimage.draw import polygon

logger = logging.getLogger("AnnotationTab.ClassManager")

# Constantes importadas do módulo principal (serão definidas em config.py)
BACKGROUND_KEYWORDS = ('bg', 'background', 'fundo', 'esteira')


class ClassManager:
    """Gerencia classes, anotações e máscaras."""

    def __init__(self):
        self.classes: Dict[int, Dict] = {}
        self.next_id: int = 1
        self.available_ids: Set[int] = set()
        self._ny: Optional[int] = None
        self._nx: Optional[int] = None
        self._background_id: Optional[int] = None
        self._mask_cache: Dict[int, np.ndarray] = {}

    def set_image_shape(self, shape: Tuple[int, int, int]) -> None:
        self._ny, self._nx = shape[:2]
        self._mask_cache.clear()

    def add_class(self, name: str, color: Optional[str] = None, desc: str = "") -> int:
        if color is None:
            from matplotlib import colormaps
            import matplotlib.colors as mcolors
            cmap = colormaps['tab20']
            idx = (self.next_id - 1) % 20
            color = cmap(idx)
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
            'artists': []
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
        if polygon_verts[0] != polygon_verts[-1]:
            polygon_verts = polygon_verts + [polygon_verts[0]]
        if len(polygon_verts) < 4:
            return False
        self.classes[class_id]['polygons'].append(polygon_verts)
        self._recalc_mask(class_id)
        return True

    def _recalc_mask(self, class_id: int) -> None:
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
        if class_id not in self.classes:
            return False
        info = self.classes[class_id]

        if info['polygons']:
            removed = info['polygons'].pop()
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