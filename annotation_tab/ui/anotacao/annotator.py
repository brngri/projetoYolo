# annotation_tab/ui/annotator.py
"""
Classe Annotator - exibição e interação de anotação (bbox/polygon) usando matplotlib.
"""

import os
import logging
import numpy as np
import tkinter as tk
from typing import List, Tuple, Optional, Callable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle, Polygon
from matplotlib.widgets import RectangleSelector

from ...models.anotacao.class_manager import ClassManager

logger = logging.getLogger("AnnotationTab.Annotator")


class Annotator:
    """Classe responsável pela exibição e interação de anotação."""

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