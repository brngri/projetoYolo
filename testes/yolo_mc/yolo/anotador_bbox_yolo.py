"""
ANOTADOR DE BBOX PARA YOLO — MULTI-GRUPO com CLASSE AUTOMÁTICA
================================================================

Mudanças em relação à versão anterior:
  - Sem botões de classe (conforme/nao_conforme removidos)
  - Classe YOLO inferida automaticamente da pasta física
    (ex: grupo_A/pseudo_rgb/AdHul/*.png -> classe "AdHul")
  - 51 classes = 51 pastas físicas (AdHul, BlackSpot, ..., Nut1-30, Moisture1-12)
  - O ID YOLO é atribuído pela ordem em CLASSES_NOMES

ATALHOS:
  z              -> desfaz último retângulo
  n ou Enter     -> salva e vai pra próxima
  p              -> pula imagem
  q              -> sai

USO:
  python anotador_bbox_yolo.py                     # todos os grupos
  python anotador_bbox_yolo.py --grupos A B        # só A e B
  python anotador_bbox_yolo.py --classes Nut1 Nut2 # só essas classes físicas
  python anotador_bbox_yolo.py --split             # gera train/val + data.yaml
"""

import os
import shutil
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.patches import Rectangle
from PIL import Image


# ==================== CONFIGURAÇÕES ====================
PASTA_RAIZ = r"E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX171"
PASTA_DESTINO = os.path.join(PASTA_RAIZ, "dataset_yolo_anotador")

GRUPOS_A_ANOTAR = ["A", "B", "C"]
PREFIXAR_NOME_COM_GRUPO = True

EXTENSOES_VALIDAS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# =====================================================================
# CLASSES DO PROJETO — a ordem aqui define o ID YOLO (0, 1, 2, ...)
# =====================================================================
# Esta lista deve cobrir TODAS as pastas físicas que existem em
# grupo_X/pseudo_rgb/. Se uma pasta não estiver aqui, o script avisa.
CLASSES_NOMES = [
    # --- Grupo A (bandeja) ---
    "AdHul",
    "BlackSpot",
    "ClassA",
    "ClassB",
    "ClassC",
    "ClassD",
    "Dirty",
    "Insect",
    "Shrivel",
    # --- Grupo B (Nut individual) ---
    *[f"Nut{i}" for i in range(1, 31)],      # Nut1..Nut30
    # --- Grupo C (Moisture) ---
    *[f"Moisture{i}" for i in range(1, 13)], # Moisture1..Moisture12
]
# Total esperado: 9 + 30 + 12 = 51 classes

# Mapeamento nome -> ID YOLO
CLASSE_PARA_ID = {nome: idx for idx, nome in enumerate(CLASSES_NOMES)}

TRAIN_SPLIT = 0.8


# ==================== CORES TERMINAL ====================
class Colors:
    HEADER = '\033[95m'
    BLUE   = '\033[94m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    BOLD   = '\033[1m'
    END    = '\033[0m'


def print_header(t):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}🔷 {t}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")


def print_success(t):
    print(f"{Colors.GREEN}✅ {t}{Colors.END}")


def print_info(t):
    print(f"{Colors.BLUE}ℹ️  {t}{Colors.END}")


def print_alerta(t):
    print(f"{Colors.RED}{Colors.BOLD}🔴 {t}{Colors.END}")


# ==================== ANOTADOR ====================
class BBoxAnnotator:
    """
    Anotador com classe automática.
    Todas as caixas desenhadas na imagem recebem a MESMA classe,
    que vem do nome da pasta física.
    """

    def __init__(self, fig, ax_image, ax_instructions, ax_status,
                 image_path, img_array,
                 grupo=None, classe_fisica=None, classe_id=None):
        self.fig = fig
        self.ax_image = ax_image
        self.ax_instructions = ax_instructions
        self.ax_status = ax_status
        self.image_path = image_path
        self.img_array = img_array
        self.h, self.w = img_array.shape[0], img_array.shape[1]

        self.grupo = grupo
        self.classe_fisica = classe_fisica
        self.classe_id = classe_id   # ID YOLO (int)

        # Cor por grupo (visual só — não é a "classe")
        self.cor = {
            "A": "red",
            "B": "blue",
            "C": "green",
        }.get(grupo, "magenta")

        self.boxes = []     # lista de (classe_id, x1, y1, x2, y2)
        self.patches = []
        self.texts = []
        self.acao = None

        self.ax_image.set_position([0.03, 0.15, 0.62, 0.80])
        self.ax_image.imshow(self.img_array)
        self.ax_image.axis('on')
        self.atualizar_titulo()

        self.setup_instructions_panel()
        self.setup_status_panel()
        self.setup_buttons()

        self.rs = RectangleSelector(
            self.ax_image, self.onselect, useblit=True, button=[1],
            minspanx=5, minspany=5, spancoords='pixels', interactive=False,
            props=dict(facecolor=self.cor, edgecolor=self.cor,
                       alpha=0.25, fill=True)
        )
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    # ------------------------------------------------------------------
    def atualizar_titulo(self):
        nome_arquivo = os.path.basename(self.image_path)
        self.ax_image.set_title(
            f"{nome_arquivo}\n"
            f"Grupo: {self.grupo}  |  Classe: {self.classe_fisica} "
            f"(ID YOLO={self.classe_id})  |  Caixas: {len(self.boxes)}",
            fontsize=11, color=self.cor
        )

    def setup_instructions_panel(self):
        self.ax_instructions.axis('off')
        self.ax_instructions.set_position([0.68, 0.55, 0.29, 0.40])

        txt = (
            'INSTRUÇÕES:\n\n'
            f'Classe desta imagem:\n'
            f'   {self.classe_fisica} (ID={self.classe_id})\n\n'
            '1. Clique e arraste para desenhar\n'
            '   a caixa em cada amêndoa\n\n'
            '2. Z -> desfaz última caixa\n\n'
            '3. N ou ENTER -> salva e vai\n'
            '   pra próxima imagem\n\n'
            '4. P -> pula imagem\n\n'
            '5. Q -> sair (progresso salvo)\n\n'
            'A classe é fixa nesta imagem —\n'
            'vem da pasta física.'
        )

        self.ax_instructions.text(
            0.02, 0.98, txt,
            transform=self.ax_instructions.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
        )

    def setup_status_panel(self):
        self.ax_status.axis('off')
        self.ax_status.set_position([0.68, 0.15, 0.29, 0.35])

        self.status_text = self.ax_status.text(
            0.02, 0.98, self.get_status_text(),
            transform=self.ax_status.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        )

    def get_status_text(self):
        return (
            f"CAIXAS NESTA IMAGEM:\n\n"
            f"[{len(self.boxes):3d}] {self.classe_fisica}\n\n"
            f"Cor do desenho:\n"
            f"  grupo {self.grupo} -> {self.cor}"
        )

    def setup_buttons(self):
        # Sem botões de classe — só ações
        self.ax_desfazer = plt.axes([0.68, 0.08, 0.13, 0.05])
        self.ax_salvar   = plt.axes([0.82, 0.08, 0.15, 0.05])
        self.ax_pular    = plt.axes([0.68, 0.02, 0.13, 0.05])
        self.ax_sair     = plt.axes([0.82, 0.02, 0.15, 0.05])

        self.btn_desfazer = Button(self.ax_desfazer, 'DESFAZER (Z)')
        self.btn_salvar   = Button(self.ax_salvar, 'SALVAR E PRÓXIMA (N)')
        self.btn_pular    = Button(self.ax_pular, 'PULAR (P)')
        self.btn_sair     = Button(self.ax_sair, 'SAIR (Q)')

        self.btn_desfazer.color = 'lightpink'
        self.btn_salvar.color = 'palegreen'
        self.btn_pular.color = 'khaki'
        self.btn_sair.color = 'lightcoral'

        self.btn_desfazer.on_clicked(self.desfazer)
        self.btn_salvar.on_clicked(lambda e: self.finalizar('salvar'))
        self.btn_pular.on_clicked(lambda e: self.finalizar('pular'))
        self.btn_sair.on_clicked(lambda e: self.finalizar('sair'))

    # ------------------------------------------------------------------
    def onselect(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 is None or x2 is None:
            return

        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        x1 = max(0, min(x1, self.w)); x2 = max(0, min(x2, self.w))
        y1 = max(0, min(y1, self.h)); y2 = max(0, min(y2, self.h))

        if (x2 - x1) < 3 or (y2 - y1) < 3:
            return

        # A classe é sempre a da imagem
        self.boxes.append((self.classe_id, x1, y1, x2, y2))

        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                          linewidth=2, edgecolor=self.cor, facecolor='none')
        self.ax_image.add_patch(rect)
        self.patches.append(rect)

        texto = self.ax_image.text(
            x1, max(y1 - 4, 0), self.classe_fisica,
            color=self.cor, fontsize=8, fontweight='bold'
        )
        self.texts.append(texto)
        self.atualizar_status()

    def desfazer(self, event=None):
        if not self.boxes:
            return
        self.boxes.pop()
        self.patches.pop().remove()
        self.texts.pop().remove()
        self.atualizar_status()

    def atualizar_status(self):
        self.status_text.set_text(self.get_status_text())
        self.atualizar_titulo()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'z':
            self.desfazer()
        elif event.key in ('n', 'enter'):
            self.finalizar('salvar')
        elif event.key == 'p':
            self.finalizar('pular')
        elif event.key == 'q':
            self.finalizar('sair')

    def finalizar(self, acao):
        self.acao = acao
        plt.close(self.fig)


# ==================== EXPORT YOLO ====================
def salvar_anotacao_yolo(image_path, boxes, largura_img, altura_img,
                         pasta_destino, nome_saida=None):
    """Copia a imagem e escreve o .txt YOLO (class_id cx cy w h normalizados)."""
    pasta_imgs = os.path.join(pasta_destino, 'images')
    pasta_labels = os.path.join(pasta_destino, 'labels')
    os.makedirs(pasta_imgs, exist_ok=True)
    os.makedirs(pasta_labels, exist_ok=True)

    if nome_saida is None:
        nome_saida = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]

    shutil.copy2(image_path, os.path.join(pasta_imgs, nome_saida + ext))

    linhas = []
    for class_id, x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2 / largura_img
        cy = (y1 + y2) / 2 / altura_img
        w = (x2 - x1) / largura_img
        h = (y2 - y1) / altura_img
        linhas.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(os.path.join(pasta_labels, nome_saida + '.txt'), 'w') as f:
        f.write('\n'.join(linhas))

    return len(linhas)


# ==================== LISTAGEM ====================
def listar_tarefas(grupos, filtro_classes=None):
    """Retorna [(grupo, classe_fisica, caminho)]."""
    tarefas = []
    pastas_desconhecidas = set()

    for grupo in grupos:
        pasta_grupo = os.path.join(PASTA_RAIZ, f"grupo_{grupo}", "pseudo_rgb")
        if not os.path.isdir(pasta_grupo):
            print_alerta(f"Pasta não encontrada (grupo {grupo}): {pasta_grupo}")
            continue

        for classe_fisica in sorted(os.listdir(pasta_grupo)):
            pasta_classe = os.path.join(pasta_grupo, classe_fisica)
            if not os.path.isdir(pasta_classe):
                continue
            if filtro_classes is not None and classe_fisica not in filtro_classes:
                continue
            if classe_fisica not in CLASSE_PARA_ID:
                pastas_desconhecidas.add(classe_fisica)
                continue

            for f in sorted(os.listdir(pasta_classe)):
                if f.lower().endswith(EXTENSOES_VALIDAS):
                    tarefas.append((grupo, classe_fisica,
                                    os.path.join(pasta_classe, f)))

    if pastas_desconhecidas:
        print_alerta(f"Pastas fora de CLASSES_NOMES (ignoradas): {sorted(pastas_desconhecidas)}")
        print_alerta("Adicione os nomes em CLASSES_NOMES no topo do script.")

    return tarefas


def imagens_ja_anotadas(pasta_destino):
    pasta_labels = os.path.join(pasta_destino, 'labels')
    if not os.path.exists(pasta_labels):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(pasta_labels)
            if f.endswith('.txt')}


def nome_saida_para(grupo, caminho_original):
    base = os.path.splitext(os.path.basename(caminho_original))[0]
    return f"{grupo}__{base}" if PREFIXAR_NOME_COM_GRUPO else base


# ==================== LOOP PRINCIPAL ====================
def anotar_pasta(grupos, filtro_classes=None):
    if not os.path.isdir(PASTA_RAIZ):
        print_alerta(f"PASTA_RAIZ não existe: {PASTA_RAIZ}")
        return

    os.makedirs(PASTA_DESTINO, exist_ok=True)

    tarefas = listar_tarefas(grupos, filtro_classes)
    if not tarefas:
        print_alerta("Nenhuma imagem encontrada para os grupos/classes selecionados.")
        return

    ja_feitas = imagens_ja_anotadas(PASTA_DESTINO)

    fila = []
    for grupo, classe_fisica, caminho in tarefas:
        nome_saida = nome_saida_para(grupo, caminho)
        if nome_saida in ja_feitas:
            continue
        fila.append((grupo, classe_fisica, caminho, nome_saida))

    print_header("ANOTADOR BBOX YOLO — CLASSE AUTOMÁTICA")
    print_info(f"Grupos: {grupos}")
    print_info(f"Total de classes configuradas: {len(CLASSES_NOMES)}")
    print_info(f"Total de imagens encontradas: {len(tarefas)}")
    print_info(f"Já anotadas: {len(tarefas) - len(fila)}")
    print_info(f"Restantes: {len(fila)}")

    dist = {}
    for grupo, classe, *_ in fila:
        chave = f"grupo {grupo} / {classe}"
        dist[chave] = dist.get(chave, 0) + 1
    if dist:
        print_info("Distribuição:")
        for k in sorted(dist.keys()):
            print(f"     {k:<35s} {dist[k]:>4d}")

    if not fila:
        print_success("Nada para anotar — tudo já foi feito!")
        return

    total_salvas = 0
    total_pulei = 0
    resumo = {}

    for i, (grupo, classe_fisica, caminho, nome_saida) in enumerate(fila, 1):
        try:
            img = Image.open(caminho).convert('RGB')
            img_array = np.array(img)
        except Exception as e:
            print_alerta(f"Erro ao abrir {caminho}: {e}")
            continue

        classe_id = CLASSE_PARA_ID[classe_fisica]

        print(f"\n[{i}/{len(fila)}] grupo={grupo}  classe={classe_fisica}  "
              f"ID={classe_id}")
        print(f"   arquivo: {os.path.basename(caminho)}")

        fig = plt.figure(figsize=(20, 12), facecolor='lightgray')
        ax_image = fig.add_axes([0.03, 0.15, 0.62, 0.80])
        ax_instructions = fig.add_axes([0.68, 0.55, 0.29, 0.40])
        ax_status = fig.add_axes([0.68, 0.15, 0.29, 0.35])

        annotator = BBoxAnnotator(
            fig, ax_image, ax_instructions, ax_status,
            caminho, img_array,
            grupo=grupo,
            classe_fisica=classe_fisica,
            classe_id=classe_id,
        )

        plt.show()

        chave = (grupo, classe_fisica)
        resumo.setdefault(chave, [0, 0])

        if annotator.acao == 'salvar':
            n = salvar_anotacao_yolo(
                caminho, annotator.boxes,
                annotator.w, annotator.h,
                PASTA_DESTINO, nome_saida=nome_saida
            )
            print_success(f"Salvo {nome_saida}: {n} caixa(s)  [classe ID={classe_id}]")
            total_salvas += 1
            resumo[chave][0] += 1

        elif annotator.acao == 'pular':
            print_info("Pulada")
            total_pulei += 1
            resumo[chave][1] += 1

        elif annotator.acao == 'sair' or annotator.acao is None:
            print_info("Sessão encerrada")
            break

    print_header("RESUMO")
    print_success(f"Salvas: {total_salvas}")
    print_info(f"Puladas: {total_pulei}")

    if resumo:
        print("\nProgresso por grupo / classe:")
        print(f"  {'grupo':<6s} {'classe':<15s} {'salvas':>7s} {'puladas':>8s}")
        for (g, c) in sorted(resumo.keys()):
            s, p = resumo[(g, c)]
            print(f"  {g:<6s} {c:<15s} {s:>7d} {p:>8d}")

    print_info("Rode com --split quando quiser gerar train/val + data.yaml")


# ==================== SPLIT ====================
def gerar_split(train_ratio=TRAIN_SPLIT, seed=42):
    pasta_imgs = os.path.join(PASTA_DESTINO, 'images')
    pasta_labels = os.path.join(PASTA_DESTINO, 'labels')

    if not os.path.exists(pasta_imgs):
        print_alerta("Nada anotado ainda.")
        return

    arquivos = sorted([f for f in os.listdir(pasta_imgs)
                       if f.lower().endswith(EXTENSOES_VALIDAS)])
    if not arquivos:
        print_alerta("Pasta images/ vazia.")
        return

    random.seed(seed)
    random.shuffle(arquivos)

    n_train = int(len(arquivos) * train_ratio)
    split_train = arquivos[:n_train]
    split_val = arquivos[n_train:]

    pasta_split = os.path.join(PASTA_DESTINO, 'dataset_split')

    for subset, lista in [('train', split_train), ('val', split_val)]:
        dir_img = os.path.join(pasta_split, 'images', subset)
        dir_lbl = os.path.join(pasta_split, 'labels', subset)
        os.makedirs(dir_img, exist_ok=True)
        os.makedirs(dir_lbl, exist_ok=True)

        for nome_img in lista:
            nome_base = os.path.splitext(nome_img)[0]
            shutil.copy2(os.path.join(pasta_imgs, nome_img),
                         os.path.join(dir_img, nome_img))
            txt_src = os.path.join(pasta_labels, nome_base + '.txt')
            if os.path.exists(txt_src):
                shutil.copy2(txt_src, os.path.join(dir_lbl, nome_base + '.txt'))
            else:
                open(os.path.join(dir_lbl, nome_base + '.txt'), 'w').close()

    # data.yaml com as 51 classes na ordem certa
    yaml_content = (
        f"path: {pasta_split}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(CLASSES_NOMES)}\n"
        f"names:\n"
    )
    for idx, nome in enumerate(CLASSES_NOMES):
        yaml_content += f"  {idx}: {nome}\n"

    with open(os.path.join(pasta_split, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

    print_header("SPLIT GERADO")
    print_success(f"Train: {len(split_train)} imagens")
    print_success(f"Val: {len(split_val)} imagens")
    print_success(f"data.yaml com {len(CLASSES_NOMES)} classes")
    print_info(f"Arquivo: {os.path.join(pasta_split, 'data.yaml')}")


# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--train_ratio', type=float, default=TRAIN_SPLIT)
    parser.add_argument('--grupos', nargs='*', default=None)
    parser.add_argument('--classes', nargs='*', default=None)
    args = parser.parse_args()

    if args.split:
        gerar_split(train_ratio=args.train_ratio)
    else:
        grupos = args.grupos if args.grupos else GRUPOS_A_ANOTAR
        anotar_pasta(grupos=grupos, filtro_classes=args.classes)



































# """
# ANOTADOR DE BBOX PARA YOLO — AMÊNDOAS (COM/SEM DEFEITO)
# =========================================================
# Ferramenta local de anotação manual, no estilo da amostragem_manual_kitkat.py,
# mas para imagens RGB comuns (jpg/png) e exportando direto no formato
# Ultralytics YOLO (bbox retangular, múltiplos objetos por imagem).

# FLUXO:
#   1. Abre cada imagem da PASTA_ORIGEM
#   2. Você desenha retângulos (clique-arraste) em cada amêndoa
#   3. Cada retângulo já é adicionado com a classe atual selecionada
#   4. Botão "SALVAR E PRÓXIMA" grava o .txt (formato YOLO) + copia a imagem
#   5. Ao final, rode o script com --split para gerar o dataset final
#      (images/train, images/val, labels/train, labels/val, data.yaml)

# ATALHOS:
#   1 / 2         -> troca classe atual
#   z              -> desfaz último retângulo
#   n ou Enter     -> salva anotações da imagem atual e vai pra próxima
#   p              -> pula imagem (não salva nada, não copia)
#   q              -> sai (progresso já salvo fica salvo)
# """

# import os
# import sys
# import shutil
# import random
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import RectangleSelector, Button
# from matplotlib.patches import Rectangle
# from PIL import Image


# # ==================== CONFIGURAÇÕES ====================
# PASTA_ORIGEM = r'E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX171\grupo_A\pseudo_rgb'          # pasta com as imagens originais (jpg/png)
# PASTA_DESTINO = r'E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX171\grupo_A\dataset_yolo_anotador'         # onde vai gerar images/ e labels/

# EXTENSOES_VALIDAS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# # Classes -> id YOLO é (chave - 1), ou seja sem_defeito=0, com_defeito=1
# CLASSES = {
#     1: {'nome': 'conforme', 'cor': 'blue'},
#     2: {'nome': 'nao_conforme', 'cor': 'red'},
# }

# TRAIN_SPLIT = 0.8  # usado só na etapa --split


# # ==================== CORES TERMINAL ====================
# class Colors:
#     HEADER = '\033[95m'
#     BLUE   = '\033[94m'
#     GREEN  = '\033[92m'
#     YELLOW = '\033[93m'
#     RED    = '\033[91m'
#     BOLD   = '\033[1m'
#     END    = '\033[0m'


# def print_header(t):
#     print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
#     print(f"{Colors.BOLD}{Colors.BLUE}🔷 {t}{Colors.END}")
#     print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")


# def print_success(t):
#     print(f"{Colors.GREEN}✅ {t}{Colors.END}")


# def print_info(t):
#     print(f"{Colors.BLUE}ℹ️  {t}{Colors.END}")


# def print_alerta(t):
#     print(f"{Colors.RED}{Colors.BOLD}🔴 {t}{Colors.END}")


# # ==================== ANOTADOR ====================
# class BBoxAnnotator:
#     def __init__(self, fig, ax_image, ax_instructions, ax_status, image_path, img_array):
#         self.fig = fig
#         self.ax_image = ax_image
#         self.ax_instructions = ax_instructions
#         self.ax_status = ax_status
#         self.image_path = image_path
#         self.img_array = img_array
#         self.h, self.w = img_array.shape[0], img_array.shape[1]

#         self.current_classe = 1
#         self.boxes = []          # lista de (classe_id_dict_key, x1, y1, x2, y2) em pixels
#         self.patches = []        # patches desenhados (pra poder remover no undo)
#         self.texts = []

#         self.acao = None         # 'salvar' | 'pular' | 'sair'

#         self.ax_image.set_position([0.03, 0.15, 0.62, 0.80])
#         self.ax_image.imshow(self.img_array)
#         self.ax_image.axis('on')
#         self.atualizar_titulo()

#         self.setup_instructions_panel()
#         self.setup_status_panel()
#         self.setup_buttons()

#         self.rs = RectangleSelector(
#             self.ax_image,
#             self.onselect,
#             useblit=True,
#             button=[1],
#             minspanx=5,
#             minspany=5,
#             spancoords='pixels',
#             interactive=False,
#             props=dict(facecolor=CLASSES[self.current_classe]['cor'], edgecolor=CLASSES[self.current_classe]['cor'], alpha=0.25, fill=True)
#         )

#         self.fig.canvas.mpl_connect('key_press_event', self.on_key)

#     # ------------------------------------------------------------------
#     def atualizar_titulo(self):
#         c = CLASSES[self.current_classe]
#         nome_arquivo = os.path.basename(self.image_path)
#         self.ax_image.set_title(
#             f"{nome_arquivo}\nClasse atual: {c['nome']}  |  Caixas: {len(self.boxes)}",
#             fontsize=11,
#             color=c['cor']
#         )

#     def setup_instructions_panel(self):
#         self.ax_instructions.axis('off')
#         self.ax_instructions.set_position([0.68, 0.55, 0.29, 0.40])

#         txt = (
#             'INSTRUÇÕES:\n\n'
#             '1. Selecione a classe (botão ou tecla 1/2)\n\n'
#             '2. Clique e arraste para desenhar\n'
#             '   a caixa em cada amêndoa\n'
#             '   (adiciona automaticamente)\n\n'
#             '3. Z -> desfaz última caixa\n\n'
#             '4. N ou ENTER -> salva e vai\n'
#             '   pra próxima imagem\n\n'
#             '5. P -> pula imagem\n'
#             '   (não salva nada)\n\n'
#             '6. Q -> sair (progresso fica salvo)\n'
#         )

#         self.ax_instructions.text(
#             0.02, 0.98, txt,
#             transform=self.ax_instructions.transAxes,
#             verticalalignment='top',
#             fontsize=9,
#             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
#         )

#     def setup_status_panel(self):
#         self.ax_status.axis('off')
#         self.ax_status.set_position([0.68, 0.15, 0.29, 0.35])

#         self.status_text = self.ax_status.text(
#             0.02, 0.98, self.get_status_text(),
#             transform=self.ax_status.transAxes,
#             verticalalignment='top',
#             fontsize=9,
#             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
#         )

#     def get_status_text(self):
#         contagem = {v['nome']: 0 for v in CLASSES.values()}
#         for classe, *_ in self.boxes:
#             contagem[CLASSES[classe]['nome']] += 1

#         status = "CAIXAS NESTA IMAGEM:\n\n"
#         for nome, qtd in contagem.items():
#             status += f"[{qtd:3d}] {nome}\n"

#         return status

#     def setup_buttons(self):
#         n = len(CLASSES)
#         largura = 0.60 / n

#         self.botoes_classe = []
#         for i, (k, v) in enumerate(CLASSES.items()):
#             ax = plt.axes([0.03 + i * (largura + 0.01), 0.03, largura, 0.05])
#             btn = Button(ax, f"{v['nome'].upper()} ({k})")
#             btn.color = v['cor']
#             btn.on_clicked(lambda e, kk=k: self.set_classe(kk))
#             self.botoes_classe.append(btn)

#         self.ax_desfazer = plt.axes([0.68, 0.08, 0.13, 0.05])
#         self.ax_salvar   = plt.axes([0.82, 0.08, 0.15, 0.05])
#         self.ax_pular    = plt.axes([0.68, 0.02, 0.13, 0.05])
#         self.ax_sair     = plt.axes([0.82, 0.02, 0.15, 0.05])

#         self.btn_desfazer = Button(self.ax_desfazer, 'DESFAZER (Z)')
#         self.btn_salvar   = Button(self.ax_salvar, 'SALVAR E PRÓXIMA (N)')
#         self.btn_pular    = Button(self.ax_pular, 'PULAR (P)')
#         self.btn_sair     = Button(self.ax_sair, 'SAIR (Q)')

#         self.btn_desfazer.color = 'lightpink'
#         self.btn_salvar.color = 'palegreen'
#         self.btn_pular.color = 'khaki'
#         self.btn_sair.color = 'lightcoral'

#         self.btn_desfazer.on_clicked(self.desfazer)
#         self.btn_salvar.on_clicked(lambda e: self.finalizar('salvar'))
#         self.btn_pular.on_clicked(lambda e: self.finalizar('pular'))
#         self.btn_sair.on_clicked(lambda e: self.finalizar('sair'))

#     # ------------------------------------------------------------------
#     def set_classe(self, classe):
#         self.current_classe = classe
#         cor = CLASSES[classe]['cor']
#         self.rs.set_props(facecolor=cor, edgecolor=cor)
#         self.atualizar_titulo()
#         self.fig.canvas.draw_idle()

#     def onselect(self, eclick, erelease):
#         x1, y1 = eclick.xdata, eclick.ydata
#         x2, y2 = erelease.xdata, erelease.ydata

#         if x1 is None or x2 is None:
#             return

#         x1, x2 = sorted([x1, x2])
#         y1, y2 = sorted([y1, y2])

#         x1 = max(0, min(x1, self.w))
#         x2 = max(0, min(x2, self.w))
#         y1 = max(0, min(y1, self.h))
#         y2 = max(0, min(y2, self.h))

#         if (x2 - x1) < 3 or (y2 - y1) < 3:
#             return

#         classe = self.current_classe
#         self.boxes.append((classe, x1, y1, x2, y2))

#         cor = CLASSES[classe]['cor']
#         rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=cor, facecolor='none')
#         self.ax_image.add_patch(rect)
#         self.patches.append(rect)

#         texto = self.ax_image.text(x1, max(y1 - 4, 0), CLASSES[classe]['nome'], color=cor, fontsize=8, fontweight='bold')
#         self.texts.append(texto)

#         self.atualizar_status()

#     def desfazer(self, event=None):
#         if not self.boxes:
#             return

#         self.boxes.pop()

#         p = self.patches.pop()
#         p.remove()

#         t = self.texts.pop()
#         t.remove()

#         self.atualizar_status()

#     def atualizar_status(self):
#         self.status_text.set_text(self.get_status_text())
#         self.atualizar_titulo()
#         self.fig.canvas.draw_idle()

#     def on_key(self, event):
#         if event.key in [str(k) for k in CLASSES.keys()]:
#             self.set_classe(int(event.key))
#         elif event.key == 'z':
#             self.desfazer()
#         elif event.key in ('n', 'enter'):
#             self.finalizar('salvar')
#         elif event.key == 'p':
#             self.finalizar('pular')
#         elif event.key == 'q':
#             self.finalizar('sair')

#     def finalizar(self, acao):
#         self.acao = acao
#         plt.close(self.fig)


# # ==================== EXPORT YOLO ====================
# def salvar_anotacao_yolo(image_path, boxes, largura_img, altura_img, pasta_destino):
#     """Copia a imagem e escreve o .txt YOLO (classe cx cy w h, tudo normalizado 0-1)."""
#     pasta_imgs = os.path.join(pasta_destino, 'images')
#     pasta_labels = os.path.join(pasta_destino, 'labels')
#     os.makedirs(pasta_imgs, exist_ok=True)
#     os.makedirs(pasta_labels, exist_ok=True)

#     nome_base = os.path.splitext(os.path.basename(image_path))[0]
#     ext = os.path.splitext(image_path)[1]

#     destino_img = os.path.join(pasta_imgs, nome_base + ext)
#     destino_txt = os.path.join(pasta_labels, nome_base + '.txt')

#     shutil.copy2(image_path, destino_img)

#     linhas = []
#     for classe, x1, y1, x2, y2 in boxes:
#         class_id = classe - 1  # YOLO é 0-indexed

#         cx = (x1 + x2) / 2 / largura_img
#         cy = (y1 + y2) / 2 / altura_img
#         w = (x2 - x1) / largura_img
#         h = (y2 - y1) / altura_img

#         linhas.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

#     with open(destino_txt, 'w') as f:
#         f.write('\n'.join(linhas))  # arquivo vazio = imagem negativa (sem objeto), válido em YOLO

#     return len(linhas)


# # ==================== LOOP PRINCIPAL DE ANOTAÇÃO ====================
# def imagens_ja_anotadas(pasta_destino):
#     pasta_labels = os.path.join(pasta_destino, 'labels')
#     if not os.path.exists(pasta_labels):
#         return set()
#     return {os.path.splitext(f)[0] for f in os.listdir(pasta_labels) if f.endswith('.txt')}


# def anotar_pasta():
#     if not os.path.exists(PASTA_ORIGEM):
#         print_alerta(f"Pasta não encontrada: {PASTA_ORIGEM}")
#         return

#     os.makedirs(PASTA_DESTINO, exist_ok=True)

#     ja_feitas = imagens_ja_anotadas(PASTA_DESTINO)

#     arquivos = sorted([
#         f for f in os.listdir(PASTA_ORIGEM)
#         if f.lower().endswith(EXTENSOES_VALIDAS)
#         and os.path.splitext(f)[0] not in ja_feitas
#     ])

#     print_header("ANOTADOR BBOX YOLO — AMÊNDOAS")
#     for k, v in CLASSES.items():
#         print(f"  {k}. {v['nome']}")
#     print_info(f"Imagens restantes: {len(arquivos)} (já anotadas: {len(ja_feitas)})")

#     total_salvas = 0
#     total_pulei = 0

#     for i, nome_arquivo in enumerate(arquivos, 1):
#         caminho = os.path.join(PASTA_ORIGEM, nome_arquivo)

#         try:
#             img = Image.open(caminho).convert('RGB')
#             img_array = np.array(img)
#         except Exception as e:
#             print_alerta(f"Erro ao abrir {nome_arquivo}: {e}")
#             continue

#         print(f"\n[{i}/{len(arquivos)}] {nome_arquivo}")

#         fig = plt.figure(figsize=(20, 12), facecolor='lightgray')
#         ax_image = fig.add_axes([0.03, 0.15, 0.62, 0.80])
#         ax_instructions = fig.add_axes([0.68, 0.55, 0.29, 0.40])
#         ax_status = fig.add_axes([0.68, 0.15, 0.29, 0.35])

#         annotator = BBoxAnnotator(fig, ax_image, ax_instructions, ax_status, caminho, img_array)

#         plt.show()

#         if annotator.acao == 'salvar':
#             n_boxes = salvar_anotacao_yolo(caminho, annotator.boxes, annotator.w, annotator.h, PASTA_DESTINO)
#             print_success(f"Salvo: {n_boxes} caixa(s)")
#             total_salvas += 1

#         elif annotator.acao == 'pular':
#             print_info("Pulada (nada salvo)")
#             total_pulei += 1

#         elif annotator.acao == 'sair' or annotator.acao is None:
#             print_info("Sessão encerrada pelo usuário")
#             break

#     print_header("RESUMO DA SESSÃO")
#     print_success(f"Imagens salvas: {total_salvas}")
#     print_info(f"Imagens puladas: {total_pulei}")
#     print_info(f"Rode com --split quando quiser gerar o dataset final (train/val)")


# # ==================== SPLIT FINAL PARA TREINO ====================
# def gerar_split(train_ratio=TRAIN_SPLIT, seed=42):
#     pasta_imgs = os.path.join(PASTA_DESTINO, 'images')
#     pasta_labels = os.path.join(PASTA_DESTINO, 'labels')

#     if not os.path.exists(pasta_imgs):
#         print_alerta("Nenhuma imagem anotada ainda. Rode o script sem --split primeiro.")
#         return

#     arquivos = sorted([f for f in os.listdir(pasta_imgs) if f.lower().endswith(EXTENSOES_VALIDAS)])

#     if not arquivos:
#         print_alerta("Pasta images/ está vazia.")
#         return

#     random.seed(seed)
#     random.shuffle(arquivos)

#     n_train = int(len(arquivos) * train_ratio)
#     split_train = arquivos[:n_train]
#     split_val = arquivos[n_train:]

#     pasta_split = os.path.join(PASTA_DESTINO, 'dataset_split')

#     for subset, lista in [('train', split_train), ('val', split_val)]:
#         dir_img = os.path.join(pasta_split, 'images', subset)
#         dir_lbl = os.path.join(pasta_split, 'labels', subset)
#         os.makedirs(dir_img, exist_ok=True)
#         os.makedirs(dir_lbl, exist_ok=True)

#         for nome_img in lista:
#             nome_base = os.path.splitext(nome_img)[0]
#             shutil.copy2(os.path.join(pasta_imgs, nome_img), os.path.join(dir_img, nome_img))

#             caminho_txt = os.path.join(pasta_labels, nome_base + '.txt')
#             if os.path.exists(caminho_txt):
#                 shutil.copy2(caminho_txt, os.path.join(dir_lbl, nome_base + '.txt'))
#             else:
#                 open(os.path.join(dir_lbl, nome_base + '.txt'), 'w').close()

#     nomes_classes = [CLASSES[k]['nome'] for k in sorted(CLASSES.keys())]

#     yaml_content = (
#         f"path: {pasta_split}\n"
#         f"train: images/train\n"
#         f"val: images/val\n\n"
#         f"names:\n"
#     )
#     for idx, nome in enumerate(nomes_classes):
#         yaml_content += f"  {idx}: {nome}\n"

#     with open(os.path.join(pasta_split, 'data.yaml'), 'w') as f:
#         f.write(yaml_content)

#     print_header("SPLIT GERADO")
#     print_success(f"Train: {len(split_train)} imagens")
#     print_success(f"Val: {len(split_val)} imagens")
#     print_success(f"data.yaml criado em: {os.path.join(pasta_split, 'data.yaml')}")
#     print_info("Pronto pra usar com: yolo detect train data=data.yaml model=yolov8n.pt")


# # ==================== MAIN ====================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--split', action='store_true', help='Gera o split train/val final em vez de anotar')
#     parser.add_argument('--train_ratio', type=float, default=TRAIN_SPLIT)
#     args = parser.parse_args()

#     if args.split:
#         gerar_split(train_ratio=args.train_ratio)
#     else:
#         anotar_pasta()
