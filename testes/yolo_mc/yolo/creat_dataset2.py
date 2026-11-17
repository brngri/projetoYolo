#!/usr/bin/env python3
"""
creat_dataset2.py (multi-grupo com sub-classes dos Nuts)

Monta dataset YOLO a partir das imagens e labels gerados pelo anotador
multi-grupo (arquivos com prefixo "X__").

ESTRUTURA DE CLASSES FINAIS:
  Grupo A (9):  AdHul, BlackSpot, ClassA, ClassB, ClassC, ClassD,
                Dirty, Insect, Shrivel
  Grupo B (4):  Nut_sub1, Nut_sub2, Nut_sub3, Nut_sub4
                (as 30 pastas Nut* viram 4 classes físicas)
  Grupo C (12): Moisture1 ... Moisture12
  TOTAL:        25 classes

SAÍDA:
  DATASET_ROOT/
  ├── data.yaml                 (25 classes, tudo junto)
  ├── images/train/
  ├── labels/train/
  └── por_grupo/
      ├── grupo_A/data.yaml, images/train/, labels/train/
      ├── grupo_B/data.yaml, images/train/, labels/train/
      └── grupo_C/data.yaml, images/train/, labels/train/

COMO USAR:
  python creat_dataset2.py
"""

import os
import re
import shutil
from pathlib import Path
from collections import Counter

# ====================================================
# CONFIGURAÇÕES
# ====================================================
IMAGES_DIR = Path(r"E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX171\dataset_yolo_anotador\images")
LABELS_DIR = Path(r"E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX171\dataset_yolo_anotador\labels")
DATASET_ROOT = Path(r"E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX171\dataset_final")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

GERAR_VIEWS_POR_GRUPO = True   # cria sub-datasets por grupo

# ---------------------------------------------------------------------
# SUB-CLASSES FÍSICAS DOS NUTS
# ---------------------------------------------------------------------
SUB_CLASSES_NUTS = {
    1: ["Nut10", "Nut13", "Nut23", "Nut24"],
    2: ["Nut1", "Nut2", "Nut3", "Nut4", "Nut5", "Nut6", "Nut7", "Nut8",
        "Nut9", "Nut11", "Nut14", "Nut16", "Nut17", "Nut19", "Nut20",
        "Nut22", "Nut25", "Nut26", "Nut27", "Nut28", "Nut29"],
    3: ["Nut15", "Nut18", "Nut21", "Nut30"],
    4: ["Nut12"],
}

# ---------------------------------------------------------------------
# ORDEM DAS CLASSES FÍSICAS NO ANOTADOR (deve espelhar CLASSES_NOMES
# do anotador_bbox_yolo.py — usada pra traduzir IDs antigos -> novos)
# ---------------------------------------------------------------------
CLASSES_FISICAS_ORDEM_ANOTADOR = (
    ["AdHul", "BlackSpot", "ClassA", "ClassB", "ClassC", "ClassD",
     "Dirty", "Insect", "Shrivel"]
    + [f"Nut{i}" for i in range(1, 31)]
    + [f"Moisture{i}" for i in range(1, 13)]
)

# ---------------------------------------------------------------------
# MAPEAMENTO: pasta física -> classe final
# ---------------------------------------------------------------------
MAPEAMENTO_CLASSE_FISICA = {
    # --- Grupo A: cada pasta = 1 classe própria ---
    "AdHul": "AdHul",
    "AdHull": "AdHul",
    "BlackSpot": "BlackSpot",
    "ClassA": "ClassA",
    "ClassB": "ClassB",
    "ClassC": "ClassC",
    "ClassD": "ClassD",
    "Dirty": "Dirty",
    "Insect": "Insect",
    "Shrivel": "Shrivel",

    # --- Grupo B: Nuts agrupados em 4 sub-classes ---
    **{nut: f"Nut_sub{s}" for s, nuts in SUB_CLASSES_NUTS.items() for nut in nuts},

    # --- Grupo C: cada Moisture = 1 classe própria ---
    # **{f"Moisture{i}": f"Moisture{i}" for i in range(1, 13)},
}

# ====================================================


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def extrair_grupo(nome_base: str):
    """'A__xxx' -> ('A', 'xxx'); 'xxx' -> (None, 'xxx')."""
    m = re.match(r"^([A-Z])__(.+)$", nome_base)
    if m:
        return m.group(1), m.group(2)
    return None, nome_base


# Aliases: nomes que aparecem no arquivo mas apontam pra outro nome canônico
ALIASES_CLASSE_FISICA = {
    "AdHull": "AdHul",     # arquivo tem 2 L's, pasta tinha 1
    # adicione aqui outros se aparecerem
}


def extrair_classe_fisica(nome_sem_grupo: str) -> str:
    """'AdHull_2026-...' -> 'AdHul'; 'Nut1_2026-...' -> 'Nut1'."""
    bruto = nome_sem_grupo.split("_")[0]
    return ALIASES_CLASSE_FISICA.get(bruto, bruto)


def extrair_nome_grupo_arquivo(nome_base: str):
    grupo, sem_grupo = extrair_grupo(nome_base)
    classe = extrair_classe_fisica(sem_grupo)
    return grupo, classe


# ---------------------------------------------------------------------
# Estrutura de pastas
# ---------------------------------------------------------------------
def ensure_dirs(root: Path):
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)


def ensure_dirs_grupo(root: Path, grupo: str):
    base = root / "por_grupo" / f"grupo_{grupo}"
    (base / "images" / "train").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "train").mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------
def write_data_yaml(root: Path, class_names: list):
    linhas = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/train",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for idx, nome in enumerate(class_names):
        linhas.append(f"  {idx}: {nome}")
    (root / "data.yaml").write_text("\n".join(linhas) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Remap de labels
# ---------------------------------------------------------------------
def remap_label_file(src_path: Path, dst_path: Path,
                     mapa_id_original_para_final: dict) -> str:
    """Reescreve o label trocando os IDs. Descarta linhas com ID não mapeado."""
    linhas_out = []
    ids_descartados = []
    for line in src_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            orig_id = int(parts[0])
        except ValueError:
            continue

        if orig_id in mapa_id_original_para_final:
            parts[0] = str(mapa_id_original_para_final[orig_id])
            linhas_out.append(" ".join(parts))
        else:
            ids_descartados.append(orig_id)

    text_out = "\n".join(linhas_out)
    dst_path.write_text(text_out, encoding="utf-8")

    if ids_descartados:
        print(f"  [aviso] {src_path.name}: IDs não mapeados: "
              f"{sorted(set(ids_descartados))}")
    return text_out


# ---------------------------------------------------------------------
# Construção do mapeamento de IDs (ordem do anotador -> ordem final)
# ---------------------------------------------------------------------
# Ordem explícita das classes finais (define os IDs YOLO).
# Ordem escolhida: grupo A -> grupo B -> grupo C
ORDEM_CLASSES_FINAIS = (
    # Grupo A
    ["AdHul", "BlackSpot", "ClassA", "ClassB", "ClassC", "ClassD",
     "Dirty", "Insect", "Shrivel"]
    # Grupo B (sub-classes em ordem numérica)
    + ["Nut_sub1", "Nut_sub2", "Nut_sub3", "Nut_sub4"]
    # Grupo C
    # + [f"Moisture{i}" for i in range(1, 13)]
)


def construir_mapeamento_id():
    """
    Retorna (classes_finais, mapa_orig_id -> final_id).

    classes_finais: lista na ORDEM EXPLÍCITA definida em ORDEM_CLASSES_FINAIS
    mapa: traduz o ID do anotador (baseado em CLASSES_FISICAS_ORDEM_ANOTADOR)
          para o ID final (baseado em ORDEM_CLASSES_FINAIS)
    """
    ordem_anotador = CLASSES_FISICAS_ORDEM_ANOTADOR
    classes_finais = list(ORDEM_CLASSES_FINAIS)

    classe_final_para_id = {c: i for i, c in enumerate(classes_finais)}

    # Sanity: todas as classes finais do mapeamento devem estar na ordem
    faltando = set(MAPEAMENTO_CLASSE_FISICA.values()) - set(classes_finais)
    if faltando:
        raise ValueError(
            f"Classes finais do MAPEAMENTO não estão em ORDEM_CLASSES_FINAIS: {faltando}"
        )

    mapa = {}
    for orig_id, classe_fisica in enumerate(ordem_anotador):
        final = MAPEAMENTO_CLASSE_FISICA.get(classe_fisica)
        if final is None:
            continue
        mapa[orig_id] = classe_final_para_id[final]

    return classes_finais, mapa

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("[INFO] Iniciando montagem do dataset multi-grupo...")
    print(f"  - Imagens: {IMAGES_DIR}")
    print(f"  - Rótulos: {LABELS_DIR}")
    print(f"  - Saída:   {DATASET_ROOT}")

    classes_finais, mapa_id = construir_mapeamento_id()
    print(f"  - Classes físicas: {len(CLASSES_FISICAS_ORDEM_ANOTADOR)}")
    print(f"  - Classes finais:  {len(classes_finais)}")
    print(f"    {classes_finais}")
    print(f"  - Gerar views por grupo: {GERAR_VIEWS_POR_GRUPO}")
    print("-" * 60)

    if not IMAGES_DIR.exists():
        print(f"[ERRO] Pasta de imagens não encontrada: {IMAGES_DIR}")
        return
    if not LABELS_DIR.exists():
        print(f"[ERRO] Pasta de labels não encontrada: {LABELS_DIR}")
        return

    # Estrutura principal
    ensure_dirs(DATASET_ROOT)
    img_out_main = DATASET_ROOT / "images" / "train"
    lbl_out_main = DATASET_ROOT / "labels" / "train"

    # Views por grupo
    bases_por_grupo = {}
    if GERAR_VIEWS_POR_GRUPO:
        for g in ("A", "B", "C"):
            bases_por_grupo[g] = ensure_dirs_grupo(DATASET_ROOT, g)

    # Lista imagens
    images = [p for p in IMAGES_DIR.rglob("*") if p.is_file() and is_image(p)]
    if not images:
        print("[ERRO] Nenhuma imagem encontrada em:", IMAGES_DIR)
        return

    # Contadores
    contagem_grupo = Counter()
    contagem_classe_fisica = Counter()
    contagem_classe_final = Counter()
    pos = 0
    neg = 0
    labels_ausentes = 0
    classes_desconhecidas = set()

    for img_path in sorted(images):
        nome_base = img_path.stem
        grupo, classe_fisica = extrair_nome_grupo_arquivo(nome_base)

        if grupo is None:
            grupo = "?"
        if classe_fisica not in MAPEAMENTO_CLASSE_FISICA:
            classes_desconhecidas.add(classe_fisica)

        contagem_grupo[grupo] += 1
        contagem_classe_fisica[classe_fisica] += 1

        # Copia imagem para o destino principal
        shutil.copy2(img_path, img_out_main / img_path.name)

        # Label correspondente
        lbl_src = LABELS_DIR / f"{nome_base}.txt"
        lbl_dst_main = lbl_out_main / f"{nome_base}.txt"

        if lbl_src.exists():
            txt = remap_label_file(lbl_src, lbl_dst_main, mapa_id)
            if txt:
                pos += 1
            else:
                neg += 1
        else:
            labels_ausentes += 1
            neg += 1
            lbl_dst_main.write_text("", encoding="utf-8")

        # Copia também para a view do grupo
        if GERAR_VIEWS_POR_GRUPO and grupo in bases_por_grupo:
            base = bases_por_grupo[grupo]
            shutil.copy2(img_path, base / "images" / "train" / img_path.name)
            if lbl_dst_main.exists():
                shutil.copy2(lbl_dst_main,
                             base / "labels" / "train" / f"{nome_base}.txt")
            else:
                (base / "labels" / "train" / f"{nome_base}.txt").write_text(
                    "", encoding="utf-8")

        # Conta objetos por classe final
        if lbl_dst_main.exists():
            try:
                for line in lbl_dst_main.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    cid = int(line.split()[0])
                    if 0 <= cid < len(classes_finais):
                        contagem_classe_final[classes_finais[cid]] += 1
            except Exception:
                pass

    # ---- data.yaml principal e por grupo ----
    write_data_yaml(DATASET_ROOT, classes_finais)
    print(f"\n[OK] data.yaml principal: {(DATASET_ROOT / 'data.yaml').as_posix()}")

    if GERAR_VIEWS_POR_GRUPO:
        for g, base in bases_por_grupo.items():
            write_data_yaml(base, classes_finais)
            print(f"[OK] data.yaml grupo {g}: {(base / 'data.yaml').as_posix()}")

    # ---- Resumo ----
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"  Imagens totais: {len(images)}")
    print(f"  Positivas (com box): {pos}")
    print(f"  Negativas (sem box): {neg}")
    print(f"  Labels ausentes: {labels_ausentes}")
    print(f"  Classes finais: {len(classes_finais)}")

    print(f"\n  Distribuição por grupo (imagens):")
    for g in sorted(contagem_grupo.keys()):
        print(f"    grupo {g}: {contagem_grupo[g]:>4d}")

    print(f"\n  Distribuição por classe final (nº de objetos anotados):")
    print(f"    {'ID':<4s} {'classe':<15s} {'objetos':>8s}")
    vazias = []
    for i, c in enumerate(classes_finais):
        n = contagem_classe_final.get(c, 0)
        print(f"    {i:<4d} {c:<15s} {n:>8d}")
        if n == 0:
            vazias.append(c)

    if vazias:
        print(f"\n  [ALERTA] {len(vazias)} classes sem nenhum objeto anotado:")
        for c in vazias:
            print(f"    - {c}")
        print(f"         Você precisa anotar essas classes antes de treinar,")
        print(f"         ou removê-las do mapeamento se não forem usadas.")

    if classes_desconhecidas:
        print(f"\n  [aviso] classes físicas fora do mapeamento: "
              f"{sorted(classes_desconhecidas)}")

    # ---- Verificação ----
    print("\n[VERIFICAÇÃO] Amostra de IDs nos labels de saída:")
    amostra = sorted(lbl_out_main.glob("*.txt"))[:3]
    for lbl_file in amostra:
        linhas = lbl_file.read_text(encoding="utf-8").splitlines()[:3]
        ids = [l.split()[0] for l in linhas if l.strip()]
        print(f"  {lbl_file.name}: {ids}")

    print(f"\n[OK] Dataset principal: {DATASET_ROOT}")
    if GERAR_VIEWS_POR_GRUPO:
        print(f"[OK] Views por grupo:   {DATASET_ROOT / 'por_grupo'}")


if __name__ == "__main__":
    main()


































# #!/usr/bin/env python3
# """
# creat_dataset2.py

# Monta dataset YOLO a partir de imagens e rótulos .txt existentes.
# Configurações fixas no início do script.

# Como usar:
#   python creat_dataset2.py
# """

# import os
# import shutil
# from pathlib import Path

# # ====================================================
# # CONFIGURAÇÕES (EDITE AQUI)
# # ====================================================

# # Pasta com as imagens de entrada (geradas pelo spectral_embedding ou pelo anotador)
# IMAGES_DIR = Path(r"E:\ProjetoDogChow\03_resultados\yolo\240726\dataset_yolo_anotador\images")

# # Pasta com os rótulos .txt gerados pelo anotador
# LABELS_DIR = Path(r"E:\ProjetoDogChow\03_resultados\yolo\240726\dataset_yolo_anotador\labels")

# # Pasta de saída do dataset final (será criada)
# DATASET_ROOT = Path(r"E:\ProjetoDogChow\03_resultados\yolo\240726\dataset_final")

# # ====== LÓGICA DE CORRESPONDÊNCIA ENTRE IMAGEM E LABEL ======
# # O anotador salva a imagem com o nome original (ex: "imagem.png") e o label com
# # o mesmo nome base + ".txt" (ex: "imagem.txt") na pasta labels.
# # Portanto, NÃO é necessário remover ou adicionar sufixos.
# # Deixe ambos vazios para usar o nome exato da imagem (sem extensão) como base.
# REMOVER_SUFIXO_IMAGEM = ""   # não remove nada
# SUFIXO_LABEL = ""             # não adiciona nada

# # Se por acaso seus labels tiverem um sufixo diferente (ex: "_mask.txt"),
# # ajuste SUFIXO_LABEL para "_mask". Exemplo:
# # SUFIXO_LABEL = "_mask"

# # ====== CONTROLE DE CLASSES ======
# # Se True, todas as classes serão unificadas para CLASS_ID_UNICO.
# # Se False, mantém os IDs originais dos labels.
# UNIFICAR_CLASSES = False

# # Se UNIFICAR_CLASSES = True, qual ID usar? (normalmente 0)
# CLASS_ID_UNICO = 0

# # Nomes das classes na ordem do ID (0, 1, 2, ...).
# # Deve corresponder aos IDs presentes nos labels.
# CLASS_NAMES = ["conforme", "nao_conforme"]   # ID 0 e 1

# # Extensões de imagem suportadas
# IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# # ====================================================

# def ensure_dirs(root: Path):
#     """Cria as pastas images/train e labels/train dentro de root"""
#     (root / "images/train").mkdir(parents=True, exist_ok=True)
#     (root / "labels/train").mkdir(parents=True, exist_ok=True)

# def write_data_yaml(root: Path, class_names: list):
#     """Gera o arquivo data.yaml"""
#     yml = (
#         f"path: {root.as_posix()}\n"
#         f"train: images/train\n"
#         f"val: images/train  # opcional: use val se tiver split\n"
#         f"nc: {len(class_names)}\n"
#         f"names: {class_names}\n"
#     )
#     (root / "data.yaml").write_text(yml, encoding="utf-8")

# def is_image(p: Path) -> bool:
#     return p.suffix.lower() in IMG_EXTS

# def remap_label(src_path: Path, dst_path: Path, unify: bool, class_id: int = 0) -> str:
#     """
#     Lê um label YOLO e reescreve.
#     Se unify=True, todas as classes viram class_id.
#     Caso contrário, mantém a classe original.
#     Retorna o texto final (vazio se sem linhas).
#     """
#     lines_out = []
#     for line in src_path.read_text(encoding="utf-8").splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         parts = line.split()
#         if unify:
#             parts[0] = str(class_id)
#         # else: mantém o ID original
#         lines_out.append(" ".join(parts))

#     text_out = "\n".join(lines_out)
#     dst_path.write_text(text_out, encoding="utf-8")
#     return text_out

# def main():
#     print("[INFO] Iniciando montagem do dataset...")
#     print(f"  - Imagens: {IMAGES_DIR}")
#     print(f"  - Rótulos: {LABELS_DIR}")
#     print(f"  - Saída:   {DATASET_ROOT}")
#     print(f"  - Remover sufixo da imagem: {REMOVER_SUFIXO_IMAGEM if REMOVER_SUFIXO_IMAGEM else 'nenhum'}")
#     print(f"  - Sufixo do label: {SUFIXO_LABEL if SUFIXO_LABEL else 'nenhum'}")
#     print(f"  - Unificar classes: {UNIFICAR_CLASSES}")
#     print(f"  - Número de classes: {len(CLASS_NAMES)} -> {CLASS_NAMES}")
#     print("-" * 60)

#     if not IMAGES_DIR.exists():
#         print(f"[ERRO] Pasta de imagens não encontrada: {IMAGES_DIR}")
#         return
#     if not LABELS_DIR.exists():
#         print(f"[AVISO] Pasta de labels não encontrada: {LABELS_DIR}. Todas as imagens serão negativas.")

#     # Cria estrutura
#     ensure_dirs(DATASET_ROOT)
#     img_out = DATASET_ROOT / "images/train"
#     lbl_out = DATASET_ROOT / "labels/train"

#     # Lista imagens
#     images = [p for p in IMAGES_DIR.rglob("*") if p.is_file() and is_image(p)]
#     if not images:
#         print("[ERRO] Nenhuma imagem encontrada em:", IMAGES_DIR)
#         return

#     pos = 0       # imagens com pelo menos uma caixa
#     neg = 0       # imagens sem caixa (label vazio ou ausente)
#     labels_copiados = 0
#     labels_ausentes = 0

#     for img_path in images:
#         nome_imagem = img_path.stem   # nome base da imagem (sem extensão)

#         # Nome base para procurar o rótulo: remove sufixo da imagem se configurado
#         nome_base = nome_imagem
#         if REMOVER_SUFIXO_IMAGEM and nome_base.endswith(REMOVER_SUFIXO_IMAGEM):
#             nome_base = nome_base[:-len(REMOVER_SUFIXO_IMAGEM)]

#         # Caminho do rótulo fonte: LABELS_DIR / (nome_base + SUFIXO_LABEL + ".txt")
#         lbl_src = LABELS_DIR / f"{nome_base}{SUFIXO_LABEL}.txt"

#         # Caminho do rótulo destino: mesmo nome da imagem (sem sufixos)
#         lbl_dst = lbl_out / f"{nome_imagem}.txt"

#         # Copia imagem (mantém o nome original)
#         shutil.copy2(img_path, img_out / img_path.name)

#         # Processa rótulo
#         if lbl_src.exists():
#             text_out = remap_label(lbl_src, lbl_dst, UNIFICAR_CLASSES, CLASS_ID_UNICO)
#             labels_copiados += 1
#             if text_out:
#                 pos += 1
#             else:
#                 neg += 1
#         else:
#             labels_ausentes += 1
#             neg += 1
#             lbl_dst.write_text("", encoding="utf-8")  # label vazio (negativo)

#     # Gera data.yaml
#     write_data_yaml(DATASET_ROOT, CLASS_NAMES)

#     print("\n[OK] Dataset montado com sucesso!")
#     print(f"  - Imagens totais: {len(images)}")
#     print(f"  - Positivas (com box): {pos}")
#     print(f"  - Negativas (sem box): {neg}")
#     print(f"  - Labels copiados: {labels_copiados}")
#     print(f"  - Labels ausentes: {labels_ausentes}")
#     print(f"  - data.yaml: {(DATASET_ROOT / 'data.yaml').as_posix()}")
#     print(f"  - Estrutura criada em: {DATASET_ROOT}")

#     # Verificação dos IDs das classes
#     if not UNIFICAR_CLASSES and pos > 0:
#         # Vamos verificar alguns labels para ver se os IDs foram preservados
#         amostra = list(lbl_out.glob("*.txt"))[:3]
#         if amostra:
#             print("\n[VERIFICAÇÃO] IDs nos labels de saída (primeiras 3 linhas de cada):")
#             for lbl_file in amostra:
#                 with open(lbl_file, 'r') as f:
#                     linhas = f.readlines()[:3]
#                 if linhas:
#                     ids = [linha.split()[0] for linha in linhas if linha.strip()]
#                     print(f"  {lbl_file.name}: {ids}")
#                 else:
#                     print(f"  {lbl_file.name}: vazio")

# if __name__ == "__main__":
#     main()

































































# # montar_dataset.py
# # - Copia as imagens de IMAGES_DIR e os .txt de LABELS_DIR
# # - REMAPEIA todas as classes do CVAT (0-5) para uma unica classe (0 = "defeito")
# # - Cria estrutura Ultralytics: dataset_root/{images/train,labels/train}
# # - Gera data.yaml apontando para esse dataset
# # - Cria .txt vazio para imagens negativas (sem defeito)

# import os, shutil
# from pathlib import Path

# # ====== CONFIG ======
# IMAGES_DIR = Path(r"E:\ProjetoDogChow\03_resultados\yolo\pseudo_rgb")
# LABELS_DIR = Path(r"E:\ProjetoDogChow\03_resultados\yolo\dataset_yolo_anotador\labels")
# DATASET_ROOT = Path(r"E:\ProjetoDogChow\03_resultados\yolo\dataset_according\dataset_according")
# CLASS_NAMES = ["nao_conforme"]  # 1 classe unica -- todas as 6 classes do CVAT viram esta
# IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# # =====================

# def ensure_dirs():
#     (DATASET_ROOT / "images/train").mkdir(parents=True, exist_ok=True)
#     (DATASET_ROOT / "labels/train").mkdir(parents=True, exist_ok=True)

# def write_data_yaml():
#     yml = (
#         f"path: {DATASET_ROOT.as_posix()}\n"
#         f"train: images/train\n"
#         f"val: images/train  # provisorio: usar Train tambem como Val\n"
#         f"names: {CLASS_NAMES}\n"
#         f"nc: {len(CLASS_NAMES)}\n"  # sem comentar -- garante que o Ultralytics leia nc explicitamente
#     )
#     (DATASET_ROOT / "data.yaml").write_text(yml, encoding="utf-8")

# def is_image(p: Path) -> bool:
#     return p.suffix.lower() in IMG_EXTS

# def remap_label_to_single_class(src_path: Path, dst_path: Path) -> str:
#     """
#     Le um label YOLO (class x_center y_center w h, uma linha por box) e
#     reescreve TODAS as linhas com class_id = 0, independente da classe
#     original do CVAT. Retorna o texto final (vazio = negativo).
#     """
#     lines_out = []
#     for line in src_path.read_text(encoding="utf-8").splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         parts = line.split()
#         # parts[0] e o class_id original (0-5) -- trocamos por 0
#         parts[0] = "0"
#         lines_out.append(" ".join(parts))

#     text_out = "\n".join(lines_out)
#     dst_path.write_text(text_out, encoding="utf-8")
#     return text_out

# def main():
#     ensure_dirs()
#     write_data_yaml()

#     img_out = DATASET_ROOT / "images/train"
#     lbl_out = DATASET_ROOT / "labels/train"

#     images = [p for p in IMAGES_DIR.rglob("*") if p.is_file() and is_image(p)]
#     if not images:
#         print("[ERRO] Nenhuma imagem encontrada em:", IMAGES_DIR)
#         return

#     pos, neg = 0, 0
#     for ip in images:
#         stem = ip.stem
#         lp = LABELS_DIR / f"{stem}.txt"

#         # copia imagem
#         shutil.copy2(ip, img_out / ip.name)

#         # copia/remapeia label
#         dst_lbl = lbl_out / f"{stem}.txt"
#         if lp.exists():
#             text_out = remap_label_to_single_class(lp, dst_lbl)
#             pos += 1 if text_out else 0
#             neg += 1 if not text_out else 0
#         else:
#             # sem txt -> negativo
#             neg += 1
#             dst_lbl.write_text("", encoding="utf-8")

#     print(f"[OK] Dataset montado em: {DATASET_ROOT}")
#     print(f"- Imagens: {len(images)}")
#     print(f"- Positivas (com box 'defeito'): {pos}")
#     print(f"- Negativas (sem box): {neg}")
#     print(f"- data.yaml: {(DATASET_ROOT/'data.yaml').as_posix()}")

# if __name__ == "__main__":
#     main()
