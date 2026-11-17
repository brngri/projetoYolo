# treinar_yolo_onnx_kfold.py
# Requisitos:
#   pip install ultralytics matplotlib pandas onnxruntime numpy pyyaml
#
# Treina YOLOv8 no dataset multi-grupo (A + B, sem grupo C).
#
# Dataset esperado: dataset_final/data.yaml com 13 classes:
#   Grupo A: AdHul, BlackSpot, ClassA, ClassB, ClassC, ClassD, Dirty, Insect, Shrivel
#   Grupo B: Nut_sub1, Nut_sub2, Nut_sub3, Nut_sub4
#
# Modos disponíveis (configurável em MODO_TREINO):
#   - "tudo_junto":  treina nos 2 grupos juntos (13 classes)
#   - "por_grupo":   treina um modelo por grupo separadamente (recomendado pra testes)
#   - "ambos":       faz os dois (2 treinos + 1 combinado)

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO


# ======================================================================
# CONFIGURAÇÕES PRINCIPAIS
# ======================================================================
# Raiz do projeto (onde ficam os datasets gerados pelo creat_dataset2.py)
PROJETO_ROOT = Path(r"E:\ProjetoCrainwalnut\preprocessed\14102026_embedding\FX171\dataset_final")

# ---- Modo de treino ----
#   "tudo_junto": treina 1 modelo com todos os grupos (13 classes)
#   "por_grupo":  treina 1 modelo por grupo (só A, só B) — recomendado se quiser comparar
#   "ambos":      faz os dois
MODO_TREINO = "por_grupo"

# ---- Datasets específicos ----
YAML_PATH_TUDO_JUNTO = PROJETO_ROOT / "data.yaml"
YAMLS_POR_GRUPO = {
    "A": PROJETO_ROOT / "por_grupo" / "grupo_A" / "data.yaml",
    "B": PROJETO_ROOT / "por_grupo" / "grupo_B" / "data.yaml",
    # Grupo C removido — não existe mais
}

MODEL_W   = "yolov8n.pt"
DEVICE    = 0
WORKERS   = 0
EPOCHS    = 350
PATIENCE  = 100
RUN_NAME  = "yolov8n_autobackoff"
VAL_SPLIT = "train"      # usado se USE_KFOLD=False

# Backoff de (imgsz, batch) até achar o maior que cabe na VRAM
TRY_SETTINGS = [(1024, 4), (1024, 2), (960, 4), (960, 2), (896, 4), (896, 2)]

# Export ONNX
OPSET     = 11
DYNAMIC   = False
SIMPLIFY  = True
WITH_NMS  = False

# K-Fold
USE_KFOLD   = False
K_FOLDS     = 5
KFOLD_SEED  = 42

# Gráficos
SHOW_PLOTS = False

# Teste (só roda se o data.yaml tiver split 'test'; senão é ignorado)
TESTAR_MODELO = False
TEST_SPLIT    = "test"
# ======================================================================


# ----------------------------------------------------------------------
# Helpers de validação do dataset
# ----------------------------------------------------------------------
def validar_dataset(yaml_path: Path):
    """Lê o data.yaml e imprime um resumo (classes, nc, pasta de imagens)."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml não encontrado: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    nc = cfg.get('nc', None)
    names = cfg.get('names', None)
    if isinstance(names, list):
        nomes = names
    elif isinstance(names, dict):
        nomes = [names[i] for i in sorted(names.keys())]
    else:
        nomes = []

    print(f"[validação] data.yaml: {yaml_path}")
    print(f"[validação] nc={nc}")
    print(f"[validação] classes ({len(nomes)}): {nomes}")

    # Conta imagens
    base = yaml_path.parent
    train_rel = cfg.get('train', 'images/train')
    train_dir = base / train_rel
    if train_dir.exists():
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        n_img = sum(1 for p in train_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in exts)
        print(f"[validação] imagens em train: {n_img}")
    else:
        print(f"[validação] [AVISO] pasta train não existe: {train_dir}")

    # Checa se nc bate com names
    if nc is not None and len(nomes) != nc:
        print(f"[validação] [AVISO] nc={nc} mas {len(nomes)} nomes — "
              f"o YOLO pode reclamar.")
    return cfg


# ----------------------------------------------------------------------
# Plot de métricas
# ----------------------------------------------------------------------
def show_png(png_path: Path, title="Matriz de confusão"):
    if not png_path.exists():
        print("[AVISO] Não encontrei:", png_path)
        return
    img = mpimg.imread(str(png_path))
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_metrics_from_csv(csv_path: Path, save_dir: Path, title_prefix: str = ""):
    if not csv_path.exists():
        print(f"[AVISO] {csv_path} não encontrado. Pulando gráficos.")
        return

    df = pd.read_csv(csv_path)
    loss_cols = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
    metric_cols = ['metrics/precision(B)', 'metrics/recall(B)',
                   'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    epochs = df['epoch'].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix} Evolução das métricas" if title_prefix
                 else "Evolução das métricas", fontsize=14)

    ax = axes[0, 0]
    for col in loss_cols:
        if col in df.columns:
            ax.plot(epochs, df[col].values, label=col.replace('train/', ''))
    ax.set_xlabel('Época'); ax.set_ylabel('Loss')
    ax.set_title('Losses (treino)'); ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    for col in metric_cols:
        if col in df.columns:
            ax.plot(epochs, df[col].values,
                    label=col.replace('metrics/', '').replace('(B)', ''))
    ax.set_xlabel('Época'); ax.set_ylabel('Valor')
    ax.set_title('Métricas (validação)'); ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    for col, cor in [('train/box_loss', 'red'),
                     ('train/cls_loss', 'green'),
                     ('train/dfl_loss', 'blue')]:
        if col in df.columns:
            ax.plot(epochs, df[col].values,
                    label=col.replace('train/', ''), color=cor)
    ax.set_xlabel('Época'); ax.set_ylabel('Loss')
    ax.set_title('Losses detalhadas'); ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    if 'metrics/mAP50-95(B)' in df.columns:
        ax.plot(epochs, df['metrics/mAP50-95(B)'].values,
                label='mAP50-95', color='purple', linewidth=2)
    if 'metrics/mAP50(B)' in df.columns:
        ax.plot(epochs, df['metrics/mAP50(B)'].values,
                label='mAP50', color='orange', linestyle='--')
    ax.set_xlabel('Época'); ax.set_ylabel('mAP')
    ax.set_title('mAP'); ax.legend(); ax.grid(True)

    plt.tight_layout()
    out_path = save_dir / "metrics.png"
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Gráficos salvos em: {out_path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ----------------------------------------------------------------------
# Treino com backoff
# ----------------------------------------------------------------------
def train_with_backoff(model: YOLO, yaml_path: str,
                       project_dir: Optional[Path] = None,
                       run_name: Optional[str] = None) -> Tuple[any, int, int]:
    last_err = None
    for imgsz, batch in TRY_SETTINGS:
        print(f"\n[INFO] Tentando treino com imgsz={imgsz}, batch={batch}, "
              f"amp=False, workers={WORKERS} ...")
        try:
            res = model.train(
                data=yaml_path,
                imgsz=imgsz,
                epochs=EPOCHS,
                batch=batch,
                device=DEVICE,
                workers=WORKERS,
                project=(project_dir or Path(yaml_path).parent / "runs").as_posix(),
                name=(run_name or RUN_NAME) + f"_sz{imgsz}_b{batch}",
                cos_lr=True,
                patience=PATIENCE,
                amp=False,
                verbose=True,
                single_cls=False,
                lr0=0.01,
                momentum=0.937,
                optimizer='SGD',
                freeze=5,
                weight_decay=0.005,
            )
            print("[OK] Treino finalizado com essa configuração.")
            return res, imgsz, batch
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            torch.cuda.empty_cache()
            print("[OOM] Ficou sem VRAM. Tentando próxima combinação...")
        except Exception as e:
            last_err = e
            print(f"[ERRO] Tentativa com imgsz={imgsz}, batch={batch} falhou: {e}")
    raise RuntimeError(f"Todas as tentativas falharam. Último erro: {last_err}")


# ----------------------------------------------------------------------
# Exports ONNX
# ----------------------------------------------------------------------
def export_ultralytics_4plusnc(model: YOLO, imgsz: int, out_dir: Path) -> Path:
    print("\n[INFO] Exportando ONNX 4+nc (padrão Ultralytics, sem NMS)...")
    onnx_path = model.export(
        format="onnx", imgsz=imgsz, opset=OPSET,
        dynamic=DYNAMIC, simplify=SIMPLIFY, half=False, nms=WITH_NMS
    )
    onnx_path = Path(onnx_path)
    dst = out_dir / "best_ultra_4plusnc.onnx"
    if dst.exists():
        dst.unlink()
    shutil.move(str(onnx_path), str(dst))
    print("[OK] ONNX 4+nc:", dst)
    return dst


class YoloV8_HSI_Wrapper(nn.Module):
    """Emite (B, 5+nc, N): [cx, cy, w, h, obj, cls0..clsN]."""
    def __init__(self, det_model: nn.Module, nc: int):
        super().__init__()
        self.det = det_model
        self.nc = nc
        self.det.export = True

    def forward(self, x):
        y = self.det(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if not isinstance(y, torch.Tensor):
            raise RuntimeError(f"Saída inesperada: {type(y)}")
        if y.dim() != 3:
            raise RuntimeError(f"Saída inesperada do head: shape={tuple(y.shape)}")

        B, A, C = y.shape
        if C == 4 + self.nc:
            y = y.permute(0, 2, 1).contiguous()
        elif A == 4 + self.nc:
            pass
        else:
            raise RuntimeError(f"Formato inesperado: {(B, A, C)} com nc={self.nc}")

        xywh = y[:, 0:4, :]
        cls_logits = y[:, 4:4 + self.nc, :]
        obj = torch.sigmoid(cls_logits).amax(dim=1, keepdim=True)
        return torch.cat([xywh, obj, cls_logits], dim=1)


def export_hsi_5plusnc(best_pt: Path, imgsz: int, out_dir: Path) -> Path:
    print("\n[INFO] Exportando ONNX 5+nc (compatível HSI, sem NMS)...")
    y = YOLO(best_pt.as_posix())
    names = y.names if isinstance(y.names, dict) else {i: n for i, n in enumerate(y.names)}
    nc = len(names)
    head = y.model.model[-1]
    print(f"[CHECK] nc={nc}, model.nc={getattr(y.model,'nc',None)}, "
          f"head.nc={getattr(head,'nc',None)}")
    assert nc == getattr(y.model, "nc", None) == getattr(head, "nc", None), "nc divergente."

    wrapper = YoloV8_HSI_Wrapper(y.model, nc).eval()
    dummy = torch.randn(1, 3, imgsz, imgsz)

    out_path = out_dir / "best_hsi_5plusnc.onnx"
    if out_path.exists():
        out_path.unlink()

    torch.onnx.export(
        wrapper, dummy, out_path.as_posix(),
        opset_version=OPSET,
        input_names=["images"], output_names=["output"],
        dynamic_axes=None if not DYNAMIC else {
            "images": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "anchors"},
        },
        do_constant_folding=True,
    )
    print("[OK] ONNX 5+nc:", out_path)
    return out_path


def optional_check_runtime(onnx_path: Path, exp_channels: int, imgsz: int):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path.as_posix(), providers=['CPUExecutionProvider'])
        x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
        y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        print(f"[CHECK] {onnx_path.name} runtime shape: {y.shape}")
        if y.shape[1] != exp_channels:
            print(f"[ALERTA] Esperado 2º eixo = {exp_channels}, obtido = {y.shape[1]}")
    except Exception as e:
        print(f"[AVISO] Checagem runtime pulada: {e}")


# ----------------------------------------------------------------------
# Treino de UM dataset (reutilizável pra tudo_junto ou por_grupo)
# ----------------------------------------------------------------------
def treinar_e_exportar(yaml_path: Path, nome_run: str):
    """Treina, avalia, testa (opcional) e exporta ONNX. Retorna dict de info."""
    print("\n" + "#" * 70)
    print(f"# TREINANDO: {nome_run}")
    print(f"# data.yaml: {yaml_path}")
    print("#" * 70)

    # Validação do dataset
    validar_dataset(yaml_path)

    project_dir = yaml_path.parent / "runs"

    # ---- Treino ----
    model = YOLO(MODEL_W)
    train_res, used_imgsz, used_batch = train_with_backoff(
        model, str(yaml_path),
        project_dir=project_dir,
        run_name=nome_run,
    )
    train_dir = Path(train_res.save_dir)
    best_pt = train_dir / "weights" / "best.pt"

    if best_pt.exists():
        print("[INFO] Carregando pesos 'best':", best_pt)
        model = YOLO(best_pt.as_posix())
        print("[DEBUG] best names:", model.names, "nc:", len(model.names))

    # ---- Avaliação ----
    print(f"[INFO] Avaliando (split='{VAL_SPLIT}')...")
    val_res = model.val(
        data=str(yaml_path), imgsz=used_imgsz, device=DEVICE, workers=WORKERS,
        split=VAL_SPLIT, project=project_dir.as_posix(),
        name=train_dir.name + "_val",
        save_json=True, save_conf=True, plots=True,
    )
    val_dir_final = Path(val_res.save_dir)

    csv_path = train_dir / "results.csv"
    if csv_path.exists():
        plot_metrics_from_csv(csv_path, train_dir, title_prefix=nome_run)

    cm_val = val_dir_final / "confusion_matrix.png"
    cm_train = train_dir / "confusion_matrix.png"
    if cm_val.exists():
        show_png(cm_val, title=f"Matriz de confusão ({nome_run})")
    elif cm_train.exists():
        show_png(cm_train, title=f"Matriz de confusão (treino, {nome_run})")

    # ---- Teste (opcional) ----
    test_dir_final = None
    if TESTAR_MODELO:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            test_cfg = yaml.safe_load(f)
        if 'test' not in test_cfg:
            print("[AVISO] data.yaml não tem split 'test'. Teste ignorado.")
        else:
            res_test_dir = yaml_path.parent / "resultados_teste"
            res_test_dir.mkdir(parents=True, exist_ok=True)
            test_res = model.val(
                data=str(yaml_path), split=TEST_SPLIT,
                imgsz=used_imgsz, device=DEVICE, workers=WORKERS,
                project=res_test_dir.as_posix(), name=f"{nome_run}_teste",
                save_json=True, save_conf=True, plots=True,
            )
            test_dir_final = Path(test_res.save_dir)
            print(f"\n[TESTE {nome_run}] mAP50-95={test_res.box.map:.4f}, "
                  f"mAP50={test_res.box.map50:.4f}, "
                  f"P={test_res.box.mp:.4f}, R={test_res.box.mr:.4f}")

    # ---- Exports ONNX ----
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artef_dir = yaml_path.parent / "artefatos_treino" / f"{train_dir.name}_{stamp}"
    artef_dir.mkdir(parents=True, exist_ok=True)
    print("[OK] Pasta de artefatos:", artef_dir)

    onnx_ultra = export_ultralytics_4plusnc(model, used_imgsz, artef_dir)
    optional_check_runtime(onnx_ultra, exp_channels=4 + len(model.names), imgsz=used_imgsz)

    onnx_hsi = export_hsi_5plusnc(best_pt, used_imgsz, artef_dir)
    optional_check_runtime(onnx_hsi, exp_channels=5 + len(model.names), imgsz=used_imgsz)

    # ---- Cópia de artefatos ----
    def copy_if_exists(src: Path, rename: Optional[str] = None):
        if src and src.exists():
            dst = artef_dir / (rename if rename else src.name)
            shutil.copy2(src, dst)
            print(f"[OK] Copiado: {src.name} -> {dst.name}")

    copy_if_exists(best_pt)
    copy_if_exists(train_dir / "results.csv")
    copy_if_exists(cm_train, rename="confusion_matrix_train.png")
    copy_if_exists(cm_val,   rename="confusion_matrix_val.png")
    copy_if_exists(val_dir_final / "predictions.json")
    if test_dir_final:
        copy_if_exists(test_dir_final / "confusion_matrix.png",
                       rename="confusion_matrix_test.png")

    return {
        "nome_run": nome_run,
        "yaml": yaml_path,
        "train_dir": train_dir,
        "val_dir": val_dir_final,
        "test_dir": test_dir_final,
        "best_pt": best_pt,
        "artef_dir": artef_dir,
        "imgsz": used_imgsz,
        "batch": used_batch,
        "onnx_ultra": onnx_ultra,
        "onnx_hsi": onnx_hsi,
    }


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f"MODO_TREINO = {MODO_TREINO}")
    print("=" * 70)

    if MODO_TREINO == "tudo_junto":
        resultado = treinar_e_exportar(YAML_PATH_TUDO_JUNTO, "tudo_junto")
        print("\n[RESUMO]")
        print(f"  run: {resultado['nome_run']}")
        print(f"  treino: {resultado['train_dir']}")
        print(f"  ONNX 4+nc: {resultado['onnx_ultra']}")
        print(f"  ONNX 5+nc: {resultado['onnx_hsi']}")

    elif MODO_TREINO == "por_grupo":
        resultados = {}
        for grupo, yaml_path in YAMLS_POR_GRUPO.items():
            if not yaml_path.exists():
                print(f"[AVISO] grupo {grupo}: yaml não existe ({yaml_path}), pulando.")
                continue
            resultados[grupo] = treinar_e_exportar(yaml_path, f"grupo_{grupo}")

        print("\n[RESUMO]")
        for g, r in resultados.items():
            print(f"  grupo {g}: {r['train_dir']}")

    elif MODO_TREINO == "ambos":
        resultados = {}

        print("\n>>> RODANDO 'por_grupo' PRIMEIRO <<<")
        for grupo, yaml_path in YAMLS_POR_GRUPO.items():
            if not yaml_path.exists():
                print(f"[AVISO] grupo {grupo}: yaml não existe, pulando.")
                continue
            resultados[f"grupo_{grupo}"] = treinar_e_exportar(yaml_path, f"grupo_{grupo}")

        print("\n>>> RODANDO 'tudo_junto' AGORA <<<")
        resultados["tudo_junto"] = treinar_e_exportar(YAML_PATH_TUDO_JUNTO, "tudo_junto")

        print("\n[RESUMO GERAL]")
        for nome, r in resultados.items():
            print(f"  {nome}: {r['train_dir']}")

    else:
        raise ValueError(f"MODO_TREINO inválido: {MODO_TREINO}")


if __name__ == "__main__":
    main()
















































# # treinar_yolo_onnx_kfold.py
# # Requisitos:
# #   pip install ultralytics matplotlib pandas onnxruntime numpy pyyaml
# #
# # O script faz:
# # 1) Define PYTORCH_CUDA_ALLOC_CONF para reduzir fragmentação de VRAM
# # 2) Opcionalmente executa K‑Fold cross‑validation (USE_KFOLD=True)
# # 3) Tenta treinar com backoff de (imgsz, batch) para caber em VRAM
# # 4) Desativa AMP e usa workers=0 (Windows)
# # 5) Avalia (split configurável) e gera matriz de confusão
# # 6) Exporta dois ONNX: 4+nc (padrão) e 5+nc (compatível HSI)
# # 7) Gera gráficos de loss e métricas a partir do results.csv
# # 8) Opcionalmente testa o modelo em um conjunto de teste separado (TESTAR_MODELO=True)

# import os
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# import shutil
# from pathlib import Path
# from datetime import datetime
# from typing import Optional, Tuple, List, Dict
# import random
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import torch
# import torch.nn as nn
# import yaml
# from ultralytics import YOLO

# # ====== CONFIGURAÇÕES PRINCIPAIS ======
# YAML_PATH = r"E:\ProjetoDogChow\03_resultados\yolo\240726\dataset_final_sem_kfold_24_imagens\data.yaml"
# MODEL_W   = "yolov8n.pt"
# DEVICE    = 0
# WORKERS   = 0
# EPOCHS    = 350
# PATIENCE  = 100
# PROJECT   = Path(YAML_PATH).parent / "runs"
# RUN_NAME  = "conforme_yolov8n_autobackoff"
# VAL_SPLIT = "train"          # usado apenas se USE_KFOLD=False e para validação extra

# TRY_SETTINGS = [(1024, 4), (1024, 2), (960, 4), (960, 2), (896, 4), (896, 2)]

# # Export configs
# OPSET        = 11
# DYNAMIC      = False
# SIMPLIFY     = True
# WITH_NMS     = False

# # Pasta de artefatos finais
# ARTEFACTS_ROOT = Path(YAML_PATH).parent / "artefatos_treino"

# # ====== CONFIGURAÇÕES K‑FOLD ======
# USE_KFOLD = False                # True para ativar validação cruzada
# K_FOLDS = 5                     # número de dobras
# KFOLD_SEED = 42                 # semente para reprodutibilidade

# # ====== CONFIGURAÇÕES DE GRÁFICOS ======
# SHOW_PLOTS = False              # True para exibir gráficos na tela (além de salvar)

# # ====== TESTE ======  # <-- ADIÇÃO
# TESTAR_MODELO = True                # True para testar após o treino
# TEST_SPLIT = "test"                  # split usado no data.yaml (ou 'test')
# RESULTADOS_TESTE_DIR = Path(YAML_PATH).parent / "resultados_teste"
# # ===================

# def show_png(png_path: Path, title="Matriz de confusão"):
#     """Exibe uma imagem PNG (matriz de confusão)."""
#     if not png_path.exists():
#         print("[AVISO] Não encontrei:", png_path)
#         return
#     img = mpimg.imread(str(png_path))
#     plt.figure(figsize=(8, 8))
#     plt.imshow(img)
#     plt.axis("off")
#     plt.title(title)
#     plt.tight_layout()
#     if SHOW_PLOTS:
#         plt.show()
#     else:
#         plt.close()


# def plot_metrics_from_csv(csv_path: Path, save_dir: Path, title_prefix: str = ""):
#     """
#     Lê o results.csv do YOLO e gera gráficos de loss e métricas.
#     Salva como 'metrics.png' em save_dir.
#     """
#     if not csv_path.exists():
#         print(f"[AVISO] {csv_path} não encontrado. Pulando gráficos.")
#         return

#     df = pd.read_csv(csv_path)
#     # Verifica colunas esperadas
#     loss_cols = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
#     metric_cols = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
#     epochs = df['epoch'].values

#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#     fig.suptitle(f"{title_prefix} Evolução das métricas" if title_prefix else "Evolução das métricas", fontsize=14)

#     # Losses
#     ax = axes[0, 0]
#     for col in loss_cols:
#         if col in df.columns:
#             ax.plot(epochs, df[col].values, label=col.replace('train/', ''))
#     ax.set_xlabel('Época')
#     ax.set_ylabel('Loss')
#     ax.set_title('Losses (treino)')
#     ax.legend()
#     ax.grid(True)

#     # Métricas
#     ax = axes[0, 1]
#     for col in metric_cols:
#         if col in df.columns:
#             ax.plot(epochs, df[col].values, label=col.replace('metrics/', '').replace('(B)', ''))
#     ax.set_xlabel('Época')
#     ax.set_ylabel('Valor')
#     ax.set_title('Métricas (validação)')
#     ax.legend()
#     ax.grid(True)

#     # Box_loss em detalhe
#     ax = axes[1, 0]
#     if 'train/box_loss' in df.columns:
#         ax.plot(epochs, df['train/box_loss'].values, label='box_loss', color='red')
#     if 'train/cls_loss' in df.columns:
#         ax.plot(epochs, df['train/cls_loss'].values, label='cls_loss', color='green')
#     if 'train/dfl_loss' in df.columns:
#         ax.plot(epochs, df['train/dfl_loss'].values, label='dfl_loss', color='blue')
#     ax.set_xlabel('Época')
#     ax.set_ylabel('Loss')
#     ax.set_title('Losses detalhadas')
#     ax.legend()
#     ax.grid(True)

#     # mAP50-95 em destaque
#     ax = axes[1, 1]
#     if 'metrics/mAP50-95(B)' in df.columns:
#         ax.plot(epochs, df['metrics/mAP50-95(B)'].values, label='mAP50-95', color='purple', linewidth=2)
#     if 'metrics/mAP50(B)' in df.columns:
#         ax.plot(epochs, df['metrics/mAP50(B)'].values, label='mAP50', color='orange', linestyle='--')
#     ax.set_xlabel('Época')
#     ax.set_ylabel('mAP')
#     ax.set_title('mAP')
#     ax.legend()
#     ax.grid(True)

#     plt.tight_layout()
#     out_path = save_dir / "metrics.png"
#     plt.savefig(out_path, dpi=150)
#     print(f"[OK] Gráficos salvos em: {out_path}")
#     if SHOW_PLOTS:
#         plt.show()
#     else:
#         plt.close()


# def train_with_backoff(model: YOLO, yaml_path: str, project_dir: Optional[Path] = None,
#                        run_name: Optional[str] = None) -> Tuple[any, int, int]:
#     """
#     Treina com backoff usando TRY_SETTINGS.
#     Se project_dir ou run_name forem fornecidos, substitui os padrões.
#     """
#     last_err = None
#     for imgsz, batch in TRY_SETTINGS:
#         print(f"\n[INFO] Tentando treino com imgsz={imgsz}, batch={batch}, amp=False, workers={WORKERS} ...")
#         try:
#             res = model.train(
#                 data=yaml_path,
#                 imgsz=imgsz,
#                 epochs=EPOCHS,
#                 batch=batch,
#                 device=DEVICE,
#                 workers=WORKERS,
#                 project=(project_dir or PROJECT).as_posix(),
#                 name=(run_name or RUN_NAME) + f"_sz{imgsz}_b{batch}",
#                 cos_lr=True,
#                 patience=PATIENCE,
#                 amp=False,
#                 verbose=True,
#                 single_cls=False,
#                 lr0=0.01,
#                 momentum=0.937,
#                 optimizer='SGD',
#                 freeze=5,
#                 weight_decay=0.005,
#                 # NADA MAIS
#             )
#             print("[OK] Treino finalizado com essa configuração.")
#             return res, imgsz, batch
#         except torch.cuda.OutOfMemoryError as e:
#             last_err = e
#             torch.cuda.empty_cache()
#             print("[OOM] Ficou sem VRAM. Tentando próxima combinação...")
#         except Exception as e:
#             last_err = e
#             print(f"[ERRO] Tentativa com imgsz={imgsz}, batch={batch} falhou: {e}")
#     raise RuntimeError(f"Todas as tentativas falharam. Último erro: {last_err}")


# def export_ultralytics_4plusnc(model: YOLO, imgsz: int, out_dir: Path) -> Path:
#     """Exporta ONNX padrão Ultralytics (4+nc, sem NMS)."""
#     print("\n[INFO] Exportando ONNX 4+nc (padrão Ultralytics, sem NMS)...")
#     onnx_path = model.export(
#         format="onnx",
#         imgsz=imgsz,
#         opset=OPSET,
#         dynamic=DYNAMIC,
#         simplify=SIMPLIFY,
#         half=False,
#         nms=WITH_NMS
#     )
#     onnx_path = Path(onnx_path)
#     dst = out_dir / "best_ultra_4plusnc.onnx"
#     if dst.exists():
#         dst.unlink()
#     shutil.move(str(onnx_path), str(dst))
#     print("[OK] ONNX 4+nc:", dst)
#     return dst


# class YoloV8_HSI_Wrapper(nn.Module):
#     """Emite (B, 5+nc, N): [cx, cy, w, h, obj, cls0..clsN]."""
#     def __init__(self, det_model: nn.Module, nc: int):
#         super().__init__()
#         self.det = det_model
#         self.nc = nc
#         self.det.export = True

#     def forward(self, x):
#         y = self.det(x)
#         if isinstance(y, (tuple, list)):
#             y = y[0]
#         if not isinstance(y, torch.Tensor):
#             raise RuntimeError(f"Saída inesperada: {type(y)}")
#         if y.dim() != 3:
#             raise RuntimeError(f"Saída inesperada do head: shape={tuple(y.shape)}")

#         B, A, C = y.shape
#         if C == 4 + self.nc:
#             y = y.permute(0, 2, 1).contiguous()
#         elif A == 4 + self.nc:
#             pass
#         else:
#             raise RuntimeError(f"Formato inesperado: {(B, A, C)} com nc={self.nc}")

#         xywh = y[:, 0:4, :]
#         cls_logits = y[:, 4:4+self.nc, :]
#         obj = torch.sigmoid(cls_logits).amax(dim=1, keepdim=True)
#         y_hsi = torch.cat([xywh, obj, cls_logits], dim=1)
#         return y_hsi


# def export_hsi_5plusnc(best_pt: Path, imgsz: int, out_dir: Path) -> Path:
#     """Exporta ONNX 5+nc (compat. HSI), sem NMS."""
#     print("\n[INFO] Exportando ONNX 5+nc (compatível HSI, sem NMS)...")
#     y = YOLO(best_pt.as_posix())
#     names = y.names if isinstance(y.names, dict) else {i: n for i, n in enumerate(y.names)}
#     nc = len(names)
#     head = y.model.model[-1]
#     print(f"[CHECK] nc={nc}, model.nc={getattr(y.model,'nc',None)}, head.nc={getattr(head,'nc',None)}")
#     assert nc == getattr(y.model, "nc", None) == getattr(head, "nc", None), "nc divergente."

#     wrapper = YoloV8_HSI_Wrapper(y.model, nc).eval()
#     dummy = torch.randn(1, 3, imgsz, imgsz)

#     out_path = out_dir / "best_hsi_5plusnc.onnx"
#     if out_path.exists():
#         out_path.unlink()

#     torch.onnx.export(
#         wrapper, dummy, out_path.as_posix(),
#         opset_version=OPSET,
#         input_names=["images"],
#         output_names=["output"],
#         dynamic_axes=None if not DYNAMIC else {
#             "images": {0: "batch", 2: "height", 3: "width"},
#             "output": {0: "batch", 2: "anchors"}
#         },
#         do_constant_folding=True
#     )
#     print("[OK] ONNX 5+nc:", out_path)
#     return out_path


# def optional_check_runtime(onnx_path: Path, exp_channels: int, imgsz: int):
#     """Checa shape via ONNX Runtime."""
#     try:
#         import onnxruntime as ort
#         import numpy as np
#         sess = ort.InferenceSession(onnx_path.as_posix(), providers=['CPUExecutionProvider'])
#         x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
#         y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
#         print(f"[CHECK] {onnx_path.name} runtime shape: {y.shape}")
#         if y.shape[1] != exp_channels:
#             print(f"[ALERTA] Esperado 2º eixo = {exp_channels}, obtido = {y.shape[1]}")
#     except Exception as e:
#         print(f"[AVISO] Checagem runtime pulada: {e}")


# def run_kfold(yaml_path: str, base_model: str) -> Tuple[Path, dict, List[dict]]:
#     """
#     Executa K‑Fold cross‑validation.
#     Retorna (melhor_modelo_path, informacoes_do_melhor_fold, lista_de_resultados).
#     """
#     print(f"\n===== INICIANDO K‑FOLD (K={K_FOLDS}) =====")

#     # Carrega configuração original
#     yaml_dir = Path(yaml_path).parent
#     with open(yaml_path, 'r') as f:
#         data_cfg = yaml.safe_load(f)

#     # Resolve o caminho das imagens de treino RELATIVO ao diretório do YAML
#     train_img_dir = yaml_dir / data_cfg['train']
#     if not train_img_dir.exists():
#         raise FileNotFoundError(f"Diretório de imagens não encontrado: {train_img_dir}")

#     # Lista todas as imagens de treino (extensões comuns)
#     img_exts = ['.jpg', '.jpeg', '.png', '.bmp']
#     all_images = sorted([p for p in train_img_dir.glob('*') if p.suffix.lower() in img_exts])
#     if not all_images:
#         raise RuntimeError(f"Nenhuma imagem encontrada em {train_img_dir}")

#     print(f"[K-FOLD] Total de imagens encontradas: {len(all_images)}")
#     random.seed(KFOLD_SEED)
#     random.shuffle(all_images)

#     # Dividir em K folds
#     fold_size = len(all_images) // K_FOLDS
#     folds = []
#     for i in range(K_FOLDS):
#         start = i * fold_size
#         end = start + fold_size if i < K_FOLDS - 1 else len(all_images)
#         folds.append(all_images[start:end])

#     best_metric = -1
#     best_model_path = None
#     best_fold_info = None
#     all_results = []

#     kfold_root = yaml_dir / "kfold_runs"
#     kfold_root.mkdir(exist_ok=True)

#     for fold_idx, val_images in enumerate(folds):
#         # Conjunto de treino = todas as outras folds
#         train_images = []
#         for j, f in enumerate(folds):
#             if j != fold_idx:
#                 train_images.extend(f)

#         # Estrutura de pastas para este fold
#         fold_dir = kfold_root / f"fold_{fold_idx+1}"
#         fold_dir.mkdir(parents=True, exist_ok=True)

#         train_img_dir_fold = fold_dir / "train" / "images"
#         val_img_dir_fold = fold_dir / "val" / "images"
#         train_lbl_dir_fold = fold_dir / "train" / "labels"
#         val_lbl_dir_fold = fold_dir / "val" / "labels"
#         for d in [train_img_dir_fold, val_img_dir_fold, train_lbl_dir_fold, val_lbl_dir_fold]:
#             d.mkdir(parents=True, exist_ok=True)

#         # Função para copiar imagens e labels
#         def copy_files(img_list, img_dest, lbl_dest):
#             copied_imgs = 0
#             copied_lbls = 0
#             for img_p in img_list:
#                 # Copia imagem
#                 try:
#                     shutil.copy2(img_p, img_dest / img_p.name)
#                     copied_imgs += 1
#                 except Exception as e:
#                     print(f"[AVISO] Falha ao copiar imagem {img_p}: {e}")
#                     continue

#                 # Monta caminho do label correspondente
#                 # Estrutura esperada: images/train -> labels/train (mesmo nome, extensão .txt)
#                 # train_img_dir = .../images/train
#                 # labels devem estar em .../labels/train
#                 lbl_base = train_img_dir.parent.parent / "labels" / train_img_dir.name
#                 lbl_path = lbl_base / img_p.name
#                 lbl_path = lbl_path.with_suffix('.txt')
#                 if lbl_path.exists():
#                     shutil.copy2(lbl_path, lbl_dest / lbl_path.name)
#                     copied_lbls += 1
#                 else:
#                     # Tenta outras estruturas comuns (fallback)
#                     fallbacks = [
#                         train_img_dir.parent.parent / "labels" / img_p.name,
#                         train_img_dir.parent / "labels" / train_img_dir.name / img_p.name,
#                         train_img_dir.parent / "labels" / img_p.name,
#                     ]
#                     found = False
#                     for fb in fallbacks:
#                         fb = fb.with_suffix('.txt')
#                         if fb.exists():
#                             shutil.copy2(fb, lbl_dest / fb.name)
#                             copied_lbls += 1
#                             found = True
#                             break
#                     if not found:
#                         print(f"[AVISO] Label não encontrado para {img_p.name}")
#             return copied_imgs, copied_lbls

#         # Copiar treino e validação
#         train_imgs, train_lbls = copy_files(train_images, train_img_dir_fold, train_lbl_dir_fold)
#         val_imgs, val_lbls = copy_files(val_images, val_img_dir_fold, val_lbl_dir_fold)

#         print(f"[K-FOLD] Fold {fold_idx+1}: {train_imgs} imagens treino, {train_lbls} labels; {val_imgs} val, {val_lbls} labels")

#         # Se não houver dados suficientes, pula este fold
#         if train_imgs == 0 or train_lbls == 0:
#             print(f"[ERRO] Fold {fold_idx+1} sem dados suficientes. Pulando.")
#             continue

#         # Cria data.yaml para este fold
#         fold_yaml = fold_dir / "data.yaml"
#         fold_cfg = {
#             'path': str(fold_dir),
#             'train': str(train_img_dir_fold.relative_to(fold_dir)),
#             'val': str(val_img_dir_fold.relative_to(fold_dir)),
#             'nc': data_cfg.get('nc', 1),
#             'names': data_cfg.get('names', ['objeto']),
#         }
#         with open(fold_yaml, 'w') as f:
#             yaml.dump(fold_cfg, f)

#         print(f"\n[K-FOLD] Treinando fold {fold_idx+1}/{K_FOLDS} ...")
#         model = YOLO(base_model)
#         try:
#             # Treina com backoff
#             res, imgsz, batch = train_with_backoff(
#                 model,
#                 str(fold_yaml),
#                 project_dir=kfold_root,
#                 run_name=f"fold_{fold_idx+1}"
#             )
#             train_dir = Path(res.save_dir)
#             best_w = train_dir / "weights" / "best.pt"
#             if not best_w.exists():
#                 print(f"[ERRO] Fold {fold_idx+1} não gerou best.pt. Pulando.")
#                 continue

#             # Avalia no conjunto de validação do fold
#             val_model = YOLO(best_w)
#             val_res = val_model.val(
#                 data=str(fold_yaml),
#                 imgsz=imgsz,
#                 device=DEVICE,
#                 workers=WORKERS,
#                 split='val',
#                 project=kfold_root.as_posix(),
#                 name=f"fold_{fold_idx+1}_val",
#                 plots=True
#             )
#             map50_95 = val_res.results_dict.get('metrics/mAP50-95(B)', 0.0)
#             print(f"[K-FOLD] Fold {fold_idx+1} mAP50-95 = {map50_95:.4f}")

#             # Gera gráficos
#             csv_path = train_dir / "results.csv"
#             if csv_path.exists():
#                 plot_metrics_from_csv(csv_path, train_dir, title_prefix=f"Fold {fold_idx+1}")

#             fold_info = {
#                 'fold': fold_idx+1,
#                 'map50_95': map50_95,
#                 'best_pt': best_w,
#                 'train_dir': train_dir,
#                 'val_dir': Path(val_res.save_dir),
#                 'imgsz': imgsz,
#                 'batch': batch,
#                 'yaml': fold_yaml
#             }
#             all_results.append(fold_info)

#             if map50_95 > best_metric:
#                 best_metric = map50_95
#                 best_model_path = best_w
#                 best_fold_info = fold_info

#         except Exception as e:
#             print(f"[ERRO] Fold {fold_idx+1} falhou: {e}")
#             continue

#     if not all_results:
#         raise RuntimeError("Nenhum fold foi concluído com sucesso.")

#     # Estatísticas finais
#     map_values = [r['map50_95'] for r in all_results]
#     print("\n===== RESULTADOS K‑FOLD =====")
#     for r in all_results:
#         print(f"Fold {r['fold']}: mAP50-95 = {r['map50_95']:.4f}")
#     print(f"Média mAP50-95: {np.mean(map_values):.4f} ± {np.std(map_values):.4f}")
#     print(f"Melhor fold: {best_fold_info['fold']} com mAP50-95 = {best_metric:.4f}")

#     return best_model_path, best_fold_info, all_results


# def main():
#     print("[INFO] YAML:", YAML_PATH)
#     base = YOLO(MODEL_W)
#     print("[DEBUG] base names:", base.names)

#     if USE_KFOLD:
#         # Executa K-Fold
#         best_pt, best_fold_info, all_results = run_kfold(YAML_PATH, MODEL_W)
#         if best_pt is None:
#             raise RuntimeError("K‑Fold não produziu modelo válido.")

#         # Recarrega o melhor modelo
#         model = YOLO(best_pt)
#         used_imgsz = best_fold_info['imgsz']
#         used_batch = best_fold_info['batch']
#         train_dir = best_fold_info['train_dir']
#         val_dir_final = best_fold_info['val_dir']

#         # Avaliação final no melhor modelo
#         print(f"\n[INFO] Avaliando melhor modelo (fold {best_fold_info['fold']}) no split='{VAL_SPLIT}'...")
#         val_res = model.val(
#             data=YAML_PATH,
#             imgsz=used_imgsz,
#             device=DEVICE,
#             workers=WORKERS,
#             split=VAL_SPLIT,
#             project=PROJECT.as_posix(),
#             name=f"{train_dir.name}_val_final",
#             save_json=True,
#             save_conf=True,
#             plots=True
#         )
#         val_dir_final = Path(val_res.save_dir)

#         # Gera gráficos do melhor modelo
#         csv_path = train_dir / "results.csv"
#         if csv_path.exists():
#             plot_metrics_from_csv(csv_path, train_dir, title_prefix="Melhor modelo (K-Fold)")

#         # Exibe matriz de confusão
#         cm_val = val_dir_final / "confusion_matrix.png"
#         cm_train = train_dir / "confusion_matrix.png"
#         if cm_val.exists():
#             show_png(cm_val, title=f"Matriz de confusão (split='{VAL_SPLIT}')")
#         elif cm_train.exists():
#             show_png(cm_train, title="Matriz de confusão (treino)")
#         else:
#             print("[AVISO] Sem confusion_matrix.png — veja a pasta:", train_dir)

#     else:
#         # Treino único
#         model = YOLO(MODEL_W)
#         train_res, used_imgsz, used_batch = train_with_backoff(model, YAML_PATH)
#         train_dir = Path(train_res.save_dir)
#         best_pt = train_dir / "weights" / "best.pt"
#         if best_pt.exists():
#             print("[INFO] Carregando pesos 'best':", best_pt)
#             model = YOLO(best_pt.as_posix())
#             print("[DEBUG] best names:", model.names, "nc:", len(model.names))
#             head = model.model.model[-1]
#             print("[DEBUG] model.nc:", getattr(model.model, "nc", None), "head.nc:", getattr(head, "nc", None))

#         print(f"[INFO] Avaliando (split='{VAL_SPLIT}') para gerar matriz de confusão…")
#         val_res = model.val(
#             data=YAML_PATH,
#             imgsz=used_imgsz,
#             device=DEVICE,
#             workers=WORKERS,
#             split=VAL_SPLIT,
#             project=PROJECT.as_posix(),
#             name=train_dir.name + "_val",
#             save_json=True,
#             save_conf=True,
#             plots=True
#         )
#         val_dir_final = Path(val_res.save_dir)

#         csv_path = train_dir / "results.csv"
#         if csv_path.exists():
#             plot_metrics_from_csv(csv_path, train_dir, title_prefix="Treino único")

#         cm_val = val_dir_final / "confusion_matrix.png"
#         cm_train = train_dir / "confusion_matrix.png"
#         if cm_val.exists():
#             show_png(cm_val, title=f"Matriz de confusão (split='{VAL_SPLIT}')")
#         elif cm_train.exists():
#             show_png(cm_train, title="Matriz de confusão (treino)")
#         else:
#             print("[AVISO] Sem confusion_matrix.png — veja a pasta:", train_dir)

#     # ====== TESTE NO CONJUNTO DE TESTE ======  # <-- ADIÇÃO
#     if TESTAR_MODELO:
#         print("\n[INFO] Iniciando teste no conjunto de teste...")
#         # Verifica se o data.yaml tem o campo 'test'
#         with open(YAML_PATH, 'r') as f:
#             test_cfg = yaml.safe_load(f)
#         if 'test' not in test_cfg:
#             print("[AVISO] O data.yaml não contém o campo 'test'. Teste ignorado.")
#         else:
#             # Cria pasta de resultados de teste
#             RESULTADOS_TESTE_DIR.mkdir(parents=True, exist_ok=True)
#             # Executa validação no teste
#             test_res = model.val(
#                 data=YAML_PATH,
#                 split=TEST_SPLIT,
#                 imgsz=used_imgsz,
#                 device=DEVICE,
#                 workers=WORKERS,
#                 project=RESULTADOS_TESTE_DIR.as_posix(),
#                 name="teste_final",
#                 save_json=True,
#                 save_conf=True,
#                 plots=True
#             )
#             # Exibe métricas
#             print("\n===== MÉTRICAS NO CONJUNTO DE TESTE =====")
#             print(f"mAP50-95: {test_res.box.map:.4f}")
#             print(f"mAP50:     {test_res.box.map50:.4f}")
#             print(f"Precisão:  {test_res.box.mp:.4f}")
#             print(f"Recall:    {test_res.box.mr:.4f}")
#             if hasattr(test_res.box, 'ap_class_index'):
#                 class_names = test_res.names
#                 ap50_95 = test_res.box.ap
#                 ap50 = test_res.box.ap50
#                 for i, idx in enumerate(test_res.box.ap_class_index):
#                     nome = class_names[idx]
#                     print(f"  {nome}: mAP50-95={ap50_95[i]:.4f}, mAP50={ap50[i]:.4f}")
#             # Salva relatório
#             relatorio_path = RESULTADOS_TESTE_DIR / "teste_final" / "resultados_teste.txt"
#             if relatorio_path.parent.exists():
#                 with open(relatorio_path, 'w') as f:
#                     f.write("===== RESULTADOS DO TESTE =====\n")
#                     f.write(f"mAP50-95: {test_res.box.map:.4f}\n")
#                     f.write(f"mAP50:     {test_res.box.map50:.4f}\n")
#                     f.write(f"Precisão:  {test_res.box.mp:.4f}\n")
#                     f.write(f"Recall:    {test_res.box.mr:.4f}\n")
#                     if hasattr(test_res.box, 'ap_class_index'):
#                         f.write("\nPor classe:\n")
#                         for i, idx in enumerate(test_res.box.ap_class_index):
#                             nome = class_names[idx]
#                             f.write(f"  {nome}: mAP50-95={ap50_95[i]:.4f}, mAP50={ap50[i]:.4f}\n")
#                 print(f"[OK] Relatório de teste salvo em: {relatorio_path}")

#     # ===== EXPORTS =====
#     stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     artef_dir = ARTEFACTS_ROOT / f"{train_dir.name}_{stamp}"
#     artef_dir.mkdir(parents=True, exist_ok=True)
#     print("[OK] Pasta de artefatos:", artef_dir)

#     onnx_ultra = export_ultralytics_4plusnc(model, used_imgsz, artef_dir)
#     optional_check_runtime(onnx_ultra, exp_channels=4 + len(model.names), imgsz=used_imgsz)

#     onnx_hsi = export_hsi_5plusnc(best_pt, used_imgsz, artef_dir)
#     optional_check_runtime(onnx_hsi, exp_channels=5 + len(model.names), imgsz=used_imgsz)

#     def copy_if_exists(src: Path, rename: Optional[str] = None):
#         if src and src.exists():
#             dst = artef_dir / (rename if rename else src.name)
#             shutil.copy2(src, dst)
#             print(f"[OK] Copiado: {src} -> {dst}")

#     copy_if_exists(best_pt)
#     copy_if_exists(train_dir / "results.csv")
#     copy_if_exists(cm_train, rename="confusion_matrix_train.png")
#     copy_if_exists(cm_val,   rename="confusion_matrix_val.png")
#     copy_if_exists(val_dir_final / "predictions.json")

#     if USE_KFOLD:
#         stats_file = artef_dir / "kfold_stats.txt"
#         with open(stats_file, 'w', encoding='utf-8') as f:
#             f.write("===== K-FOLD RESULTS =====\n")
#             for r in all_results:
#                 f.write(f"Fold {r['fold']}: mAP50-95 = {r['map50_95']:.4f}\n")
#             f.write(f"Média: {np.mean([r['map50_95'] for r in all_results]):.4f} ± {np.std([r['map50_95'] for r in all_results]):.4f}\n")
#         print(f"[OK] Estatísticas K‑Fold salvas em {stats_file}")

#     # Copia também os resultados do teste (se existirem)  # <-- ADIÇÃO
#     if TESTAR_MODELO:
#         test_dir = RESULTADOS_TESTE_DIR / "teste_final"
#         if test_dir.exists():
#             copy_if_exists(test_dir / "confusion_matrix.png", rename="confusion_matrix_test.png")
#             copy_if_exists(test_dir / "predictions.json", rename="predictions_test.json")

#     print("\n[OK] Finalizado.")
#     print("Config usada:")
#     print(f" - imgsz={used_imgsz}, batch={used_batch}, amp=False, workers={WORKERS}")
#     print("Runs:")
#     print(" - Treino:", train_dir)
#     print(" - Avaliação:", val_dir_final)
#     if TESTAR_MODELO:
#         test_dir = RESULTADOS_TESTE_DIR / "teste_final"
#         if test_dir.exists():
#             print(" - Teste:", test_dir)
#     print(" - Artefatos:", artef_dir)
#     print("Arquivos ONNX:")
#     print(" - Ultralytics (4+nc):", onnx_ultra)
#     print(" - HSI (5+nc):        ", onnx_hsi)


# if __name__ == "__main__":
#     main()



















































# # treinar_yolo.py
# # Requisitos:
# #   pip install ultralytics matplotlib
# # (opcional para checagem de ONNX em runtime)
# #   pip install onnxruntime numpy
# #
# # O script faz:
# # 1) Define PYTORCH_CUDA_ALLOC_CONF para reduzir fragmentação de VRAM
# # 2) Tenta treinar com backoff de (imgsz, batch) para caber em 4GB
# # 3) Desativa AMP e usa workers=0 (Windows)
# # 4) Avalia (split configurável) para gerar/confimar matriz de confusão
# # 5) Exporta dois ONNX: 4+nc (padrão) e 5+nc (compatível HSI)

# import os
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# from pathlib import Path
# import shutil
# from datetime import datetime
# from typing import Optional, Tuple

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import torch
# import torch.nn as nn
# from ultralytics import YOLO

# # ====== CONFIG ======
# YAML_PATH = r"E:\ProjetoDogChow\03_resultados\yolo\230726\dataset_final\data.yaml"
# MODEL_W   = "yolov8n.pt"
# DEVICE    = 0
# WORKERS   = 0
# EPOCHS    = 250
# PATIENCE  = 50
# PROJECT   = Path(YAML_PATH).parent / "runs"
# RUN_NAME  = "conforme_yolov8n_autobackoff"
# VAL_SPLIT = "train"  # use "val" se seu dataset tiver split de validação

# TRY_SETTINGS = [(1024, 4), (1024, 2), (960, 4), (960, 2), (896, 4), (896, 2)]

# # Export configs
# OPSET        = 11      # mais compatível
# DYNAMIC      = False   # shape estático facilita integração
# SIMPLIFY     = True
# WITH_NMS     = False   # mantenha False (pós-processamento fora)

# # Pasta de artefatos
# ARTEFACTS_ROOT = Path(YAML_PATH).parent / "artefatos_treino"
# # ====================


# def show_png(png_path: Path, title="Matriz de confusão"):
#     if not png_path.exists():
#         print("[AVISO] Não encontrei:", png_path)
#         return
#     img = mpimg.imread(str(png_path))
#     plt.figure(figsize=(8, 8))
#     plt.imshow(img)
#     plt.axis("off")
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()


# def train_with_backoff(model: YOLO, yaml_path: str) -> Tuple[any, int, int]:
#     last_err = None
#     for imgsz, batch in TRY_SETTINGS:
#         print(f"\n[INFO] Tentando treino com imgsz={imgsz}, batch={batch}, amp=False, workers={WORKERS} ...")
#         try:
#             res = model.train(
#                 data=yaml_path,
#                 imgsz=imgsz,
#                 epochs=EPOCHS,
#                 batch=batch,
#                 device=DEVICE,
#                 workers=WORKERS,
#                 project=PROJECT.as_posix(),
#                 name=RUN_NAME + f"_sz{imgsz}_b{batch}",
#                 cos_lr=True,
#                 patience=PATIENCE,
#                 amp=False,
#                 verbose=True,
#                 single_cls=False,
#                 lr0=0.01,          # forçar lr inicial
#                 momentum=0.937,    # forçar momentum
#                 optimizer='SGD'    # ou 'AdamW' (padrão auto já escolheu AdamW)
#             )
#             print("[OK] Treino finalizado com essa configuração.")
#             return res, imgsz, batch
#         except torch.cuda.OutOfMemoryError as e:
#             last_err = e
#             torch.cuda.empty_cache()
#             print("[OOM] Ficou sem VRAM. Tentando próxima combinação...")
#         except Exception as e:
#             last_err = e
#             print(f"[ERRO] Tentativa com imgsz={imgsz}, batch={batch} falhou: {e}")
#     raise RuntimeError(f"Todas as tentativas falharam. Último erro: {last_err}")


# def export_ultralytics_4plusnc(model: YOLO, imgsz: int, out_dir: Path) -> Path:
#     """Exporta ONNX padrão Ultralytics (4+nc, sem NMS)."""
#     print("\n[INFO] Exportando ONNX 4+nc (padrão Ultralytics, sem NMS)...")
#     onnx_path = model.export(
#         format="onnx",
#         imgsz=imgsz,
#         opset=OPSET,
#         dynamic=DYNAMIC,
#         simplify=SIMPLIFY,
#         half=False,
#         nms=WITH_NMS
#     )
#     onnx_path = Path(onnx_path)
#     dst = out_dir / "best_ultra_4plusnc.onnx"
#     if dst.exists():
#         dst.unlink()
#     onnx_path.replace(dst)
#     print("[OK] ONNX 4+nc:", dst)
#     return dst


# class YoloV8_HSI_Wrapper(nn.Module):
#     """
#     Emite (B, 5+nc, N): [cx, cy, w, h, obj, cls0..clsN]
#     com obj = max(sigmoid(cls_logits)) por âncora.
#     """
#     def __init__(self, det_model: nn.Module, nc: int):
#         super().__init__()
#         self.det = det_model
#         self.nc  = nc
#         self.det.export = True  # força caminho de export no forward

#     def forward(self, x):
#         y = self.det(x)
#         # YOLOv8 pode retornar uma tupla (detections, features) ou apenas o tensor
#         if isinstance(y, (tuple, list)):
#             y = y[0]          # pega o tensor de detecções (primeiro elemento)
#         if not isinstance(y, torch.Tensor):
#             raise RuntimeError(f"Saída inesperada: {type(y)}")
#         if y.dim() != 3:
#             raise RuntimeError(f"Saída inesperada do head: shape={tuple(y.shape)}")
        
#         B, A, C = y.shape  # pode ser (B, N, 4+nc) ou (B, 4+nc, N)
#         if C == 4 + self.nc:
#             y = y.permute(0, 2, 1).contiguous()  # (B, 4+nc, N)
#         elif A == 4 + self.nc:
#             pass  # já está (B, 4+nc, N)
#         else:
#             raise RuntimeError(f"Formato inesperado: {(B, A, C)} com nc={self.nc}")

#         xywh = y[:, 0:4, :]                  # (B, 4, N)
#         cls_logits = y[:, 4:4+self.nc, :]    # (B, nc, N)
#         obj = torch.sigmoid(cls_logits).amax(dim=1, keepdim=True)  # (B, 1, N)

#         y_hsi = torch.cat([xywh, obj, cls_logits], dim=1)          # (B, 5+nc, N)
#         return y_hsi


# def export_hsi_5plusnc(best_pt: Path, imgsz: int, out_dir: Path) -> Path:
#     """Exporta ONNX 5+nc (compat. HSI), sem NMS."""
#     print("\n[INFO] Exportando ONNX 5+nc (compatível HSI, sem NMS)...")
#     y = YOLO(best_pt.as_posix())
#     # sanity
#     names = y.names if isinstance(y.names, dict) else {i: n for i, n in enumerate(y.names)}
#     nc = len(names)
#     head = y.model.model[-1]
#     print(f"[CHECK] nc={nc}, model.nc={getattr(y.model,'nc',None)}, head.nc={getattr(head,'nc',None)}")
#     assert nc == getattr(y.model, "nc", None) == getattr(head, "nc", None), "Modelo e head com nc divergente."

#     wrapper = YoloV8_HSI_Wrapper(y.model, nc).eval()
#     dummy = torch.randn(1, 3, imgsz, imgsz)

#     out_path = out_dir / "best_hsi_5plusnc.onnx"
#     if out_path.exists():
#         out_path.unlink()

#     torch.onnx.export(
#         wrapper, dummy, out_path.as_posix(),
#         opset_version=OPSET,
#         input_names=["images"],
#         output_names=["output"],
#         dynamic_axes=None if not DYNAMIC else {
#             "images": {0: "batch", 2: "height", 3: "width"},
#             "output": {0: "batch", 2: "anchors"}
#         },
#         do_constant_folding=True
#     )
#     print("[OK] ONNX 5+nc:", out_path)
#     return out_path


# def optional_check_runtime(onnx_path: Path, exp_channels: int, imgsz: int):
#     """Checa shape via ONNX Runtime (opcional)."""
#     try:
#         import onnxruntime as ort
#         import numpy as np
#         sess = ort.InferenceSession(onnx_path.as_posix(), providers=['CPUExecutionProvider'])
#         x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
#         y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
#         print(f"[CHECK] {onnx_path.name} runtime shape: {y.shape}")
#         if y.shape[1] != exp_channels:
#             print(f"[ALERTA] Esperado 2º eixo = {exp_channels}, obtido = {y.shape[1]}")
#     except Exception as e:
#         print(f"[AVISO] Checagem runtime pulada: {e}")


# def main():
#     print("[INFO] YAML:", YAML_PATH)
#     base = YOLO(MODEL_W)
#     print("[DEBUG] base names:", base.names)

#     model = YOLO(MODEL_W)
#     train_res, used_imgsz, used_batch = train_with_backoff(model, YAML_PATH)
#     train_dir = Path(train_res.save_dir)
#     best_w = train_dir / "weights" / "best.pt"
#     if best_w.exists():
#         print("[INFO] Carregando pesos 'best':", best_w)
#         model = YOLO(best_w.as_posix())
#         print("[DEBUG] best names:", model.names, "nc:", len(model.names))
#         head = model.model.model[-1]
#         print("[DEBUG] model.nc:", getattr(model.model, "nc", None), "head.nc:", getattr(head, "nc", None))

#     print(f"[INFO] Avaliando (split='{VAL_SPLIT}') para gerar matriz de confusão…")
#     val_res = model.val(
#         data=YAML_PATH,
#         imgsz=used_imgsz,
#         device=DEVICE,
#         workers=WORKERS,
#         split=VAL_SPLIT,
#         project=PROJECT.as_posix(),
#         name=train_dir.name + "_val",
#         save_json=True,
#         save_conf=True,
#         plots=True
#     )
#     val_dir = Path(val_res.save_dir)

#     cm_val   = val_dir / "confusion_matrix.png"
#     cm_train = train_dir / "confusion_matrix.png"
#     if cm_val.exists():
#         print("[INFO] Exibindo matriz (val):", cm_val)
#         show_png(cm_val, title=f"Matriz de confusão (split='{VAL_SPLIT}')")
#     elif cm_train.exists():
#         print("[INFO] Exibindo matriz (train):", cm_train)
#         show_png(cm_train, title="Matriz de confusão (durante treino)")
#     else:
#         print("[AVISO] Sem confusion_matrix.png — veja a pasta:", train_dir)

#     # ===== EXPORTS =====
#     # Pasta de artefatos datada
#     stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     artef_dir = ARTEFACTS_ROOT / f"{train_dir.name}_{stamp}"
#     artef_dir.mkdir(parents=True, exist_ok=True)
#     print("[OK] Pasta de artefatos:", artef_dir)

#     # 1) ONNX 4+nc (padrão Ultralytics)
#     onnx_ultra = export_ultralytics_4plusnc(model, used_imgsz, artef_dir)
#     optional_check_runtime(onnx_ultra, exp_channels=4 + len(model.names), imgsz=used_imgsz)

#     # 2) ONNX 5+nc (compatível HSI)
#     onnx_hsi = export_hsi_5plusnc(best_w, used_imgsz, artef_dir)
#     optional_check_runtime(onnx_hsi, exp_channels=5 + len(model.names), imgsz=used_imgsz)

#     # Copiar alguns artefatos úteis
#     def copy_if_exists(src: Path, rename: Optional[str] = None):
#         if src and src.exists():
#             dst = artef_dir / (rename if rename else src.name)
#             shutil.copy2(src, dst)
#             print(f"[OK] Copiado: {src} -> {dst}")

#     copy_if_exists(best_w)
#     copy_if_exists(train_dir / "results.csv")
#     copy_if_exists(cm_train, rename="confusion_matrix_train.png")
#     copy_if_exists(cm_val,   rename="confusion_matrix_val.png")
#     copy_if_exists(val_dir / "predictions.json")

#     print("\n[OK] Finalizado.")
#     print("Config usada:")
#     print(f" - imgsz={used_imgsz}, batch={used_batch}, amp=False, workers={WORKERS}")
#     print("Runs:")
#     print(" - Treino:", train_dir)
#     print(" - Avaliação:", val_dir)
#     print(" - Artefatos:", artef_dir)
#     print("Arquivos ONNX:")
#     print(" - Ultralytics (4+nc):", onnx_ultra)
#     print(" - HSI (5+nc):        ", onnx_hsi)


# if __name__ == "__main__":
#     main()
