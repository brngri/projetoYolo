# treinar_yolo.py
# Requisitos:
#   pip install ultralytics matplotlib
# (opcional para checagem de ONNX em runtime)
#   pip install onnxruntime numpy
#
# O script faz:
# 1) Define PYTORCH_CUDA_ALLOC_CONF para reduzir fragmentação de VRAM
# 2) Tenta treinar com backoff de (imgsz, batch) para caber em 4GB
# 3) Desativa AMP e usa workers=0 (Windows)
# 4) Avalia (split configurável) para gerar/confimar matriz de confusão
# 5) Exporta dois ONNX: 4+nc (padrão) e 5+nc (compatível HSI)

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
import shutil
from datetime import datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
from ultralytics import YOLO

# ====== CONFIG ======
YAML_PATH = r"E:\ProjetoDogChow\01_dados\preprocessamento\06_07_2026_amostragem\amostras_gerais\furado\dataset_yolo_anotador\dataset_split\data.yaml"
MODEL_W   = "yolov8n.pt"
DEVICE    = 0
WORKERS   = 0
EPOCHS    = 300
PATIENCE  = 100
PROJECT   = Path(YAML_PATH).parent / "runs"
RUN_NAME  = "according_yolov8n_autobackoff"
VAL_SPLIT = "train"  # use "val" se seu dataset tiver split de validação
VAL_CONF  = 0.07      # limiar de confianca usado na matriz de confusao/val -- ajuste conforme
                       # o pico da curva F1-Confidence (ver F1_curve.png gerado no treino)

TRY_SETTINGS = [(1024, 4), (1024, 2), (960, 4), (960, 2), (896, 4), (896, 2)]

# Export configs
OPSET        = 11      # mais compatível
DYNAMIC      = False   # shape estático facilita integração
SIMPLIFY     = True
WITH_NMS     = False   # mantenha False (pós-processamento fora)

# Pasta de artefatos
ARTEFACTS_ROOT = Path(YAML_PATH).parent / "artefatos_treino"
# ====================


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
    plt.show()


def train_with_backoff(model: YOLO, yaml_path: str) -> Tuple[any, int, int]:
    last_err = None
    for imgsz, batch in TRY_SETTINGS:
        print(f"\n[INFO] Tentando treino com imgsz={imgsz}, batch={batch}, amp=False, workers={WORKERS} ...")
        try:
            res = model.train(
                data=yaml_path,
                imgsz=imgsz,
                epochs=EPOCHS,
                batch=batch,
                device=DEVICE,
                workers=WORKERS,
                project=PROJECT.as_posix(),
                name=RUN_NAME + f"_sz{imgsz}_b{batch}",
                cos_lr=True,
                patience=PATIENCE,
                amp=False,
                verbose=True,
                single_cls=False
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


def export_ultralytics_4plusnc(model: YOLO, imgsz: int, out_dir: Path) -> Path:
    """Exporta ONNX padrão Ultralytics (4+nc, sem NMS)."""
    print("\n[INFO] Exportando ONNX 4+nc (padrão Ultralytics, sem NMS)...")
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=OPSET,
        dynamic=DYNAMIC,
        simplify=SIMPLIFY,
        half=False,
        nms=WITH_NMS
    )
    onnx_path = Path(onnx_path)
    dst = out_dir / "best_ultra_4plusnc.onnx"
    if dst.exists():
        dst.unlink()
    onnx_path.replace(dst)
    print("[OK] ONNX 4+nc:", dst)
    return dst


class YoloV8_HSI_Wrapper(nn.Module):
    """
    Emite (B, 5+nc, N): [cx, cy, w, h, obj, cls0..clsN]
    com obj = max(sigmoid(cls_logits)) por âncora.
    """
    def __init__(self, det_model: nn.Module, nc: int):
        super().__init__()
        self.det = det_model
        self.nc  = nc
        # A flag "export" que controla o formato de saida do forward vive na
        # camada Detect (ultima camada do model.model), NAO no DetectionModel
        # como um todo. Setar no objeto errado nao tem efeito nenhum -- o
        # forward continua devolvendo uma tupla (y, raw) em vez do tensor
        # unico esperado, causando "AttributeError: 'tuple' object has no
        # attribute 'dim'" mais adiante.
        head = self.det.model[-1]
        head.export = True
        self.det.eval()

    def forward(self, x):
        y = self.det(x)  # saída crua do head
        if y.dim() != 3:
            raise RuntimeError(f"Saída inesperada do head: shape={tuple(y.shape)}")

        B, A, C = y.shape  # pode ser (B, N, 4+nc) ou (B, 4+nc, N)
        if C == 4 + self.nc:
            y = y.permute(0, 2, 1).contiguous()  # (B, 4+nc, N)
        elif A == 4 + self.nc:
            pass  # já está (B, 4+nc, N)
        else:
            raise RuntimeError(f"Formato inesperado: {(B, A, C)} com nc={self.nc}")

        xywh = y[:, 0:4, :]                  # (B, 4, N)
        cls_logits = y[:, 4:4+self.nc, :]    # (B, nc, N)
        obj = torch.sigmoid(cls_logits).amax(dim=1, keepdim=True)  # (B, 1, N)

        y_hsi = torch.cat([xywh, obj, cls_logits], dim=1)          # (B, 5+nc, N)
        return y_hsi


def export_hsi_5plusnc(best_pt: Path, imgsz: int, out_dir: Path) -> Path:
    """Exporta ONNX 5+nc (compat. HSI), sem NMS."""
    print("\n[INFO] Exportando ONNX 5+nc (compatível HSI, sem NMS)...")
    y = YOLO(best_pt.as_posix())
    # sanity
    names = y.names if isinstance(y.names, dict) else {i: n for i, n in enumerate(y.names)}
    nc = len(names)
    head = y.model.model[-1]
    print(f"[CHECK] nc={nc}, model.nc={getattr(y.model,'nc',None)}, head.nc={getattr(head,'nc',None)}")
    assert nc == getattr(y.model, "nc", None) == getattr(head, "nc", None), "Modelo e head com nc divergente."

    wrapper = YoloV8_HSI_Wrapper(y.model, nc).eval()
    dummy = torch.randn(1, 3, imgsz, imgsz)

    out_path = out_dir / "best_hsi_5plusnc.onnx"
    if out_path.exists():
        out_path.unlink()

    torch.onnx.export(
        wrapper, dummy, out_path.as_posix(),
        opset_version=OPSET,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes=None if not DYNAMIC else {
            "images": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "anchors"}
        },
        do_constant_folding=True
    )
    print("[OK] ONNX 5+nc:", out_path)
    return out_path


def optional_check_runtime(onnx_path: Path, exp_channels: int, imgsz: int):
    """Checa shape via ONNX Runtime (opcional)."""
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(onnx_path.as_posix(), providers=['CPUExecutionProvider'])
        x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
        y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        print(f"[CHECK] {onnx_path.name} runtime shape: {y.shape}")
        if y.shape[1] != exp_channels:
            print(f"[ALERTA] Esperado 2º eixo = {exp_channels}, obtido = {y.shape[1]}")
    except Exception as e:
        print(f"[AVISO] Checagem runtime pulada: {e}")


def main():
    print("[INFO] YAML:", YAML_PATH)
    base = YOLO(MODEL_W)
    print("[DEBUG] base names:", base.names)

    model = YOLO(MODEL_W)
    train_res, used_imgsz, used_batch = train_with_backoff(model, YAML_PATH)
    train_dir = Path(train_res.save_dir)
    best_w = train_dir / "weights" / "best.pt"
    if best_w.exists():
        print("[INFO] Carregando pesos 'best':", best_w)
        model = YOLO(best_w.as_posix())
        print("[DEBUG] best names:", model.names, "nc:", len(model.names))
        head = model.model.model[-1]
        print("[DEBUG] model.nc:", getattr(model.model, "nc", None), "head.nc:", getattr(head, "nc", None))

    print(f"[INFO] Avaliando (split='{VAL_SPLIT}', conf={VAL_CONF}) para gerar matriz de confusão…")
    val_res = model.val(
        data=YAML_PATH,
        imgsz=used_imgsz,
        device=DEVICE,
        workers=WORKERS,
        split=VAL_SPLIT,
        conf=VAL_CONF,
        project=PROJECT.as_posix(),
        name=train_dir.name + "_val",
        save_json=True,
        save_conf=True,
        plots=True
    )
    val_dir = Path(val_res.save_dir)

    cm_val   = val_dir / "confusion_matrix.png"
    cm_train = train_dir / "confusion_matrix.png"
    if cm_val.exists():
        print("[INFO] Exibindo matriz (val):", cm_val)
        show_png(cm_val, title=f"Matriz de confusão (split='{VAL_SPLIT}')")
    elif cm_train.exists():
        print("[INFO] Exibindo matriz (train):", cm_train)
        show_png(cm_train, title="Matriz de confusão (durante treino)")
    else:
        print("[AVISO] Sem confusion_matrix.png — veja a pasta:", train_dir)

    # ===== EXPORTS =====
    # Pasta de artefatos datada
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artef_dir = ARTEFACTS_ROOT / f"{train_dir.name}_{stamp}"
    artef_dir.mkdir(parents=True, exist_ok=True)
    print("[OK] Pasta de artefatos:", artef_dir)

    # 1) ONNX 4+nc (padrão Ultralytics)
    onnx_ultra = export_ultralytics_4plusnc(model, used_imgsz, artef_dir)
    optional_check_runtime(onnx_ultra, exp_channels=4 + len(model.names), imgsz=used_imgsz)

    # 2) ONNX 5+nc (compatível HSI)
    onnx_hsi = export_hsi_5plusnc(best_w, used_imgsz, artef_dir)
    optional_check_runtime(onnx_hsi, exp_channels=5 + len(model.names), imgsz=used_imgsz)

    # Copiar alguns artefatos úteis
    def copy_if_exists(src: Path, rename: Optional[str] = None):
        if src and src.exists():
            dst = artef_dir / (rename if rename else src.name)
            shutil.copy2(src, dst)
            print(f"[OK] Copiado: {src} -> {dst}")

    copy_if_exists(best_w)
    copy_if_exists(train_dir / "results.csv")
    copy_if_exists(cm_train, rename="confusion_matrix_train.png")
    copy_if_exists(cm_val,   rename="confusion_matrix_val.png")
    copy_if_exists(val_dir / "predictions.json")

    print("\n[OK] Finalizado.")
    print("Config usada:")
    print(f" - imgsz={used_imgsz}, batch={used_batch}, amp=False, workers={WORKERS}")
    print("Runs:")
    print(" - Treino:", train_dir)
    print(" - Avaliação:", val_dir)
    print(" - Artefatos:", artef_dir)
    print("Arquivos ONNX:")
    print(" - Ultralytics (4+nc):", onnx_ultra)
    print(" - HSI (5+nc):        ", onnx_hsi)


if __name__ == "__main__":
    main()
