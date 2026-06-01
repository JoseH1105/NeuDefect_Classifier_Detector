"""
3_entrenamiento_detector.py
----------------------------
Entrenamiento del detector de regiones defectuosas en acero laminado
mediante fine-tuning de YOLOv8s preentrenado sobre COCO.

YOLOv8 es una arquitectura de detección de objetos en una sola pasada
(single-stage) que predice simultáneamente las coordenadas del bounding
box y la clase del objeto detectado. Su backbone CSPDarknet preentrenado
sobre COCO provee representaciones transferibles al dominio industrial.

La función de pérdida compuesta de YOLOv8 incluye:
    - CIoU Loss      : regresión precisa de coordenadas del bounding box
    - BCE Loss       : clasificación binaria de objetividad por celda
    - DFL Loss       : distribución focal para precisión de bordes

El entrenamiento incluye un estudio de hiperparámetros previo sobre
combinaciones de tamaño de imagen, batch size y learning rate inicial.
Si el CSV del estudio ya existe, se omite y se cargan los resultados
previos directamente.

Salidas:
    models/detector/best.pt
    results/detector/estudio_hiperparametros.csv
    results/detector/curvas_entrenamiento.png
    results/detector/resultados_yolo/         ← generado por Ultralytics

Uso:
    python 3_entrenamiento_detector.py
"""

import os
import csv
import shutil
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
MODEL_DIR     = PROJECT_ROOT / "models/detector"
RESULTS_DIR   = PROJECT_ROOT / "results/detector"
YAML_PATH     = PROJECT_ROOT / "dataset.yaml"
MODEL_PATH    = MODEL_DIR / "best.pt"

DEVICE        = 0 if torch.cuda.is_available() else "cpu"
YOLO_WEIGHTS  = "yolov8s.pt"

# Épocas para el estudio de hiperparámetros (rápido) y entrenamiento final
TUNE_EPOCHS   = 10
TRAIN_EPOCHS  = 100       # YOLOv8 aplica early stopping internamente

# Combinaciones a evaluar en el estudio de hiperparámetros
HYPERPARAM_GRID = [
    {"imgsz": 416, "batch": 16, "lr0": 0.01},
    {"imgsz": 416, "batch": 16, "lr0": 0.001},
    {"imgsz": 640, "batch": 16, "lr0": 0.01},
    {"imgsz": 640, "batch": 16, "lr0": 0.001},
]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. VERIFICACIÓN DEL DATASET
# ─────────────────────────────────────────────

def verify_dataset():
    """
    Verifica que el dataset procesado y el archivo dataset.yaml están
    disponibles y correctamente configurados antes de iniciar el entrenamiento.
    Detiene la ejecución si detecta problemas críticos.
    """
    print("\n[0/2] Verificación del dataset")

    if not YAML_PATH.exists():
        print(f"  ERROR: No se encontró {YAML_PATH}")
        print("  Ejecute primero: python 1_preparacion_datos.py")
        exit(1)

    for split in ["train", "val"]:
        img_dir = PROCESSED_DIR / "images" / split
        lbl_dir = PROCESSED_DIR / "labels" / split

        n_imgs   = len(list(img_dir.glob("*.jpg")))
        n_labels = len(list(lbl_dir.glob("*.txt")))

        if n_imgs == 0:
            print(f"  ERROR: No se encontraron imágenes en {img_dir}")
            exit(1)

        print(f"  ✔  [{split}] {n_imgs} imágenes | {n_labels} etiquetas")

    print(f"  ✔  dataset.yaml encontrado en {YAML_PATH}")


# ─────────────────────────────────────────────
# 2. ESTUDIO DE HIPERPARÁMETROS
# ─────────────────────────────────────────────

def hyperparameter_study() -> dict:
    """
    Evalúa combinaciones de imgsz, batch y lr0 mediante entrenamientos
    cortos de TUNE_EPOCHS épocas cada uno. Selecciona la configuración
    que produce el mayor mAP@50 en validación.

    Si el CSV de resultados ya existe de una ejecución previa, se omite
    el estudio y se carga directamente la mejor configuración encontrada.

    Retorna:
        Diccionario con la mejor combinación de hiperparámetros.
    """
    csv_path = RESULTS_DIR / "estudio_hiperparametros.csv"

    # Si el CSV ya existe, cargar mejor configuración sin reejecutar
    if csv_path.exists():
        print("\n[1/2] Estudio de hiperparámetros")
        print(f"  – CSV encontrado en {csv_path}, se omite el estudio.")
        best_cfg = None
        best_map = 0.0
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                map50 = float(row["map50"])
                if map50 > best_map:
                    best_map = map50
                    best_cfg = {
                        "imgsz": int(row["imgsz"]),
                        "batch": int(row["batch"]),
                        "lr0":   float(row["lr0"]),
                    }
        print(f"  ✔  Mejor configuración cargada: imgsz={best_cfg['imgsz']}, "
              f"batch={best_cfg['batch']}, lr0={best_cfg['lr0']} "
              f"(mAP50={best_map:.4f})")
        return best_cfg

    print("\n[1/2] Estudio de hiperparámetros")
    print(f"  Combinaciones a evaluar: {len(HYPERPARAM_GRID)}")
    print(f"  Épocas por combinación : {TUNE_EPOCHS}")
    print(f"  Dispositivo            : {DEVICE}\n")

    results  = []
    best_cfg = None
    best_map = 0.0

    for i, cfg in enumerate(HYPERPARAM_GRID):
        imgsz = cfg["imgsz"]
        batch = cfg["batch"]
        lr0   = cfg["lr0"]

        print(f"  ── Combinación {i+1}/{len(HYPERPARAM_GRID)}: "
              f"imgsz={imgsz}, batch={batch}, lr0={lr0}")

        model = YOLO(YOLO_WEIGHTS)

        run_name = f"tune_{i+1}_imgsz{imgsz}_lr{lr0}"
        train_results = model.train(
            data      = str(YAML_PATH),
            epochs    = TUNE_EPOCHS,
            imgsz     = imgsz,
            batch     = batch,
            lr0       = lr0,
            device    = DEVICE,
            project   = str(RESULTS_DIR / "runs_tune"),
            name      = run_name,
            verbose   = False,
            plots     = False,
        )

        # Extraer mAP50 del último epoch
        map50 = float(train_results.results_dict.get("metrics/mAP50(B)", 0.0))
        results.append({
            "imgsz": imgsz, "batch": batch, "lr0": lr0, "map50": map50
        })
        print(f"  → mAP50: {map50:.4f}\n")

        if map50 > best_map:
            best_map = map50
            best_cfg = cfg

    # Guardar resultados del estudio
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["imgsz", "batch", "lr0", "map50"])
        writer.writeheader()
        writer.writerows(results)

    print(f"  ✔  Mejor configuración: imgsz={best_cfg['imgsz']}, "
          f"batch={best_cfg['batch']}, lr0={best_cfg['lr0']} "
          f"(mAP50={best_map:.4f})")
    print(f"  ✔  Resultados guardados en {csv_path}")

    return best_cfg


# ─────────────────────────────────────────────
# 3. ENTRENAMIENTO FINAL
# ─────────────────────────────────────────────

def train_detector(cfg: dict):
    """
    Ejecuta el entrenamiento final de YOLOv8s con la configuración óptima
    encontrada en el estudio de hiperparámetros.

    YOLOv8 aplica early stopping internamente (patience=50 por defecto),
    deteniendo el entrenamiento si el mAP50 no mejora durante ese número
    de épocas consecutivas. El mejor modelo se guarda automáticamente
    como 'best.pt' dentro del directorio de resultados de Ultralytics.

    Parámetros:
        cfg: Diccionario con imgsz, batch y lr0 óptimos.
    """
    print(f"\n[2/2] Entrenamiento final YOLOv8s")
    print(f"  imgsz  : {cfg['imgsz']}")
    print(f"  batch  : {cfg['batch']}")
    print(f"  lr0    : {cfg['lr0']}")
    print(f"  épocas : hasta {TRAIN_EPOCHS} (con early stopping)\n")

    model = YOLO(YOLO_WEIGHTS)

    results = model.train(
        data      = str(YAML_PATH),
        epochs    = TRAIN_EPOCHS,
        imgsz     = cfg["imgsz"],
        batch     = cfg["batch"],
        lr0       = cfg["lr0"],
        device    = DEVICE,
        project   = str(RESULTS_DIR),
        name      = "resultados_yolo",
        plots     = True,       # Genera curvas PR, F1, confusion matrix
        save      = True,
        patience  = 20,         # Early stopping: épocas sin mejora en mAP50
        verbose   = True,
    )

    # Localizar best.pt generado por Ultralytics y copiarlo a models/detector/
    best_src = RESULTS_DIR / "resultados_yolo" / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, MODEL_PATH)
        print(f"\n  ✔  Modelo guardado en {MODEL_PATH}")
    else:
        print(f"\n  ADVERTENCIA: No se encontró best.pt en {best_src}")
        print("  Revise la carpeta results/detector/resultados_yolo/weights/")

    return results


# ─────────────────────────────────────────────
# 4. CURVAS DE ENTRENAMIENTO
# ─────────────────────────────────────────────

def plot_training_curves():
    """
    Genera una figura consolidada con las curvas de pérdida y mAP
    a partir de los archivos de resultados producidos por Ultralytics.
    La imagen se guarda en results/detector/curvas_entrenamiento.png.
    """
    results_png = RESULTS_DIR / "resultados_yolo" / "results.png"

    if not results_png.exists():
        print("  ADVERTENCIA: No se encontró results.png de Ultralytics.")
        return

    # Copiar y renombrar la figura de Ultralytics como curvas_entrenamiento.png
    dst = RESULTS_DIR / "curvas_entrenamiento.png"
    shutil.copy2(results_png, dst)
    print(f"  ✔  Curvas de entrenamiento guardadas en {dst}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  NEU Surface Defect — Entrenamiento Detector YOLOv8s")
    print("=" * 60)
    print(f"\n  Dispositivo : {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Verificar si ya existe un modelo entrenado
    if MODEL_PATH.exists():
        print(f"\n  AVISO: Se encontró un modelo en {MODEL_PATH}")
        resp = input("  ¿Desea reentrenar? (s/n): ").strip().lower()
        if resp != "s":
            print("  Entrenamiento cancelado. Cargando modelo existente.")
            exit(0)

    # Paso 0: Verificación del dataset
    verify_dataset()

    # Paso 1: Estudio de hiperparámetros
    best_cfg = hyperparameter_study()

    # Paso 2: Entrenamiento final
    train_detector(best_cfg)

    # Paso 3: Curvas de entrenamiento
    print("\n  Generando visualizaciones...")
    plot_training_curves()

    print(f"\n{'=' * 60}")
    print("  Entrenamiento del detector completado.")
    print("  Continúe con: python 4_validacion.py")
    print(f"{'=' * 60}")
