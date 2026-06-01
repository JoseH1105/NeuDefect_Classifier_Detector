"""
4_validacion.py
----------------
Validación integral del sistema de inspección visual de dos etapas.
Evalúa individualmente el clasificador ResNet-50 y el detector YOLOv8s,
y realiza una validación cruzada que compara las predicciones de ambas
redes sobre el mismo conjunto de imágenes.

El análisis cruzado identifica tres categorías de resultado:
    - Detección confirmada  : ambas redes coinciden en la clase
    - Fallo del detector    : clasificador detecta clase, detector no localiza
    - Discrepancia          : ambas redes predicen clases distintas

Salidas:
    results/validation/1_metricas_clasificador.png
    results/validation/2_metricas_detector.png
    results/validation/3_validacion_cruzada.png
    results/validation/4_discrepancias.png
    results/validation/5_resumen_comparativo.png
    results/validation/validacion_cruzada.csv
    results/validation/reporte_final.csv

Uso:
    python 4_validacion.py
"""

import csv
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from ultralytics import YOLO
from collections import defaultdict


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
MODEL_CLF     = PROJECT_ROOT / "models/classifier/resnet50_neu.pth"
MODEL_DET     = PROJECT_ROOT / "models/detector/best.pt"
RESULTS_DIR   = PROJECT_ROOT / "results/validation"

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES   = 6
BATCH_SIZE    = 32
CONF_THRESHOLD = 0.35     # Umbral de confianza para YOLOv8 (F1 óptimo)
CLF_CONF_LOW   = 0.80     # Umbral bajo el cual se considera baja confianza

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71",
    "#F39C12", "#9B59B6", "#1ABC9C"
]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. CARGA DE MODELOS Y DATOS
# ─────────────────────────────────────────────

def load_classifier() -> nn.Module:
    """
    Carga el modelo ResNet-50 entrenado desde el archivo de pesos guardado.
    Reconstruye la arquitectura completa antes de cargar los pesos para
    garantizar compatibilidad con el modelo entrenado.

    Retorna:
        Modelo ResNet-50 en modo evaluación listo para inferencia.
    """
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_CLF, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"  ✔  Clasificador cargado desde {MODEL_CLF}")
    return model


def load_val_dataloader() -> DataLoader:
    """
    Carga el conjunto de validación con las mismas transformaciones
    usadas durante el entrenamiento del clasificador (sin augmentation).

    Retorna:
        DataLoader del conjunto de validación.
    """
    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(
        root=PROCESSED_DIR / "images/val",
        transform=val_tf
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=2)


def get_val_image_paths() -> list[Path]:
    """
    Retorna lista ordenada de rutas a las imágenes del conjunto de
    validación, recorriendo todas las subcarpetas por clase.
    """
    val_dir = PROCESSED_DIR / "images/val"
    paths   = sorted(val_dir.glob("**/*.jpg"))
    return paths


# ─────────────────────────────────────────────
# 2. VALIDACIÓN DEL CLASIFICADOR
# ─────────────────────────────────────────────

def validate_classifier(model: nn.Module, dataloader: DataLoader) -> dict:
    """
    Evalúa el clasificador sobre el conjunto de validación completo.
    Calcula métricas por clase y registra las predicciones con su
    nivel de confianza (probabilidad softmax de la clase predicha).

    Parámetros:
        model:      Modelo ResNet-50 en modo evaluación.
        dataloader: DataLoader del conjunto de validación.

    Retorna:
        Diccionario con etiquetas reales, predicciones y confianzas.
    """
    all_labels  = []
    all_preds   = []
    all_confs   = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in dataloader:
            images  = images.to(DEVICE)
            outputs = model(images)
            probs   = softmax(outputs)
            confs, preds = probs.max(1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    return {
        "labels": np.array(all_labels),
        "preds":  np.array(all_preds),
        "confs":  np.array(all_confs),
    }


def plot_classifier_results(clf_results: dict):
    """
    Genera figura con cuatro paneles para el análisis del clasificador:
        1. Matriz de confusión normalizada
        2. F1-score por clase
        3. Distribución de confianza por clase
        4. Casos de baja confianza por clase

    Parámetros:
        clf_results: Diccionario con labels, preds y confs del clasificador.
    """
    labels = clf_results["labels"]
    preds  = clf_results["preds"]
    confs  = clf_results["confs"]

    report = classification_report(
        labels, preds, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Validación del Clasificador ResNet-50", fontsize=14, y=1.01)

    # Panel 1: Matriz de confusión
    ax1 = fig.add_subplot(2, 2, 1)
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax1)
    ax1.set_xticks(range(NUM_CLASSES))
    ax1.set_yticks(range(NUM_CLASSES))
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax1.set_xlabel("Predicción")
    ax1.set_ylabel("Etiqueta real")
    ax1.set_title("Matriz de Confusión Normalizada")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax1.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                     color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=8)

    # Panel 2: F1-score por clase
    ax2  = fig.add_subplot(2, 2, 2)
    f1s  = [report[c]["f1-score"] for c in CLASS_NAMES]
    bars = ax2.barh(CLASS_NAMES, f1s, color=COLORS)
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel("F1-score")
    ax2.set_title("F1-score por Clase")
    ax2.axvline(x=np.mean(f1s), color="black", linestyle="--",
                linewidth=1, label=f"Media: {np.mean(f1s):.3f}")
    ax2.legend(fontsize=8)
    for bar, val in zip(bars, f1s):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=8)
    ax2.grid(True, axis="x", alpha=0.3)

    # Panel 3: Distribución de confianza por clase
    ax3 = fig.add_subplot(2, 2, 3)
    conf_by_class = [confs[labels == i] for i in range(NUM_CLASSES)]
    bp = ax3.boxplot(conf_by_class, labels=CLASS_NAMES, patch_artist=True)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.axhline(y=CLF_CONF_LOW, color="red", linestyle="--",
                linewidth=1, label=f"Umbral baja confianza ({CLF_CONF_LOW})")
    ax3.set_ylabel("Confianza (Softmax)")
    ax3.set_title("Distribución de Confianza por Clase")
    ax3.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax3.legend(fontsize=8)
    ax3.grid(True, axis="y", alpha=0.3)

    # Panel 4: Casos de baja confianza por clase
    ax4 = fig.add_subplot(2, 2, 4)
    low_conf_counts = [
        np.sum((labels == i) & (confs < CLF_CONF_LOW))
        for i in range(NUM_CLASSES)
    ]
    total_counts = [np.sum(labels == i) for i in range(NUM_CLASSES)]
    low_pct      = [l/t*100 if t > 0 else 0
                    for l, t in zip(low_conf_counts, total_counts)]

    bars2 = ax4.bar(CLASS_NAMES, low_pct, color=COLORS, alpha=0.8)
    ax4.set_ylabel("% de imágenes con baja confianza")
    ax4.set_title(f"Casos de Baja Confianza (< {CLF_CONF_LOW})")
    ax4.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    for bar, pct in zip(bars2, low_pct):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{pct:.1f}%", ha="center", fontsize=8)
    ax4.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "1_metricas_clasificador.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Métricas clasificador guardadas en {out}")

    return report


# ─────────────────────────────────────────────
# 3. VALIDACIÓN DEL DETECTOR
# ─────────────────────────────────────────────

def validate_detector(img_paths: list[Path]) -> dict:
    """
    Evalúa el detector YOLOv8s sobre las imágenes de validación.
    Para cada imagen registra: clases detectadas, confianzas y bounding
    boxes. Identifica imágenes sin ninguna detección (falsos negativos totales).

    Parámetros:
        img_paths: Lista de rutas a imágenes de validación.

    Retorna:
        Diccionario con resultados por imagen indexados por ruta.
    """
    detector = YOLO(str(MODEL_DET))
    det_results = {}

    for img_path in img_paths:
        results = detector.predict(
            str(img_path),
            conf=CONF_THRESHOLD,
            verbose=False
        )
        boxes  = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            det_results[img_path] = {
                "classes": boxes.cls.cpu().numpy().astype(int).tolist(),
                "confs":   boxes.conf.cpu().numpy().tolist(),
                "xyxy":    boxes.xyxy.cpu().numpy().tolist(),
                "detected": True
            }
        else:
            det_results[img_path] = {"detected": False, "classes": [], "confs": [], "xyxy": []}

    n_detected = sum(1 for v in det_results.values() if v["detected"])
    print(f"  ✔  Detector: {n_detected}/{len(img_paths)} imágenes con detección")
    return det_results


def plot_detector_results(det_results: dict, img_paths: list[Path]):
    """
    Genera figura con análisis del detector:
        1. Tasa de detección por clase
        2. Distribución de confianza de detecciones por clase
        3. Número promedio de detecciones por imagen por clase
        4. Ejemplos visuales de detecciones (grid 2x3)

    Parámetros:
        det_results: Diccionario con resultados del detector por imagen.
        img_paths:   Lista de rutas a imágenes de validación.
    """
    # Agrupar detecciones por clase real (extraída del nombre de archivo)
    det_by_class    = defaultdict(list)
    nodet_by_class  = defaultdict(int)
    total_by_class  = defaultdict(int)

    for img_path in img_paths:
        true_class = img_path.parent.name
        total_by_class[true_class] += 1
        result = det_results[img_path]
        if result["detected"]:
            det_by_class[true_class].extend(result["confs"])
        else:
            nodet_by_class[true_class] += 1

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Validación del Detector YOLOv8s", fontsize=14)

    # Panel 1: Tasa de detección por clase
    ax1 = fig.add_subplot(2, 2, 1)
    det_rates = [
        (total_by_class[c] - nodet_by_class[c]) / total_by_class[c] * 100
        if total_by_class[c] > 0 else 0
        for c in CLASS_NAMES
    ]
    bars = ax1.barh(CLASS_NAMES, det_rates, color=COLORS, alpha=0.8)
    ax1.set_xlim(0, 110)
    ax1.set_xlabel("Tasa de detección (%)")
    ax1.set_title("Tasa de Detección por Clase")
    for bar, val in zip(bars, det_rates):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9)
    ax1.axvline(x=np.mean(det_rates), color="black", linestyle="--",
                linewidth=1, label=f"Media: {np.mean(det_rates):.1f}%")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="x", alpha=0.3)

    # Panel 2: Distribución de confianza por clase
    ax2 = fig.add_subplot(2, 2, 2)
    conf_data = [det_by_class.get(c, [0]) for c in CLASS_NAMES]
    bp = ax2.boxplot(conf_data, labels=CLASS_NAMES, patch_artist=True)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(y=CONF_THRESHOLD, color="red", linestyle="--",
                linewidth=1, label=f"Umbral ({CONF_THRESHOLD})")
    ax2.set_ylabel("Confianza de detección")
    ax2.set_title("Distribución de Confianza por Clase")
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    # Panel 3: Falsos negativos totales por clase
    ax3 = fig.add_subplot(2, 2, 3)
    fn_counts = [nodet_by_class[c] for c in CLASS_NAMES]
    bars3 = ax3.bar(CLASS_NAMES, fn_counts, color=COLORS, alpha=0.8)
    ax3.set_ylabel("Imágenes sin ninguna detección")
    ax3.set_title("Falsos Negativos Totales por Clase")
    ax3.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    for bar, val in zip(bars3, fn_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 str(val), ha="center", fontsize=9)
    ax3.grid(True, axis="y", alpha=0.3)

    # Panel 4: Ejemplos visuales — una imagen detectada por clase
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    ax4.set_title("Ejemplos de Detección por Clase", pad=10)

    detector = YOLO(str(MODEL_DET))
    example_imgs = []

    for cls_name in CLASS_NAMES:
        cls_paths = [p for p in img_paths if p.parent.name == cls_name
                     and det_results[p]["detected"]]
        if cls_paths:
            img_path = cls_paths[0]
            res      = detector.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)
            annotated = res[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            example_imgs.append((cls_name, annotated))

    if example_imgs:
        n = len(example_imgs)
        cols = 3
        rows = (n + cols - 1) // cols
        sub_fig, sub_axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
        sub_axes = sub_axes.flatten() if n > 1 else [sub_axes]
        for ax, (cls_name, img) in zip(sub_axes, example_imgs):
            ax.imshow(img)
            ax.set_title(cls_name, fontsize=9)
            ax.axis("off")
        for ax in sub_axes[len(example_imgs):]:
            ax.axis("off")
        sub_fig.suptitle("Ejemplos de Detección YOLOv8s por Clase", fontsize=12)
        plt.tight_layout()
        ex_out = RESULTS_DIR / "2b_ejemplos_deteccion.png"
        sub_fig.savefig(ex_out, dpi=150, bbox_inches="tight")
        plt.close(sub_fig)
        print(f"  ✔  Ejemplos de detección guardados en {ex_out}")

    plt.tight_layout()
    out = RESULTS_DIR / "2_metricas_detector.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Métricas detector guardadas en {out}")


# ─────────────────────────────────────────────
# 4. VALIDACIÓN CRUZADA
# ─────────────────────────────────────────────

def cross_validate(
    clf_model:   nn.Module,
    det_results: dict,
    img_paths:   list[Path]
) -> list[dict]:
    """
    Ejecuta ambas redes sobre cada imagen del conjunto de validación
    y compara sus predicciones. Clasifica cada imagen en tres categorías:
        - confirmed  : ambas redes coinciden en la clase
        - fn_detector: clasificador detecta clase, detector no localiza
        - discrepancy: ambas redes predicen clases distintas

    Parámetros:
        clf_model:   Modelo clasificador en modo evaluación.
        det_results: Resultados del detector por imagen.
        img_paths:   Lista de rutas a imágenes de validación.

    Retorna:
        Lista de diccionarios con el resultado de cada imagen.
    """
    softmax  = nn.Softmax(dim=1)
    val_tf   = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from PIL import Image
    cross_results = []

    for img_path in img_paths:
        true_class = img_path.parent.name
        true_idx   = CLASS_NAMES.index(true_class) if true_class in CLASS_NAMES else -1

        # Predicción del clasificador
        img_pil  = Image.open(img_path).convert("RGB")
        img_t    = val_tf(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out      = clf_model(img_t)
            probs    = softmax(out)
            clf_conf, clf_pred = probs.max(1)

        clf_class = CLASS_NAMES[clf_pred.item()]
        clf_conf  = clf_conf.item()

        # Predicción del detector
        det_res   = det_results[img_path]
        det_class = None
        det_conf  = 0.0

        if det_res["detected"] and det_res["classes"]:
            # Clase con mayor confianza detectada
            best_idx  = int(np.argmax(det_res["confs"]))
            det_class = CLASS_NAMES[det_res["classes"][best_idx]]
            det_conf  = det_res["confs"][best_idx]

        # Categorización del resultado
        if not det_res["detected"]:
            status = "fn_detector"
        elif clf_class == det_class:
            status = "confirmed"
        else:
            status = "discrepancy"

        cross_results.append({
            "image":      img_path.name,
            "true_class": true_class,
            "clf_class":  clf_class,
            "clf_conf":   round(clf_conf, 4),
            "det_class":  det_class if det_class else "none",
            "det_conf":   round(det_conf, 4),
            "status":     status,
        })

    return cross_results


def plot_cross_validation(cross_results: list[dict]):
    """
    Genera figura con análisis de la validación cruzada:
        1. Distribución de categorías (confirmados / FN / discrepancias)
        2. Desglose por clase de true label
        3. Confianza del clasificador según categoría
        4. Tabla resumen de discrepancias

    Parámetros:
        cross_results: Lista de resultados por imagen de la validación cruzada.
    """
    statuses   = [r["status"] for r in cross_results]
    confirmed  = statuses.count("confirmed")
    fn_det     = statuses.count("fn_detector")
    discrepancy = statuses.count("discrepancy")
    total      = len(statuses)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Validación Cruzada — Clasificador vs Detector", fontsize=14)

    # Panel 1: Distribución global de categorías
    ax1 = fig.add_subplot(2, 2, 1)
    counts = [confirmed, fn_det, discrepancy]
    labels = [
        f"Confirmadas\n({confirmed})",
        f"FN Detector\n({fn_det})",
        f"Discrepancias\n({discrepancy})"
    ]
    colors_pie = ["#2ECC71", "#E74C3C", "#F39C12"]
    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, colors=colors_pie,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9}
    )
    ax1.set_title("Distribución Global de Resultados")

    # Panel 2: Desglose por clase
    ax2 = fig.add_subplot(2, 2, 2)
    status_by_class = {c: {"confirmed": 0, "fn_detector": 0, "discrepancy": 0}
                       for c in CLASS_NAMES}
    for r in cross_results:
        tc = r["true_class"]
        if tc in status_by_class:
            status_by_class[tc][r["status"]] += 1

    x     = np.arange(len(CLASS_NAMES))
    width = 0.25
    ax2.bar(x - width, [status_by_class[c]["confirmed"]   for c in CLASS_NAMES],
            width, label="Confirmadas",   color="#2ECC71", alpha=0.8)
    ax2.bar(x,         [status_by_class[c]["fn_detector"] for c in CLASS_NAMES],
            width, label="FN Detector",   color="#E74C3C", alpha=0.8)
    ax2.bar(x + width, [status_by_class[c]["discrepancy"] for c in CLASS_NAMES],
            width, label="Discrepancias", color="#F39C12", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Número de imágenes")
    ax2.set_title("Desglose por Clase Real")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    # Panel 3: Confianza del clasificador según categoría
    ax3 = fig.add_subplot(2, 2, 3)
    conf_confirmed   = [r["clf_conf"] for r in cross_results if r["status"] == "confirmed"]
    conf_fn          = [r["clf_conf"] for r in cross_results if r["status"] == "fn_detector"]
    conf_discrepancy = [r["clf_conf"] for r in cross_results if r["status"] == "discrepancy"]

    data_box   = [conf_confirmed, conf_fn, conf_discrepancy]
    labels_box = ["Confirmadas", "FN Detector", "Discrepancias"]
    bp = ax3.boxplot(data_box, labels=labels_box, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors_pie):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.axhline(y=CLF_CONF_LOW, color="black", linestyle="--",
                linewidth=1, label=f"Umbral baja confianza ({CLF_CONF_LOW})")
    ax3.set_ylabel("Confianza del clasificador")
    ax3.set_title("Confianza del Clasificador por Categoría")
    ax3.legend(fontsize=8)
    ax3.grid(True, axis="y", alpha=0.3)

    # Panel 4: Tabla resumen
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    table_data = [
        ["Métrica", "Valor"],
        ["Total imágenes validadas", str(total)],
        ["Detecciones confirmadas", f"{confirmed} ({confirmed/total*100:.1f}%)"],
        ["Fallos del detector (FN)", f"{fn_det} ({fn_det/total*100:.1f}%)"],
        ["Discrepancias entre redes", f"{discrepancy} ({discrepancy/total*100:.1f}%)"],
        ["Confianza media (confirmadas)", f"{np.mean(conf_confirmed):.4f}" if conf_confirmed else "N/A"],
        ["Confianza media (discrepancias)", f"{np.mean(conf_discrepancy):.4f}" if conf_discrepancy else "N/A"],
    ]

    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ECF0F1")
    ax4.set_title("Resumen de Validación Cruzada", pad=15)

    plt.tight_layout()
    out = RESULTS_DIR / "3_validacion_cruzada.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Validación cruzada guardada en {out}")


def plot_discrepancies(cross_results: list[dict], img_paths: list[Path]):
    """
    Genera figura con ejemplos visuales de imágenes donde las predicciones
    del clasificador y el detector no coinciden. Para cada discrepancia
    muestra la imagen original con la predicción de cada red anotada.

    Parámetros:
        cross_results: Lista de resultados de validación cruzada.
        img_paths:     Lista de rutas a imágenes de validación.
    """
    discrepancies = [r for r in cross_results if r["status"] == "discrepancy"]

    if not discrepancies:
        print("  ✔  No se encontraron discrepancias entre redes.")
        return

    # Mostrar hasta 6 ejemplos de discrepancia
    n_show   = min(6, len(discrepancies))
    path_map = {p.name: p for p in img_paths}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Discrepancias entre Clasificador y Detector", fontsize=13)
    axes = axes.flatten()

    detector = YOLO(str(MODEL_DET))

    for i, disc in enumerate(discrepancies[:n_show]):
        img_path = path_map.get(disc["image"])
        if img_path is None:
            continue

        res       = detector.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        annotated = res[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        axes[i].imshow(annotated)
        axes[i].set_title(
            f"Real: {disc['true_class']}\n"
            f"Clf: {disc['clf_class']} ({disc['clf_conf']:.2f}) | "
            f"Det: {disc['det_class']} ({disc['det_conf']:.2f})",
            fontsize=8
        )
        axes[i].axis("off")

    for j in range(n_show, 6):
        axes[j].axis("off")

    plt.tight_layout()
    out = RESULTS_DIR / "4_discrepancias.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Discrepancias guardadas en {out}")


# ─────────────────────────────────────────────
# 5. RESUMEN COMPARATIVO FINAL
# ─────────────────────────────────────────────

def plot_comparative_summary(clf_report: dict, cross_results: list[dict]):
    """
    Genera figura de resumen comparativo entre ambas redes con:
        1. F1-score del clasificador vs AP del detector por clase
        2. Tasa de acuerdo entre redes por clase
        3. Radar chart de rendimiento por clase

    Parámetros:
        clf_report:    Reporte de clasificación del clasificador.
        cross_results: Resultados de la validación cruzada.
    """
    # AP aproximado del detector desde la curva PR (valores de BoxPR_curve)
    det_ap = {
        "crazing":        0.449,
        "inclusion":      0.804,
        "patches":        0.903,
        "pitted_surface": 0.790,
        "rolled-in_scale":0.561,
        "scratches":      0.766,
    }

    clf_f1 = {c: clf_report[c]["f1-score"] for c in CLASS_NAMES}

    # Tasa de acuerdo por clase
    agree_by_class = {}
    for cls in CLASS_NAMES:
        cls_results = [r for r in cross_results if r["true_class"] == cls]
        if cls_results:
            confirmed = sum(1 for r in cls_results if r["status"] == "confirmed")
            agree_by_class[cls] = confirmed / len(cls_results)
        else:
            agree_by_class[cls] = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Resumen Comparativo — ResNet-50 vs YOLOv8s", fontsize=14)

    # Panel 1: F1 clasificador vs AP detector
    ax1 = axes[0]
    x   = np.arange(len(CLASS_NAMES))
    w   = 0.35
    ax1.bar(x - w/2, [clf_f1[c] for c in CLASS_NAMES],
            w, label="F1 Clasificador", color="#3498DB", alpha=0.8)
    ax1.bar(x + w/2, [det_ap[c] for c in CLASS_NAMES],
            w, label="AP@50 Detector", color="#E74C3C", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Métrica")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("F1 (Clasificador) vs AP@50 (Detector)")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel 2: Tasa de acuerdo entre redes por clase
    ax2   = axes[1]
    rates = [agree_by_class[c] * 100 for c in CLASS_NAMES]
    bars  = ax2.bar(CLASS_NAMES, rates, color=COLORS, alpha=0.8)
    ax2.set_ylabel("Tasa de acuerdo (%)")
    ax2.set_title("Tasa de Acuerdo entre Redes por Clase")
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax2.axhline(y=np.mean(rates), color="black", linestyle="--",
                linewidth=1, label=f"Media: {np.mean(rates):.1f}%")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 115)
    for bar, val in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    # Panel 3: Radar chart
    ax3 = fig.add_subplot(1, 3, 3, projection="polar")
    angles = np.linspace(0, 2 * np.pi, len(CLASS_NAMES),
                         endpoint=False).tolist()
    angles += angles[:1]

    clf_vals = [clf_f1[c] for c in CLASS_NAMES] + [clf_f1[CLASS_NAMES[0]]]
    det_vals = [det_ap[c] for c in CLASS_NAMES] + [det_ap[CLASS_NAMES[0]]]

    ax3.plot(angles, clf_vals, "o-", linewidth=2,
             color="#3498DB", label="F1 Clasificador")
    ax3.fill(angles, clf_vals, alpha=0.15, color="#3498DB")
    ax3.plot(angles, det_vals, "s-", linewidth=2,
             color="#E74C3C", label="AP@50 Detector")
    ax3.fill(angles, det_vals, alpha=0.15, color="#E74C3C")
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.set_title("Radar de Rendimiento por Clase", pad=20)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "5_resumen_comparativo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Resumen comparativo guardado en {out}")


# ─────────────────────────────────────────────
# 6. EXPORTAR CSV
# ─────────────────────────────────────────────

def export_csvs(cross_results: list[dict], clf_report: dict):
    """
    Exporta los resultados de validación a archivos CSV para
    referencia y uso en la memoria del proyecto.
    """
    # CSV validación cruzada completa
    csv_cross = RESULTS_DIR / "validacion_cruzada.csv"
    with open(csv_cross, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cross_results[0].keys())
        writer.writeheader()
        writer.writerows(cross_results)
    print(f"  ✔  CSV validación cruzada guardado en {csv_cross}")

    # CSV reporte final consolidado
    csv_report = RESULTS_DIR / "reporte_final.csv"
    det_ap = {
        "crazing": 0.449, "inclusion": 0.804, "patches": 0.903,
        "pitted_surface": 0.790, "rolled-in_scale": 0.561, "scratches": 0.766
    }
    with open(csv_report, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Clase", "CLF F1", "CLF Precision",
                         "CLF Recall", "DET AP@50"])
        for cls in CLASS_NAMES:
            r = clf_report[cls]
            writer.writerow([
                cls,
                f"{r['f1-score']:.4f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{det_ap[cls]:.4f}"
            ])
    print(f"  ✔  Reporte final guardado en {csv_report}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  NEU Surface Defect — Validación Integral del Sistema")
    print("=" * 60)
    print(f"\n  Dispositivo : {DEVICE}")

    # Cargar modelos y datos
    print("\n[0/4] Cargando modelos y datos")
    clf_model  = load_classifier()
    dataloader = load_val_dataloader()
    img_paths  = get_val_image_paths()
    print(f"  ✔  {len(img_paths)} imágenes de validación encontradas")

    # Sección 1: Validación del clasificador
    print("\n[1/4] Validación del clasificador")
    clf_results = validate_classifier(clf_model, dataloader)
    clf_report  = plot_classifier_results(clf_results)

    # Sección 2: Validación del detector
    print("\n[2/4] Validación del detector")
    det_results = validate_detector(img_paths)
    plot_detector_results(det_results, img_paths)

    # Sección 3: Validación cruzada
    print("\n[3/4] Validación cruzada entre redes")
    cross_results = cross_validate(clf_model, det_results, img_paths)
    plot_cross_validation(cross_results)
    plot_discrepancies(cross_results, img_paths)

    # Sección 4: Resumen comparativo y exportación
    print("\n[4/4] Resumen comparativo y exportación")
    plot_comparative_summary(clf_report, cross_results)
    export_csvs(cross_results, clf_report)

    print(f"\n{'=' * 60}")
    print("  Validación completada.")
    print("  Archivos generados en results/validation/:")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"    {f.name}")
    print(f"\n  Continúe con: python 5_aplicacion.py")
    print(f"{'=' * 60}")
