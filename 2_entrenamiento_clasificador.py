"""
2_entrenamiento_clasificador.py
--------------------------------
Entrenamiento del clasificador de defectos superficiales en acero laminado
mediante fine-tuning de ResNet-50 preentrenado sobre ImageNet.

El entrenamiento se divide en dos fases:
    Fase 1 (Feature Extraction): El encoder permanece congelado. Solo se
        entrena la cabeza de clasificación personalizada. Permite que la
        nueva cabeza converja antes de modificar las representaciones
        aprendidas por el encoder.
    Fase 2 (Fine-tuning): Se descongelan las capas profundas del encoder
        y se continúa el entrenamiento con una tasa de aprendizaje reducida,
        permitiendo que las representaciones de alto nivel se adapten al
        dominio de defectos industriales.

El número de épocas en ambas fases es determinado automáticamente mediante
early stopping, deteniendo el entrenamiento cuando la pérdida de validación
no mejora durante un número definido de épocas consecutivas (patience).

Previo al entrenamiento final se realiza un estudio de hiperparámetros
sobre combinaciones de learning rate y dropout, seleccionando la
configuración de mayor accuracy de validación.

Salidas:
    models/classifier/resnet50_neu.pth      — Pesos del mejor modelo
    results/classifier/curvas_entrenamiento.png
    results/classifier/matriz_confusion.png
    results/classifier/metricas_por_clase.csv
    results/classifier/estudio_hiperparametros.csv

Uso:
    python 2_entrenamiento_clasificador.py
"""

import os
import csv
import time
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report
from copy import deepcopy


# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────

# Raíz del proyecto: carpeta donde se encuentra este script
PROJECT_ROOT  = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
MODEL_DIR     = PROJECT_ROOT / "models/classifier"
RESULTS_DIR   = PROJECT_ROOT / "results/classifier"

MODEL_PATH      = MODEL_DIR / "resnet50_neu.pth"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES     = 6
BATCH_SIZE      = 32
EARLY_STOP_PAT  = 7        # Épocas sin mejora antes de detener

# Clases en el mismo orden que ImageFolder (alfabético)
CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

# Combinaciones a evaluar en el estudio de hiperparámetros
HYPERPARAM_GRID = [
    {"lr": 1e-3, "dropout": 0.3},
    {"lr": 1e-3, "dropout": 0.5},
    {"lr": 5e-4, "dropout": 0.3},
    {"lr": 5e-4, "dropout": 0.5},
]

# Hiperparámetros para fine-tuning (Fase 2)
LR_FINETUNE = 1e-5

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. TRANSFORMACIONES Y DATASET
# ─────────────────────────────────────────────

# Media y desviación estándar de ImageNet para normalización
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms():
    """
    Define las transformaciones de preprocesamiento para cada split.

    Entrenamiento: incluye aumentación de datos (flip, rotación, variación
        de color) para mejorar la generalización del modelo.
    Validación: solo redimensión y normalización, sin aumentación, para
        obtener métricas representativas del rendimiento real.

    Retorna:
        Diccionario con transformaciones para 'train' y 'val'.
    """
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {"train": train_tf, "val": val_tf}


def load_dataloaders(transforms: dict) -> dict:
    """
    Carga los datasets de entrenamiento y validación usando ImageFolder,
    que asigna etiquetas automáticamente según la estructura de carpetas.

    Parámetros:
        transforms: Diccionario con transformaciones por split.

    Retorna:
        Diccionario con DataLoaders para 'train' y 'val'.
    """
    datasets = {
        split: ImageFolder(
            root=PROCESSED_DIR / "images" / split,
            transform=transforms[split]
        )
        for split in ["train", "val"]
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        ),
    }

    print(f"  Train : {len(datasets['train'])} imágenes")
    print(f"  Val   : {len(datasets['val'])} imágenes")
    return dataloaders


# ─────────────────────────────────────────────
# 2. DEFINICIÓN DEL MODELO
# ─────────────────────────────────────────────

def build_model(dropout: float = 0.5) -> nn.Module:
    """
    Construye el modelo ResNet-50 con cabeza de clasificación personalizada.

    Se reemplaza la capa fully connected original (1000 clases, ImageNet)
    por una secuencia: GlobalAveragePooling → Dense(256) → Dropout → Dense(6).

    Parámetros:
        dropout: Tasa de dropout para regularización en la cabeza.

    Retorna:
        Modelo con cabeza personalizada, listo para entrenamiento Fase 1.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Congelar todo el encoder para Fase 1
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar cabeza de clasificación
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, NUM_CLASSES)
    )

    return model.to(DEVICE)


def unfreeze_top_layers(model: nn.Module, num_layers: int = 2):
    """
    Descongela las últimas capas del encoder ResNet-50 para Fase 2.
    ResNet-50 tiene 4 bloques residuales (layer1-layer4); se descongelan
    los últimos num_layers bloques para adaptar representaciones de alto
    nivel al dominio de defectos industriales.

    Parámetros:
        model:      Modelo ResNet-50.
        num_layers: Número de bloques residuales a descongelar desde el final.
    """
    layers_to_unfreeze = [f"layer{4 - i}" for i in range(num_layers)]
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_unfreeze):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros entrenables: {trainable:,}")


# ─────────────────────────────────────────────
# 3. BUCLE DE ENTRENAMIENTO CON EARLY STOPPING
# ─────────────────────────────────────────────

def train_model(
    model: nn.Module,
    dataloaders: dict,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    phase_name: str,
    patience: int = EARLY_STOP_PAT
) -> tuple[nn.Module, dict]:
    """
    Ejecuta el bucle de entrenamiento con early stopping.

    El entrenamiento se detiene automáticamente cuando la pérdida de
    validación no mejora durante 'patience' épocas consecutivas, evitando
    sobreajuste y determinando el número de épocas de forma objetiva.

    Parámetros:
        model:       Modelo a entrenar.
        dataloaders: DataLoaders de train y val.
        optimizer:   Optimizador configurado.
        criterion:   Función de pérdida.
        phase_name:  Nombre de la fase para logging.
        patience:    Épocas sin mejora antes de detener.

    Retorna:
        Tupla (mejor modelo, historial de métricas).
    """
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss  = float("inf")
    best_model = deepcopy(model.state_dict())
    no_improve = 0

    epoch = 0
    print(f"\n  {'Época':<8} {'Loss Train':>12} {'Acc Train':>10} {'Loss Val':>10} {'Acc Val':>9}")
    print(f"  {'─'*8} {'─'*12} {'─'*10} {'─'*10} {'─'*9}")

    while True:
        epoch += 1
        for split in ["train", "val"]:
            model.train() if split == "train" else model.eval()

            running_loss = 0.0
            correct      = 0
            total        = 0

            with torch.set_grad_enabled(split == "train"):
                for images, labels in dataloaders[split]:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                    if split == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    _, predicted  = outputs.max(1)
                    correct      += predicted.eq(labels).sum().item()
                    total        += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc  = correct / total

            history[f"{split}_loss"].append(epoch_loss)
            history[f"{split}_acc"].append(epoch_acc)

        # Logging por época
        print(
            f"  {epoch:<8} {history['train_loss'][-1]:>12.4f} "
            f"{history['train_acc'][-1]:>10.4f} "
            f"{history['val_loss'][-1]:>10.4f} "
            f"{history['val_acc'][-1]:>9.4f}"
        )

        val_loss = history["val_loss"][-1]
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss  = val_loss
            best_model = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping en época {epoch} "
                      f"(sin mejora en {patience} épocas).")
                break

    model.load_state_dict(best_model)
    return model, history


# ─────────────────────────────────────────────
# 4. ESTUDIO DE HIPERPARÁMETROS
# ─────────────────────────────────────────────

def hyperparameter_study(dataloaders: dict) -> dict:
    """
    Evalúa combinaciones de learning rate y dropout definidas en
    HYPERPARAM_GRID. Cada combinación se entrena durante un número
    reducido de épocas (patience=3) para estimar su rendimiento
    sin incurrir en el costo computacional del entrenamiento completo.

    Parámetros:
        dataloaders: DataLoaders de train y val.

    Retorna:
        Diccionario con la mejor combinación de hiperparámetros encontrada.
    """
    print("\n[1/3] Estudio de hiperparámetros")
    print(f"  Combinaciones a evaluar: {len(HYPERPARAM_GRID)}")
    print(f"  Dispositivo: {DEVICE}\n")

    results  = []
    best_cfg = None
    best_acc = 0.0

    criterion = nn.CrossEntropyLoss()

    for i, cfg in enumerate(HYPERPARAM_GRID):
        lr      = cfg["lr"]
        dropout = cfg["dropout"]
        print(f"  ── Combinación {i+1}/{len(HYPERPARAM_GRID)}: "
              f"lr={lr}, dropout={dropout}")

        model     = build_model(dropout=dropout)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )

        # Entrenamiento rápido con patience reducido para comparación
        _, history = train_model(
            model, dataloaders, optimizer, criterion,
            phase_name=f"tune_{i}", patience=3
        )

        val_acc = max(history["val_acc"])
        results.append({"lr": lr, "dropout": dropout, "val_acc": val_acc})
        print(f"  → Val Acc máximo: {val_acc:.4f}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            best_cfg = cfg

    # Guardar resultados del estudio
    csv_path = RESULTS_DIR / "estudio_hiperparametros.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lr", "dropout", "val_acc"])
        writer.writeheader()
        writer.writerows(results)

    print(f"  ✔  Mejor configuración: lr={best_cfg['lr']}, "
          f"dropout={best_cfg['dropout']} (Val Acc={best_acc:.4f})")
    print(f"  ✔  Resultados guardados en {csv_path}")

    return best_cfg


# ─────────────────────────────────────────────
# 5. VISUALIZACIÓN DE RESULTADOS
# ─────────────────────────────────────────────

def plot_training_curves(history_p1: dict, history_p2: dict):
    """
    Genera y guarda las curvas de pérdida y accuracy para ambas fases
    de entrenamiento en una única figura.

    Parámetros:
        history_p1: Historial de métricas de la Fase 1.
        history_p2: Historial de métricas de la Fase 2.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curvas de Entrenamiento — ResNet-50", fontsize=13)

    ep1 = range(1, len(history_p1["train_loss"]) + 1)
    ep2 = range(
        len(ep1) + 1,
        len(ep1) + len(history_p2["train_loss"]) + 1
    )

    for ax, metric, ylabel in zip(
        axes,
        ["loss", "acc"],
        ["Pérdida (Cross-Entropy)", "Accuracy"]
    ):
        ax.plot(ep1, history_p1[f"train_{metric}"], "b-",  label="Train Fase 1")
        ax.plot(ep1, history_p1[f"val_{metric}"],   "b--", label="Val Fase 1")
        ax.plot(ep2, history_p2[f"train_{metric}"], "r-",  label="Train Fase 2")
        ax.plot(ep2, history_p2[f"val_{metric}"],   "r--", label="Val Fase 2")
        ax.axvline(x=len(ep1), color="gray", linestyle=":", label="Inicio Fase 2")
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "curvas_entrenamiento.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Curvas guardadas en {out}")


def plot_confusion_matrix(model: nn.Module, dataloader: DataLoader):
    """
    Genera y guarda la matriz de confusión sobre el conjunto de validación.

    Parámetros:
        model:      Modelo entrenado en modo evaluación.
        dataloader: DataLoader del conjunto de validación.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de Confusión Normalizada — Conjunto de Validación")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black",
                    fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "matriz_confusion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Matriz de confusión guardada en {out}")

    # Reporte por clase
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True
    )

    csv_path = RESULTS_DIR / "metricas_por_clase.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Clase", "Precision", "Recall", "F1-score", "Support"])
        for cls in CLASS_NAMES:
            r = report[cls]
            writer.writerow([
                cls,
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1-score']:.4f}",
                int(r["support"])
            ])

    print(f"  ✔  Métricas por clase guardadas en {csv_path}")
    print(f"\n  Accuracy global en validación: "
          f"{report['accuracy']:.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  NEU Surface Defect — Entrenamiento Clasificador ResNet-50")
    print("=" * 60)
    print(f"\n  Dispositivo : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")

    # Verificar si ya existe un modelo entrenado
    if MODEL_PATH.exists():
        print(f"\n  AVISO: Se encontró un modelo en {MODEL_PATH}")
        resp = input("  ¿Desea reentrenar? (s/n): ").strip().lower()
        if resp != "s":
            print("  Entrenamiento cancelado. Cargando modelo existente.")
            exit(0)

    # Cargar datos
    print("\n[0/3] Cargando dataset")
    transforms   = get_transforms()
    dataloaders  = load_dataloaders(transforms)
    criterion    = nn.CrossEntropyLoss()

    # Paso 1: Estudio de hiperparámetros
    best_cfg = hyperparameter_study(dataloaders)
    lr_opt       = best_cfg["lr"]
    dropout_opt  = best_cfg["dropout"]

    # Paso 2: Fase 1 — Feature Extraction
    print("\n[2/3] Fase 1 — Feature Extraction (encoder congelado)")
    model     = build_model(dropout=dropout_opt)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_opt
    )

    model, history_p1 = train_model(
        model, dataloaders, optimizer, criterion, phase_name="fase1"
    )

    # Paso 3: Fase 2 — Fine-tuning
    print("\n[3/3] Fase 2 — Fine-tuning (descongelando capas profundas)")
    unfreeze_top_layers(model, num_layers=2)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINETUNE
    )

    model, history_p2 = train_model(
        model, dataloaders, optimizer, criterion, phase_name="fase2"
    )

    # Guardar modelo
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n  ✔  Modelo guardado en {MODEL_PATH}")

    # Visualizaciones y métricas
    print("\n  Generando visualizaciones...")
    plot_training_curves(history_p1, history_p2)
    plot_confusion_matrix(model, dataloaders["val"])

    t = time.strftime("%H:%M:%S")
    print(f"\n{'=' * 60}")
    print(f"  Entrenamiento completado a las {t}.")
    print(f"  Continúe con: python 3_entrenamiento_detector.py")
    print(f"{'=' * 60}")