"""
setup_project.py
----------------
Script de inicialización del proyecto NEU Surface Defect Detection.
Crea la estructura de carpetas, verifica el entorno y muestra instrucciones
para la configuración del entorno Conda y la descarga del dataset.

Uso:
    python setup_project.py
"""

import os
import sys
import platform


# ─────────────────────────────────────────────
# 1. ESTRUCTURA DE CARPETAS
# ─────────────────────────────────────────────

FOLDERS = [
    "data/raw",
    "data/processed/images/train",
    "data/processed/images/val",
    "data/processed/labels/train",
    "data/processed/labels/val",
    "models/classifier",
    "models/detector",
    "results/classifier",
    "results/detector",
    "results/validation",
]

def create_structure():
    print("\n[1/3] Creando estructura de carpetas...")
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"  ✔  {folder}")

    # Archivo .gitkeep para mantener carpetas vacías en Git
    for folder in FOLDERS:
        gitkeep = os.path.join(folder, ".gitkeep")
        if not os.path.exists(gitkeep):
            open(gitkeep, "w").close()

    print("  → Estructura creada correctamente.\n")


# ─────────────────────────────────────────────
# 2. GENERACIÓN DE ARCHIVOS DE CONFIGURACIÓN
# ─────────────────────────────────────────────

DATASET_YAML = """\
# dataset.yaml
# Configuración del dataset NEU Surface Defect para YOLOv8
# Generado automáticamente por setup_project.py

path: data/processed        # Ruta raíz del dataset procesado
train: images/train
val: images/val

nc: 6                        # Número de clases

names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""

REQUIREMENTS = """\
# requirements.txt
# Dependencias del proyecto NEU Surface Defect Detection

torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
albumentations>=1.3.0
scikit-learn>=1.3.0
gradio>=4.0.0
matplotlib>=3.7.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
kaggle>=1.5.16
PyYAML>=6.0
"""

README = """\
# NEU Surface Defect Detection
## Sistema de Inspección Visual Automatizada

Proyecto de detección de defectos en acero laminado usando el dataset
NEU Surface Defect Database. Implementa un pipeline de dos etapas:
clasificación con ResNet-50 y detección con YOLOv8.

---

## Configuración del entorno

### 1. Crear entorno Conda
```bash
conda create -n neu_defect python=3.10
conda activate neu_defect
```

### 2. Instalar PyTorch con soporte GPU (CUDA 11.8)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

> Si no dispone de GPU NVIDIA compatible, instale la versión CPU:
> `conda install pytorch torchvision torchaudio cpuonly -c pytorch`

### 3. Instalar dependencias del proyecto
```bash
pip install -r requirements.txt
```

### 4. Configurar Kaggle API (para descarga automática del dataset)
- Ingresar a https://www.kaggle.com/settings → API → Create New Token
- Descargar el archivo `kaggle.json`
- Colocarlo en:
  - Linux/Mac: `~/.kaggle/kaggle.json`
  - Windows:   `C:\\Users\\<usuario>\\.kaggle\\kaggle.json`

---

## Flujo de ejecución

| Script                              | Descripción                            | Ejecutar       |
|-------------------------------------|----------------------------------------|----------------|
| `setup_project.py`                  | Inicialización del proyecto            | Una vez        |
| `1_preparacion_datos.py`            | Descarga y preprocesamiento            | Una vez        |
| `2_entrenamiento_clasificador.py`   | Fine-tuning ResNet-50                  | Una vez        |
| `3_entrenamiento_detector.py`       | Fine-tuning YOLOv8                     | Una vez        |
| `4_validacion.py`                   | Métricas y validación cruzada          | N veces        |
| `5_aplicacion.py`                   | Interfaz de inferencia al usuario      | N veces        |

---

## Modelos generados

| Modelo       | Ruta                          |
|--------------|-------------------------------|
| Clasificador | `models/classifier/resnet50_neu.pth` |
| Detector     | `models/detector/best.pt`     |

---

## Estructura del proyecto

```
neu_defect_project/
├── data/
│   ├── raw/                    ← Dataset descargado de Kaggle
│   └── processed/              ← Datos en formato YOLO
│       ├── images/train|val/
│       └── labels/train|val/
├── models/
│   ├── classifier/             ← resnet50_neu.pth
│   └── detector/               ← best.pt
├── results/
│   ├── classifier/             ← Curvas, métricas
│   ├── detector/               ← mAP, curvas PR
│   └── validation/             ← Validación cruzada
├── dataset.yaml
├── requirements.txt
├── README.md
└── setup_project.py
    1_preparacion_datos.py
    2_entrenamiento_clasificador.py
    3_entrenamiento_detector.py
    4_validacion.py
    5_aplicacion.py
```
"""

def create_config_files():
    print("[2/3] Generando archivos de configuración...")

    files = {
        "dataset.yaml": DATASET_YAML,
        "requirements.txt": REQUIREMENTS,
        "README.md": README,
    }

    for filename, content in files.items():
        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✔  {filename}")
        else:
            print(f"  –  {filename} ya existe, se omite.")

    print("  → Archivos generados correctamente.\n")


# ─────────────────────────────────────────────
# 3. VERIFICACIÓN DEL ENTORNO
# ─────────────────────────────────────────────

def check_environment():
    print("[3/3] Verificando entorno de ejecución...")
    print(f"  Sistema operativo : {platform.system()} {platform.release()}")
    print(f"  Python            : {sys.version.split()[0]}")

    # Verificar PyTorch y GPU
    try:
        import torch
        print(f"  PyTorch           : {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU disponible    : {gpu_name} ({vram:.1f} GB VRAM)")
            print(f"  CUDA              : {torch.version.cuda}")
        else:
            print("  GPU disponible    : No (se usará CPU)")
    except ImportError:
        print("  PyTorch           : NO instalado — ejecute los pasos del README")

    # Verificar Ultralytics (YOLOv8)
    try:
        import ultralytics
        print(f"  Ultralytics       : {ultralytics.__version__}")
    except ImportError:
        print("  Ultralytics       : NO instalado")

    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  NEU Surface Defect Detection — Setup del Proyecto")
    print("=" * 55)

    create_structure()
    create_config_files()
    check_environment()

    print("=" * 55)
    print("  Setup completado.")
    print("  Consulte README.md para los siguientes pasos.")
    print("=" * 55)
