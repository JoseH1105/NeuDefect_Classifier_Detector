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
  - Windows:   `C:\Users\<usuario>\.kaggle\kaggle.json`

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
