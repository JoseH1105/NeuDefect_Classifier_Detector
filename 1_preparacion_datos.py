"""
1_preparacion_datos.py
-----------------------
Descarga, extracción y preprocesamiento del dataset NEU Surface Defect Database.
Convierte las anotaciones PASCAL VOC (XML) al formato requerido por YOLOv8 (TXT),
y verifica la integridad del dataset procesado.

Flujo:
    1. Descarga del dataset desde Kaggle
    2. Extracción del ZIP
    3. Conversión de anotaciones XML → YOLO TXT
    4. Verificación de integridad
    5. Reporte del dataset procesado

Uso:
    python 1_preparacion_datos.py
"""

import os
import sys
import zipfile
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

KAGGLE_DATASET  = "kaustubhdikshit/neu-surface-defect-database"
ZIP_NAME        = "neu-surface-defect-database.zip"
RAW_DIR         = Path("data/raw")
PROCESSED_DIR   = Path("data/processed")

# Mapeo de nombres de clase a índice YOLO (orden alfabético consistente)
CLASS_MAP = {
    "crazing":        0,
    "inclusion":      1,
    "patches":        2,
    "pitted_surface": 3,
    "rolled-in_scale":4,
    "scratches":      5,
}

# Dimensiones originales de las imágenes NEU (200x200 px)
IMG_W = 200
IMG_H = 200


# ─────────────────────────────────────────────
# 1. DESCARGA DEL DATASET
# ─────────────────────────────────────────────

def download_dataset():
    """Descarga el dataset desde Kaggle si no existe ya en data/raw."""
    zip_path = RAW_DIR / ZIP_NAME

    if zip_path.exists():
        print(f"  – Dataset ya descargado en {zip_path}, se omite descarga.")
        return zip_path

    print(f"  → Descargando dataset desde Kaggle: {KAGGLE_DATASET}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ret = os.system(f'kaggle datasets download -d {KAGGLE_DATASET} -p "{RAW_DIR}"')
    if ret != 0:
        print("\n  ERROR: La descarga falló.")
        print("  Verifique que kaggle.json está configurado correctamente.")
        sys.exit(1)

    print(f"  ✔  Descarga completada: {zip_path}")
    return zip_path


# ─────────────────────────────────────────────
# 2. EXTRACCIÓN DEL ZIP
# ─────────────────────────────────────────────

def extract_dataset(zip_path: Path) -> Path:
    """Extrae el ZIP del dataset en data/raw si no fue extraído previamente."""
    extract_dir = RAW_DIR / "NEU-DET"

    if extract_dir.exists():
        print(f"  – Dataset ya extraído en {extract_dir}, se omite extracción.")
        return extract_dir

    print(f"  → Extrayendo {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)

    print(f"  ✔  Extracción completada: {extract_dir}")
    return extract_dir


# ─────────────────────────────────────────────
# 3. CONVERSIÓN XML → YOLO TXT
# ─────────────────────────────────────────────

def convert_xml_to_yolo(xml_path: Path) -> list[str]:
    """
    Convierte un archivo de anotación PASCAL VOC (XML) al formato YOLO.

    Formato YOLO por línea:
        <class_id> <x_center> <y_center> <width> <height>
    Todos los valores normalizados en [0, 1] respecto a las dimensiones de la imagen.

    Parámetros:
        xml_path: Ruta al archivo XML de anotación.

    Retorna:
        Lista de strings, una por objeto anotado en la imagen.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []

    for obj in root.findall("object"):
        # Nombre de clase en minúsculas para coincidir con CLASS_MAP
        class_name = obj.find("name").text.strip().lower()

        if class_name not in CLASS_MAP:
            print(f"  ADVERTENCIA: Clase desconocida '{class_name}' en {xml_path.name}")
            continue

        class_id = CLASS_MAP[class_name]
        bbox     = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Conversión a formato YOLO normalizado
        x_center = ((xmin + xmax) / 2) / IMG_W
        y_center = ((ymin + ymax) / 2) / IMG_H
        width    = (xmax - xmin) / IMG_W
        height   = (ymax - ymin) / IMG_H

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines


def find_image(img_src_dir: Path, stem: str) -> Path | None:
    """
    Busca una imagen por nombre (stem) en la carpeta de imágenes,
    incluyendo subcarpetas por clase si existen.

    Parámetros:
        img_src_dir: Carpeta raíz de imágenes del split.
        stem:        Nombre del archivo sin extensión.

    Retorna:
        Path a la imagen encontrada, o None si no existe.
    """
    # Buscar primero en la carpeta raíz (estructura plana)
    for ext in [".jpg", ".bmp", ".png"]:
        candidate = img_src_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # Buscar en subcarpetas por clase (estructura del dataset NEU)
    for ext in [".jpg", ".bmp", ".png"]:
        matches = list(img_src_dir.glob(f"*/{stem}{ext}"))
        if matches:
            return matches[0]

    return None


def process_split(raw_split_dir: Path, split_name: str):
    """
    Procesa un split (train o val) del dataset:
    copia imágenes y convierte anotaciones XML a formato YOLO TXT.

    Parámetros:
        raw_split_dir: Carpeta del split en el dataset original.
        split_name:    Nombre del split ('train' o 'val').
    """
    img_src_dir  = raw_split_dir / "images"
    ann_src_dir  = raw_split_dir / "annotations"
    img_dst_dir  = PROCESSED_DIR / "images" / split_name
    lbl_dst_dir  = PROCESSED_DIR / "labels" / split_name

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    xml_files = list(ann_src_dir.glob("*.xml"))
    converted = 0
    skipped   = 0

    for xml_path in xml_files:
        stem      = xml_path.stem
        img_src   = find_image(img_src_dir, stem)
        label_dst = lbl_dst_dir / f"{stem}.txt"

        # Verificar que existe la imagen correspondiente
        if img_src is None:
            print(f"  ADVERTENCIA: Imagen no encontrada para {xml_path.name}")
            skipped += 1
            continue

        # Copiar imagen al directorio procesado manteniendo extensión original
        img_dst = img_dst_dir / f"{stem}{img_src.suffix}"
        shutil.copy2(img_src, img_dst)

        # Convertir y guardar anotación en formato YOLO
        yolo_lines = convert_xml_to_yolo(xml_path)
        with open(label_dst, "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    print(f"  ✔  [{split_name}] {converted} imágenes procesadas, {skipped} omitidas.")
    return converted


# ─────────────────────────────────────────────
# 4. VERIFICACIÓN DE INTEGRIDAD
# ─────────────────────────────────────────────

def verify_dataset():
    """
    Verifica que cada imagen en el directorio procesado tiene su archivo
    de etiqueta correspondiente y que ningún archivo de etiqueta está vacío.
    Reporta cualquier inconsistencia encontrada.
    """
    print("\n  Verificando integridad del dataset procesado...")
    issues = 0

    for split in ["train", "val"]:
        img_dir = PROCESSED_DIR / "images" / split
        lbl_dir = PROCESSED_DIR / "labels" / split

        images = list(img_dir.glob("*.jpg"))
        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            if not lbl_path.exists():
                print(f"  ERROR: Sin etiqueta para {img_path.name} [{split}]")
                issues += 1
            elif lbl_path.stat().st_size == 0:
                print(f"  ADVERTENCIA: Etiqueta vacía para {img_path.name} [{split}]")
                issues += 1

    if issues == 0:
        print("  ✔  Sin problemas de integridad detectados.")
    else:
        print(f"  ⚠  Se encontraron {issues} problema(s) de integridad.")


# ─────────────────────────────────────────────
# 5. REPORTE DEL DATASET
# ─────────────────────────────────────────────

def print_report():
    """Imprime un resumen del dataset procesado: conteo por split y por clase."""
    print("\n  ── Reporte del dataset procesado ──")

    for split in ["train", "val"]:
        lbl_dir    = PROCESSED_DIR / "labels" / split
        class_count = {name: 0 for name in CLASS_MAP}
        total       = 0

        for lbl_path in lbl_dir.glob("*.txt"):
            with open(lbl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        class_name = list(CLASS_MAP.keys())[class_id]
                        class_count[class_name] += 1
                        total += 1

        print(f"\n  [{split}] — {total} anotaciones totales")
        for name, count in class_count.items():
            bar = "█" * (count // 10)
            print(f"    {name:<20} {count:>4}  {bar}")

    img_train = len(list((PROCESSED_DIR / "images" / "train").glob("*.jpg")))
    img_val   = len(list((PROCESSED_DIR / "images" / "val").glob("*.jpg")))
    print(f"\n  Imágenes totales — train: {img_train} | val: {img_val}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  NEU Surface Defect — Preparación de Datos")
    print("=" * 55)

    # Paso 1: Descarga
    print("\n[1/4] Descarga del dataset")
    zip_path = download_dataset()

    # Paso 2: Extracción
    print("\n[2/4] Extracción del dataset")
    extract_dir = extract_dataset(zip_path)

    # Paso 3: Conversión de anotaciones por split
    print("\n[3/4] Conversión de anotaciones XML → YOLO")

    # El dataset NEU en Kaggle usa 'train' y 'validation' como nombres de carpeta
    splits = {
        "train": extract_dir / "train",
        "val":   extract_dir / "valid",
    }

    total_converted = 0
    for split_name, split_path in splits.items():
        if not split_path.exists():
            # Intentar nombre alternativo 'validation'
            alt = extract_dir / "validation"
            if alt.exists():
                split_path = alt
            else:
                print(f"  ADVERTENCIA: Carpeta '{split_path}' no encontrada, se omite.")
                continue
        total_converted += process_split(split_path, split_name)

    # Paso 4: Verificación e integridad
    print("\n[4/4] Verificación de integridad")
    verify_dataset()

    # Reporte final
    print_report()

    print("\n" + "=" * 55)
    print(f"  Preparación completada. {total_converted} imágenes procesadas.")
    print("  Puede continuar con: python 2_entrenamiento_clasificador.py")
    print("=" * 55)