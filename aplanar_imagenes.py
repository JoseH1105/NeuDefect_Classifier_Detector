# aplanar_imagenes.py
from pathlib import Path
import shutil

for split in ["train", "val"]:
    img_dir = Path(f"data/processed/images/{split}")
    for img in img_dir.glob("**/*.jpg"):
        if img.parent != img_dir:          # si está en subcarpeta
            shutil.move(str(img), str(img_dir / img.name))

    # Eliminar subcarpetas vacías
    for sub in img_dir.iterdir():
        if sub.is_dir():
            sub.rmdir()

    print(f"✔ {split} aplanado")