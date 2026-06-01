# reorganizar_val.py
from pathlib import Path
import shutil

VAL_DIR = Path("data/processed/images/val")

CLASES = ["crazing", "inclusion", "patches",
          "pitted_surface", "rolled-in_scale", "scratches"]

# Crear subcarpetas por clase
for clase in CLASES:
    (VAL_DIR / clase).mkdir(exist_ok=True)

# Mover cada imagen a su subcarpeta según prefijo de nombre
for img_path in VAL_DIR.glob("*.jpg"):
    for clase in CLASES:
        if img_path.stem.startswith(clase):
            shutil.move(str(img_path), str(VAL_DIR / clase / img_path.name))
            break

print("✔ Val reorganizado en subcarpetas por clase")