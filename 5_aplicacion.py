"""
5_aplicacion.py
----------------
Interfaz de inferencia del sistema de inspección visual de defectos
en acero laminado. Implementada con Gradio, expone una interfaz web
accesible desde el navegador en http://localhost:7860

El pipeline de inferencia ejecuta secuencialmente:
    1. Clasificación  : ResNet-50 determina el tipo de defecto
    2. Detección      : YOLOv8s localiza y anota la región defectuosa
    3. Coherencia     : Se compara la predicción de ambas redes

Uso:
    python 5_aplicacion.py
    Abrir navegador en http://localhost:7860
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import gradio as gr
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from ultralytics import YOLO
from PIL import Image


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CLF    = PROJECT_ROOT / "models/classifier/resnet50_neu.pth"
MODEL_DET    = PROJECT_ROOT / "models/detector/best.pt"

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES  = 6
CONF_DEFAULT = 0.35

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

CLASS_DESCRIPTIONS = {
    "crazing":         "Red de microfisuras superficiales distribuidas uniformemente.",
    "inclusion":       "Partículas extrañas incrustadas durante el proceso de laminado.",
    "patches":         "Zonas de textura irregular o manchas en la superficie.",
    "pitted_surface":  "Cavidades o picaduras puntales en la superficie del acero.",
    "rolled-in_scale": "Óxido o cascarilla incrustada durante el laminado en caliente.",
    "scratches":       "Rayaduras lineales provocadas por fricción mecánica.",
}

COLORS = {
    "crazing":         "#E74C3C",
    "inclusion":       "#3498DB",
    "patches":         "#2ECC71",
    "pitted_surface":  "#F39C12",
    "rolled-in_scale": "#9B59B6",
    "scratches":       "#1ABC9C",
}


# ─────────────────────────────────────────────
# CARGA DE MODELOS (una sola vez al iniciar)
# ─────────────────────────────────────────────

def load_classifier() -> nn.Module:
    """
    Carga el clasificador ResNet-50 desde el archivo de pesos entrenado.
    El modelo se coloca en modo evaluación para desactivar dropout
    y garantizar inferencia determinista.
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
    return model


print("Cargando modelos...")
clf_model = load_classifier()
det_model = YOLO(str(MODEL_DET))
print(f"  ✔  Clasificador cargado ({DEVICE})")
print(f"  ✔  Detector cargado")

# Transformación de preprocesamiento para el clasificador
clf_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# FUNCIONES DE INFERENCIA
# ─────────────────────────────────────────────

def run_classifier(img_pil: Image.Image) -> tuple[str, float, list[float]]:
    """
    Ejecuta el clasificador ResNet-50 sobre la imagen de entrada.

    Parámetros:
        img_pil: Imagen PIL en formato RGB.

    Retorna:
        Tupla con (clase predicha, confianza, lista de probabilidades por clase).
    """
    img_t = clf_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = clf_model(img_t)
        probs   = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    pred_idx  = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs.tolist()


def run_detector(img_pil: Image.Image, conf_threshold: float) -> tuple[np.ndarray, list, bool]:
    """
    Ejecuta el detector YOLOv8s sobre la imagen de entrada y dibuja
    los bounding boxes con etiquetas y confianzas sobre la imagen.

    Parámetros:
        img_pil:        Imagen PIL en formato RGB.
        conf_threshold: Umbral mínimo de confianza para aceptar detecciones.

    Retorna:
        Tupla con (imagen anotada en numpy RGB, lista de detecciones, flag de detección).
    """
    results     = det_model.predict(img_pil, conf=conf_threshold, verbose=False)
    boxes       = results[0].boxes
    img_np      = np.array(img_pil)
    detections  = []
    detected    = False

    if boxes is not None and len(boxes) > 0:
        detected = True
        # Dibujar cada bounding box manualmente para mayor control visual
        for box in boxes:
            cls_idx    = int(box.cls.item())
            conf       = float(box.conf.item())
            cls_name   = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else "unknown"
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Color del bounding box según la clase
            hex_color  = COLORS.get(cls_name, "#FFFFFF")
            r, g, b    = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            color_bgr  = (b, g, r)

            # Dibujar rectángulo y etiqueta
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color_bgr, 2)
            label      = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img_np, (x1, y1 - th - 8), (x1 + tw + 4, y1), color_bgr, -1)
            cv2.putText(img_np, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            detections.append({
                "class": cls_name, "conf": conf,
                "bbox": [x1, y1, x2, y2]
            })

    return img_np, detections, detected


def build_confidence_chart(probs: list[float], pred_class: str) -> plt.Figure:
    """
    Genera una figura de barras horizontales con la probabilidad
    de cada clase según el clasificador. La clase predicha se resalta.

    Parámetros:
        probs:      Lista de probabilidades por clase (softmax).
        pred_class: Clase con mayor probabilidad.

    Retorna:
        Figura de matplotlib con el gráfico de confianza.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bar_colors = [
        COLORS.get(c, "#888888") if c == pred_class else "#444466"
        for c in CLASS_NAMES
    ]
    bars = ax.barh(CLASS_NAMES, probs, color=bar_colors, height=0.6)

    for bar, prob, cls in zip(bars, probs, CLASS_NAMES):
        ax.text(min(prob + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", fontsize=9,
                color="white" if cls == pred_class else "#aaaaaa",
                fontweight="bold" if cls == pred_class else "normal")

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Probabilidad", color="white", fontsize=9)
    ax.set_title("Distribución de probabilidad por clase",
                 color="white", fontsize=10, pad=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#444466")
    ax.xaxis.set_visible(False)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def inspect_image(image: np.ndarray, conf_threshold: float):
    """
    Pipeline completo de inspección: clasificación → detección → coherencia.
    Función principal llamada por Gradio en cada inferencia.

    Parámetros:
        image:          Imagen en formato numpy RGB (entrada de Gradio).
        conf_threshold: Umbral de confianza para el detector (slider).

    Yields:
        Actualizaciones progresivas de los componentes de la interfaz,
        simulando la ejecución secuencial de cada etapa del pipeline.
    """
    if image is None:
        yield (
            gr.update(value="⚠️ Por favor suba una imagen."),
            None, None, None, gr.update(value="")
        )
        return

    img_pil = Image.fromarray(image).convert("RGB")

    # ── Etapa 1: Clasificación ──────────────────
    yield (
        gr.update(value="🔍 **Etapa 1 — Clasificando defecto...**"),
        None, None, None,
        gr.update(value="")
    )

    pred_class, confidence, probs = run_classifier(img_pil)
    conf_chart = build_confidence_chart(probs, pred_class)
    clf_color  = COLORS.get(pred_class, "#888888")

    clf_text = (
        f"### 🏷️ Clasificación\n"
        f"**Clase detectada:** `{pred_class}`\n\n"
        f"**Confianza:** `{confidence*100:.2f}%`\n\n"
        f"*{CLASS_DESCRIPTIONS.get(pred_class, '')}*"
    )

    yield (
        gr.update(value="⚙️ **Etapa 2 — Localizando región defectuosa...**"),
        conf_chart,
        None,
        clf_text,
        gr.update(value="")
    )

    # ── Etapa 2: Detección ──────────────────────
    annotated_np, detections, detected = run_detector(img_pil, conf_threshold)

    # ── Etapa 3: Coherencia ─────────────────────
    if not detected:
        status_icon = "⚠️"
        status_msg  = "El clasificador identificó el defecto pero el detector no localizó la región."
        status_color = "orange"
    else:
        det_classes = [d["class"] for d in detections]
        if pred_class in det_classes:
            status_icon  = "✅"
            status_msg   = "Ambas redes coinciden. Detección confirmada."
            status_color = "green"
        else:
            status_icon  = "❌"
            status_msg   = (f"Discrepancia: clasificador predice `{pred_class}` "
                           f"pero detector predice `{det_classes[0]}`.")
            status_color = "red"

    # Construir texto de detección
    if detected:
        det_lines = "\n".join([
            f"- **{d['class']}** — Confianza: `{d['conf']*100:.2f}%` "
            f"| BBox: `{d['bbox']}`"
            for d in detections
        ])
        det_text = (
            f"### 📍 Detección\n"
            f"**Regiones encontradas:** {len(detections)}\n\n"
            f"{det_lines}"
        )
    else:
        det_text = "### 📍 Detección\n**Sin regiones localizadas** con el umbral actual."

    coherence_text = (
        f"### {status_icon} Coherencia entre redes\n"
        f"{status_msg}"
    )

    full_result = f"{clf_text}\n\n---\n\n{det_text}\n\n---\n\n{coherence_text}"

    yield (
        gr.update(value="✅ **Análisis completado.**"),
        conf_chart,
        annotated_np,
        full_result,
        gr.update(value="")
    )


# ─────────────────────────────────────────────
# INTERFAZ GRADIO
# ─────────────────────────────────────────────

CSS = """
#title { text-align: center; font-size: 1.6em; font-weight: bold;
         color: #3498DB; margin-bottom: 4px; }
#subtitle { text-align: center; color: #aaaaaa; margin-bottom: 16px; }
#status_box { font-size: 1.05em; padding: 8px; border-radius: 6px; }
.gr-box { border-radius: 10px; }
"""

with gr.Blocks(css=CSS, title="NEU Defect Inspector") as app:

    gr.Markdown("# 🔬 NEU Surface Defect Inspector", elem_id="title")
    gr.Markdown(
        "Sistema de inspección visual automatizada de defectos en acero laminado. "
        "Cargue una imagen para clasificar el tipo de defecto y localizar la región afectada.",
        elem_id="subtitle"
    )

    with gr.Row():
        # ── Columna izquierda: entrada ──
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Imagen de entrada",
                type="numpy",
                height=300
            )
            conf_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=CONF_DEFAULT, step=0.05,
                label=f"Umbral de confianza del detector (óptimo: {CONF_DEFAULT})"
            )
            run_btn = gr.Button("🚀 Analizar imagen", variant="primary")

            gr.Markdown("**Clases de defecto:**")
            for cls, desc in CLASS_DESCRIPTIONS.items():
                gr.Markdown(f"- **{cls}:** {desc}")

        # ── Columna derecha: resultados ──
        with gr.Column(scale=2):
            status_box = gr.Markdown(
                "Cargue una imagen y presione **Analizar imagen**.",
                elem_id="status_box"
            )

            with gr.Row():
                conf_chart   = gr.Plot(label="Probabilidad por clase (Clasificador)")
                annotated_out = gr.Image(label="Imagen anotada (Detector)", height=300)

            result_text = gr.Markdown(label="Resultado del análisis")
            dummy       = gr.Textbox(visible=False)

    # Ejemplos de uso
    gr.Examples(
        examples=[],
        inputs=[input_image],
        label="Ejemplos"
    )

    # Conexión del botón con el pipeline
    run_btn.click(
        fn=inspect_image,
        inputs=[input_image, conf_slider],
        outputs=[status_box, conf_chart, annotated_out, result_text, dummy],
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  NEU Surface Defect Inspector — Iniciando aplicación")
    print("=" * 55)
    print(f"  Dispositivo : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print("\n  Abra su navegador en: http://localhost:7860")
    print("  Presione Ctrl+C para detener la aplicación.")
    print("=" * 55 + "\n")

    app.launch(
        server_name="localhost",
        server_port=7860,
        share=False,        # True para generar enlace público temporal
        inbrowser=True,     # Abre el navegador automáticamente
    )
