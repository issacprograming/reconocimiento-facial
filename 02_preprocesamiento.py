# ============================================================
# RESULTADO 3.2 - Preprocesamiento y detección del rostro
# Artículo: Clasificación de patrones en reconocimiento facial
# Autor: Isaac
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace

# ------------------------------------------------------------
# 1. CARGAR LA IMAGEN (misma que en 3.1)
# ------------------------------------------------------------
ruta_imagen = "dataset/Ana/ana1.jpg"
imagen_bgr = cv2.imread(ruta_imagen)

if imagen_bgr is None:
    print("ERROR: No se encontró la imagen.")
    exit()

imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

# ------------------------------------------------------------
# 2. DETECTAR EL ROSTRO CON DEEPFACE
# ------------------------------------------------------------
print("=" * 50)
print("PREPROCESAMIENTO Y DETECCIÓN DEL ROSTRO")
print("=" * 50)

resultado = DeepFace.extract_faces(
    img_path=ruta_imagen,
    detector_backend="opencv",
    enforce_detection=True
)

# Obtener coordenadas del rostro detectado
facial_area = resultado[0]["facial_area"]
top    = facial_area["y"]
left   = facial_area["x"]
bottom = facial_area["y"] + facial_area["h"]
right  = facial_area["x"] + facial_area["w"]

print(f"Rostro detectado en coordenadas:")
print(f"  top    : {top}")
print(f"  left   : {left}")
print(f"  bottom : {bottom}")
print(f"  right  : {right}")
print(f"  ancho  : {facial_area['w']} píxeles")
print(f"  alto   : {facial_area['h']} píxeles")

# ------------------------------------------------------------
# 3. RECORTAR EL ROSTRO
# ------------------------------------------------------------
rostro_recortado = imagen_rgb[top:bottom, left:right]

print(f"\nDimensiones imagen original : {imagen_rgb.shape}")
print(f"Dimensiones rostro recortado: {rostro_recortado.shape}")
print("=" * 50)

# ------------------------------------------------------------
# 4. DIBUJAR EL RECTÁNGULO EN LA IMAGEN ORIGINAL
# ------------------------------------------------------------
imagen_con_rectangulo = imagen_rgb.copy()
cv2.rectangle(
    imagen_con_rectangulo,
    (left, top),
    (right, bottom),
    (0, 255, 0),  # color verde
    3             # grosor
)

# ------------------------------------------------------------
# 5. VISUALIZAR RESULTADO
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Preprocesamiento: detección y recorte del rostro", fontsize=14)

axes[0].imshow(imagen_con_rectangulo)
axes[0].set_title("Imagen original\n(rostro detectado en verde)")
axes[0].axis("off")

axes[1].imshow(rostro_recortado)
axes[1].set_title(f"Rostro recortado\n{rostro_recortado.shape[1]}x{rostro_recortado.shape[0]} píxeles")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("dataset/resultado_3_2_preprocesamiento.png", dpi=150)
plt.show(block=True)

print("\nImagen guardada en: dataset/resultado_3_2_preprocesamiento.png")