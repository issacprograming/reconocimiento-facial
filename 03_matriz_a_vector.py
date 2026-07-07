# ============================================================
# RESULTADO 3.3 - Transformación de matriz facial a vector
# Autor: Hans
# ============================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace

# 1. Cargar imagen y detectar/recortar el rostro (igual que Isaac en 3.2)
ruta_imagen = "dataset/Ana/ana1.jpg"
imagen_bgr = cv2.imread(ruta_imagen)
imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

resultado_deteccion = DeepFace.extract_faces(
    img_path=ruta_imagen,
    detector_backend="opencv",
    enforce_detection=True
)
area = resultado_deteccion[0]["facial_area"]
top, left = area["y"], area["x"]
bottom, right = top + area["h"], left + area["w"]
rostro_recortado_rgb = imagen_rgb[top:bottom, left:right]
rostro_recortado_bgr = imagen_bgr[top:bottom, left:right]

# 2. Convertir el rostro recortado a escala de grises
rostro_gris = cv2.cvtColor(rostro_recortado_bgr, cv2.COLOR_BGR2GRAY)

# 3. Aplanar la matriz en un vector
vector = rostro_gris.flatten()

# 4. Mostrar información
print("=" * 50)
print("TRANSFORMACIÓN MATRIZ → VECTOR")
print("=" * 50)
print(f"Dimensiones rostro (color)  : {rostro_recortado_rgb.shape}")
print(f"Dimensiones rostro (gris)   : {rostro_gris.shape}")
print(f"Longitud del vector         : {vector.shape[0]} elementos")
print(f"Primeros 10 valores         : {vector[:10]}")
print("=" * 50)

# 5. Visualizar
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Transformación: Matriz facial → Vector de datos", fontsize=14)
axes[0].imshow(rostro_gris, cmap="gray")
axes[0].set_title(f"Matriz en escala de grises\n{rostro_gris.shape[0]}x{rostro_gris.shape[1]}")
axes[0].axis("off")
axes[1].plot(vector[:500], color='blue', linewidth=0.5)
axes[1].set_title(f"Vector de datos (primeros 500 valores)\nTotal: {vector.shape[0]} elementos")
axes[1].set_xlabel("Índice")
axes[1].set_ylabel("Valor del píxel (0-255)")
plt.tight_layout()
plt.savefig("dataset/resultado_3_3_vector.png", dpi=150)
print("\nImagen guardada en: dataset/resultado_3_3_vector.png")
