# ============================================================
# RESULTADO 3.1 - Representación de datos faciales
# Artículo: Clasificación de patrones en reconocimiento facial
# Autor: Isaac
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. CARGAR LA IMAGEN
# ------------------------------------------------------------
ruta_imagen = "dataset/Ana/ana1.jpg"
imagen_bgr = cv2.imread(ruta_imagen)

# Verificar que la imagen se cargó correctamente
if imagen_bgr is None:
    print("ERROR: No se encontró la imagen en la ruta indicada.")
    exit()

# ------------------------------------------------------------
# 2. CONVERTIR DE BGR A RGB
# (OpenCV carga en BGR, pero trabajamos en RGB)
# ------------------------------------------------------------
imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

# ------------------------------------------------------------
# 3. MOSTRAR INFORMACIÓN DE LA MATRIZ
# ------------------------------------------------------------
print("=" * 50)
print("REPRESENTACIÓN DE LA IMAGEN FACIAL COMO MATRIZ")
print("=" * 50)
print(f"Tipo de dato        : {type(imagen_rgb)}")
print(f"Tipo de cada pixel  : {imagen_rgb.dtype}")
print(f"Dimensiones         : {imagen_rgb.shape}")
print(f"  → Alto  (filas)   : {imagen_rgb.shape[0]} píxeles")
print(f"  → Ancho (columnas): {imagen_rgb.shape[1]} píxeles")
print(f"  → Canales (R,G,B) : {imagen_rgb.shape[2]}")
print(f"Total de píxeles    : {imagen_rgb.shape[0] * imagen_rgb.shape[1]}")
print(f"Total de valores    : {imagen_rgb.size}")
print("=" * 50)

# ------------------------------------------------------------
# 4. MOSTRAR UNA PORCIÓN DE LA MATRIZ (esquina superior)
# ------------------------------------------------------------
print("\nPrimeros 5x5 píxeles de la imagen (valores R,G,B):")
print(imagen_rgb[:5, :5])

# ------------------------------------------------------------
# 5. VISUALIZAR LA IMAGEN Y SU REPRESENTACIÓN
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Representación de imagen facial como matriz", fontsize=14)

# Imagen original
axes[0].imshow(imagen_rgb)
axes[0].set_title("Imagen original (RGB)")
axes[0].axis("off")

# Canal Rojo
axes[1].imshow(imagen_rgb[:, :, 0], cmap="Reds")
axes[1].set_title("Canal Rojo (R)\nMatriz de valores 0-255")
axes[1].axis("off")

# Canal Verde
axes[2].imshow(imagen_rgb[:, :, 1], cmap="Greens")
axes[2].set_title("Canal Verde (G)\nMatriz de valores 0-255")
axes[2].axis("off")

# Canal Azul
axes[3].imshow(imagen_rgb[:, :, 2], cmap="Blues")
axes[3].set_title("Canal Azul (B)\nMatriz de valores 0-255")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("dataset/resultado_3_1_representacion.png", dpi=150)
plt.show()

print("\nImagen guardada en: dataset/resultado_3_1_representacion.png")