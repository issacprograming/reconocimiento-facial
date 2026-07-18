# ============================================================
# RESULTADO 3.9 - Creación de nueva identidad
# Artículo: Clasificación de patrones en reconocimiento facial
# Autor: Sam
# ============================================================

import os
import numpy as np
from deepface import DeepFace

# ------------------------------------------------------------
# 1. CARGAR LA BASE DE IDENTIDADES
# ------------------------------------------------------------
identidades = np.load(
    "modelos/identidades.npy",
    allow_pickle=True
).item()

print("=" * 50)
print("REGISTRO DE NUEVA IDENTIDAD")
print("=" * 50)

# ------------------------------------------------------------
# 2. SOLICITAR LOS DATOS DE LA NUEVA PERSONA
# ------------------------------------------------------------
nuevo_nombre = input("Ingrese el nombre de la nueva persona: ")

ruta_imagen = input(
    "Ingrese la ruta de la imagen: "
)

nombre_archivo = os.path.basename(ruta_imagen)

# ------------------------------------------------------------
# 3. GENERAR EL EMBEDDING DEL ROSTRO
# ------------------------------------------------------------
print("\nGenerando embedding facial...")

resultado = DeepFace.represent(
    img_path=ruta_imagen,
    model_name="Facenet",
    enforce_detection=True
)

nuevo_embedding = np.array(
    resultado[0]["embedding"]
)

print("Embedding generado correctamente.")

# ------------------------------------------------------------
# 4. REGISTRAR LA NUEVA IDENTIDAD
# ------------------------------------------------------------
if nuevo_nombre not in identidades:
    identidades[nuevo_nombre] = []

# Verificar si la imagen ya fue registrada
imagen_existente = any(
    registro["archivo"] == nombre_archivo
    for registro in identidades[nuevo_nombre]
)

if imagen_existente:

    print("\nLa imagen ya se encuentra registrada para esta persona.")

else:

    identidades[nuevo_nombre].append({

        "archivo": nombre_archivo,

        "embedding": nuevo_embedding

    })

    # --------------------------------------------------------
    # 5. GUARDAR LA BASE ACTUALIZADA
    # --------------------------------------------------------
    os.makedirs("modelos", exist_ok=True)

    np.save(
        "modelos/identidades.npy",
        identidades
    )

    # --------------------------------------------------------
    # 6. MOSTRAR EL RESUMEN
    # --------------------------------------------------------
    print("\n" + "=" * 50)
    print("NUEVA IDENTIDAD REGISTRADA")
    print("=" * 50)
    print(f"Persona registrada   : {nuevo_nombre}")
    print(f"Imagen utilizada     : {nombre_archivo}")
    print(f"Total de identidades : {len(identidades)}")
    print("=" * 50)
    print("La base de datos fue actualizada correctamente.")