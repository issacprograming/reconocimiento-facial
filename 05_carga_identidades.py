# ============================================================
# RESULTADO 3.5 - Carga de identidades conocidas
# Artículo: Clasificación de patrones en reconocimiento facial
# Autor: Samir
# ============================================================

import os
import numpy as np
from deepface import DeepFace

# ------------------------------------------------------------
# 1. BASE DE DATOS DE IDENTIDADES
# ------------------------------------------------------------
# Cada clave representa el nombre de una persona y su valor
# será una lista con los embeddings obtenidos de sus imágenes.
base_identidades = {}

carpeta_dataset = "dataset"

print("=" * 50)
print("CARGA DE IDENTIDADES CONOCIDAS")
print("=" * 50)

total_personas = 0
total_embeddings = 0

# ------------------------------------------------------------
# 2. RECORRER CADA PERSONA DEL DATASET
# ------------------------------------------------------------
for nombre in sorted(os.listdir(carpeta_dataset)):

    ruta_persona = os.path.join(carpeta_dataset, nombre)

    # Ignorar archivos que no sean carpetas
    if not os.path.isdir(ruta_persona):
        continue

    base_identidades[nombre] = []
    total_personas += 1

    print(f"\nPersona: {nombre}")

    # --------------------------------------------------------
    # 3. RECORRER LAS IMÁGENES DE CADA PERSONA
    # --------------------------------------------------------
    for archivo in sorted(os.listdir(ruta_persona)):

        if archivo.lower().endswith((".jpg", ".jpeg", ".png")):

            ruta_img = os.path.join(ruta_persona, archivo)

            try:

                # Generar el embedding facial
                resultado = DeepFace.represent(
                    img_path=ruta_img,
                    model_name="Facenet",
                    enforce_detection=True
                )

                embedding = np.array(resultado[0]["embedding"])

                # Guardar el embedding en la base de identidades
                base_identidades[nombre].append(embedding)

                total_embeddings += 1

                print(f"   ✓ {archivo}")

            except Exception as e:

                print(f"   ✗ Error en {archivo}: {e}")

# ------------------------------------------------------------
# 4. RESUMEN DE IDENTIDADES CARGADAS
# ------------------------------------------------------------
print("\n" + "=" * 50)
print("RESUMEN DE IDENTIDADES")
print("=" * 50)

for nombre, embeddings in base_identidades.items():
    print(f"{nombre:<12}: {len(embeddings)} embedding(s)")

print("-" * 50)
print(f"Total de personas   : {total_personas}")
print(f"Total de embeddings : {total_embeddings}")
print("=" * 50)

# ------------------------------------------------------------
# 5. GUARDAR LA BASE DE DATOS
# ------------------------------------------------------------
os.makedirs("modelos", exist_ok=True)

np.save("modelos/identidades.npy", base_identidades)

print("\nBase de datos guardada en:")
print("modelos/identidades.npy")