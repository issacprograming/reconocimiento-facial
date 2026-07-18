# ============================================================
# RESULTADO 3.10 - Identificación y clasificación facial
# Artículo: Clasificación de patrones en reconocimiento facial
# Autor: Samir
# ============================================================

import cv2
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
print("IDENTIFICACIÓN FACIAL EN TIEMPO REAL")
print("=" * 50)

# ------------------------------------------------------------
# 2. PREPARAR LOS EMBEDDINGS CONOCIDOS
# ------------------------------------------------------------
embeddings_conocidos = []
nombres_conocidos = []

for nombre, registros in identidades.items():

    for registro in registros:

        embeddings_conocidos.append(
            registro["embedding"]
        )

        nombres_conocidos.append(
            nombre
        )

print(f"Embeddings cargados : {len(embeddings_conocidos)}")
print(f"Personas conocidas  : {len(identidades)}")

# ------------------------------------------------------------
# 3. DEFINIR EL UMBRAL DE CLASIFICACIÓN
# ------------------------------------------------------------
UMBRAL = 10.0

# ------------------------------------------------------------
# 4. ABRIR LA CÁMARA
# ------------------------------------------------------------
camara = cv2.VideoCapture(0)

if not camara.isOpened():

    print("\nNo se encontró una cámara disponible.")
    print("Conecte una cámara e intente nuevamente.")
    exit()

print("\nCámara iniciada.")
print("Presione la tecla Q para salir.")

# ------------------------------------------------------------
# 5. IDENTIFICACIÓN EN TIEMPO REAL
# ------------------------------------------------------------
while True:

    ret, frame = camara.read()

    if not ret:
        break

    nombre_detectado = "Desconocido"
    distancia_minima = None

    try:

        # --------------------------------------------
        # Generar embedding del rostro detectado
        # --------------------------------------------
        resultado = DeepFace.represent(

            img_path=frame,

            model_name="Facenet",

            enforce_detection=True

        )

        embedding_actual = np.array(
            resultado[0]["embedding"]
        )

        # --------------------------------------------
        # Comparar con todos los embeddings conocidos
        # --------------------------------------------
        distancias = []

        for embedding_guardado in embeddings_conocidos:

            distancia = np.linalg.norm(

                embedding_actual -

                embedding_guardado

            )

            distancias.append(distancia)

        # Buscar la distancia mínima
        indice_minimo = np.argmin(distancias)

        distancia_minima = distancias[indice_minimo]

        # --------------------------------------------
        # Clasificar la identidad
        # --------------------------------------------
        if distancia_minima < UMBRAL:

            nombre_detectado = nombres_conocidos[
                indice_minimo
            ]

    except Exception:
        pass

    # --------------------------------------------------------
    # 6. MOSTRAR RESULTADO EN LA CÁMARA
    # --------------------------------------------------------
    color = (

        (0, 255, 0)

        if nombre_detectado != "Desconocido"

        else (0, 0, 255)

    )

    cv2.putText(

        frame,

        nombre_detectado,

        (30, 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        color,

        2

    )

    cv2.imshow(

        "Reconocimiento Facial",

        frame

    )

    # Mostrar distancia en consola
    if distancia_minima is not None:

        print(

            f"Persona: {nombre_detectado:<15}"

            f" Distancia mínima: {distancia_minima:.4f}"

        )

    # Salir con la tecla Q
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# ------------------------------------------------------------
# 7. CERRAR LA CÁMARA
# ------------------------------------------------------------
camara.release()

cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("SISTEMA FINALIZADO CORRECTAMENTE")
print("=" * 50)