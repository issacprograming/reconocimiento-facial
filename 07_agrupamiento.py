# ============================================================
# RESULTADO 3.7 - Agrupamiento de patrones faciales
# Artículo: Clasificación de patrones en reconocimiento facial
# Autor: Sam
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# 1. CARGAR LA BASE DE IDENTIDADES
# ------------------------------------------------------------
identidades = np.load(
    "modelos/identidades.npy",
    allow_pickle=True
).item()

print("=" * 50)
print("AGRUPAMIENTO DE PATRONES FACIALES")
print("=" * 50)

# ------------------------------------------------------------
# 2. EXTRAER LOS EMBEDDINGS Y METADATOS
# ------------------------------------------------------------
todos_embeddings = []
todas_etiquetas = []
nombres_imagenes = []

for nombre, registros in identidades.items():

    for registro in registros:

        todos_embeddings.append(registro["embedding"])
        todas_etiquetas.append(nombre)

        # Guardar el nombre del archivo sin la extensión
        nombres_imagenes.append(
            os.path.splitext(registro["archivo"])[0]
        )

# ------------------------------------------------------------
# 3. CREAR LA MATRIZ DE EMBEDDINGS
# ------------------------------------------------------------
X = np.array(todos_embeddings)

total_embeddings = len(X)
n_personas = len(identidades)

print(f"\nPersonas registradas : {n_personas}")
print(f"Embeddings cargados  : {total_embeddings}")
print(f"Dimensión vectorial  : {X.shape[1]}")

# ------------------------------------------------------------
# 4. APLICAR K-MEANS
# ------------------------------------------------------------
print("\nAplicando algoritmo K-Means...")

kmeans = KMeans(
    n_clusters=n_personas,
    random_state=42,
    n_init=10
)

grupos = kmeans.fit_predict(X)

print("Agrupamiento completado.")

# ------------------------------------------------------------
# 5. MOSTRAR LA ASIGNACIÓN DE GRUPOS
# ------------------------------------------------------------
print("\nAsignación de grupos")
print("-" * 50)

for imagen, persona, grupo in zip(
        nombres_imagenes,
        todas_etiquetas,
        grupos):

    print(f"{imagen:<15} ({persona})  → Grupo {grupo}")

print("-" * 50)

print("\nCentroides generados:")
print(kmeans.cluster_centers_.shape)

# ------------------------------------------------------------
# 6. REDUCIR DIMENSIONES CON PCA
# ------------------------------------------------------------
print("\nReduciendo dimensiones con PCA...")

pca = PCA(n_components=2)

X_2d = pca.fit_transform(X)

print("Reducción completada.")

# ------------------------------------------------------------
# 7. VISUALIZAR EL AGRUPAMIENTO
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=grupos,
    cmap="tab10",
    s=120,
    edgecolors="black"
)

# Etiquetar cada punto con el nombre de la imagen
for i, imagen in enumerate(nombres_imagenes):

    plt.annotate(
        imagen,
        (X_2d[i, 0], X_2d[i, 1]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

plt.title("Agrupamiento de embeddings mediante K-Means (PCA 2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True, alpha=0.3)

plt.tight_layout()

# ------------------------------------------------------------
# 8. CREAR LA CARPETA DE RESULTADOS
# ------------------------------------------------------------
os.makedirs("resultados", exist_ok=True)

# ------------------------------------------------------------
# 9. GUARDAR LA FIGURA
# ------------------------------------------------------------
ruta_figura = os.path.join(
    "resultados",
    "resultado_3_7_agrupamiento.png"
)

plt.savefig(
    ruta_figura,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nFigura guardada en:")
print(ruta_figura)

print("\n" + "=" * 50)
print("AGRUPAMIENTO FINALIZADO CORRECTAMENTE")
print("=" * 50)