# ============================================================
# RESULTADO 3.6 - Cálculo de distancias entre vectores faciales
# Autor: Hans
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
from sklearn.metrics.pairwise import euclidean_distances

# 1. Generar embeddings de dos personas
ruta_ana = "dataset/Ana/ana1.jpg"
ruta_luis = "dataset/Luis/luis1.jpg"

emb_ana = np.array(DeepFace.represent(img_path=ruta_ana, model_name='Facenet')[0]['embedding'])
emb_luis = np.array(DeepFace.represent(img_path=ruta_luis, model_name='Facenet')[0]['embedding'])

# 2. Calcular distancia euclidiana
distancia = np.linalg.norm(emb_ana - emb_luis)

# 3. Mostrar resultado
print("=" * 50)
print("CÁLCULO DE DISTANCIAS ENTRE VECTORES")
print("=" * 50)
print(f"Embedding Ana — dimensión: {emb_ana.shape}")
print(f"Embedding Luis — dimensión: {emb_luis.shape}")
print(f"Distancia euclidiana entre Ana y Luis: {distancia:.4f}")
if distancia < 10:
    print("→ Resultado: MISMA PERSONA (distancia baja)")
else:
    print("→ Resultado: PERSONAS DISTINTAS (distancia alta)")
print("=" * 50)

# 4. Visualizar comparación de embeddings
plt.figure(figsize=(12, 5))
plt.plot(emb_ana, label='Ana', color='blue', linewidth=0.8)
plt.plot(emb_luis, label='Luis', color='red', linewidth=0.8)
plt.title("Comparación de embeddings faciales: Ana vs Luis")
plt.xlabel("Índice de característica")
plt.ylabel("Valor")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("dataset/resultado_3_6_distancias.png", dpi=150)
print("\nImagen guardada en: dataset/resultado_3_6_distancias.png")
