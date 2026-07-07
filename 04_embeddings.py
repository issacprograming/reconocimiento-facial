# ============================================================
# RESULTADO 3.4 - Extracción de características: embedding facial
# Autor: Hans
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace

ruta_imagen = "dataset/Ana/ana1.jpg"

resultado = DeepFace.represent(
    img_path=ruta_imagen,
    model_name="Facenet",
    enforce_detection=True
)
embedding = np.array(resultado[0]['embedding'])

print("=" * 50)
print("EMBEDDING FACIAL GENERADO")
print("=" * 50)
print(f"Dimensión del embedding : {embedding.shape}")
print(f"Primeros 10 valores : {embedding[:10].round(4)}")
print(f"Valor mínimo : {embedding.min():.4f}")
print(f"Valor máximo : {embedding.max():.4f}")
print("=" * 50)

plt.figure(figsize=(12, 4))
plt.plot(embedding, color='darkblue', linewidth=0.8)
plt.title("Embedding facial — Vector de 128 características")
plt.xlabel("Índice de característica")
plt.ylabel("Valor")
plt.grid(True, alpha=0.3)
plt.savefig("dataset/resultado_3_4_embedding.png", dpi=150)
print("\nImagen guardada en: dataset/resultado_3_4_embedding.png")
