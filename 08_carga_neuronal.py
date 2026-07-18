# ============================================================
# RESULTADO 3.8 - Carga del modelo neuronal
# Artículo: Clasificación de patrones en reconocimiento facial
# Autor: Sam
# ============================================================

from deepface import DeepFace

# ------------------------------------------------------------
# 1. CARGAR EL MODELO FACENET
# ------------------------------------------------------------
print("Cargando modelo neuronal Facenet...")

modelo = DeepFace.build_model("Facenet")

# El objeto que retorna DeepFace envuelve un modelo Keras interno
red_neuronal = modelo.model

# ------------------------------------------------------------
# 2. EXTRAER INFORMACIÓN DE LA ARQUITECTURA
# ------------------------------------------------------------
nombre_modelo = getattr(modelo, "model_name", "Facenet")
forma_entrada = red_neuronal.input_shape
forma_salida = red_neuronal.output_shape
n_capas = len(red_neuronal.layers)
n_parametros = red_neuronal.count_params()

# ------------------------------------------------------------
# 3. MOSTRAR INFORMACIÓN DEL MODELO
# ------------------------------------------------------------
print("=" * 50)
print("MODELO NEURONAL CARGADO")
print("=" * 50)
print(f"Nombre del modelo        : {nombre_modelo}")
print(f"Tipo de objeto           : {type(modelo).__name__}")
print(f"Forma de entrada (input) : {forma_entrada}")
print(f"Forma de salida (output) : {forma_salida}")
print(f"Dimensión del embedding  : {forma_salida[-1]}")
print(f"Número de capas          : {n_capas}")
print(f"Total de parámetros      : {n_parametros:,}")
print("=" * 50)
print("Modelo listo para generar embeddings faciales.")