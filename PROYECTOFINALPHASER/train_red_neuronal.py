import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- Configuración ---
NOMBRE_ARCHIVO_DATOS = 'datos_combinados_ia.csv'
CARPETA_MODELOS = 'modelos_entrenados'

# Crear carpeta de modelos si no existe
if not os.path.exists(CARPETA_MODELOS):
    os.makedirs(CARPETA_MODELOS)
    print(f"Carpeta '{CARPETA_MODELOS}' creada.")

# --- 1. Cargar Datos ---
try:
    df = pd.read_csv(NOMBRE_ARCHIVO_DATOS)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de datos '{NOMBRE_ARCHIVO_DATOS}'.")
    print("Asegúrate de haber jugado en modo manual para generar este archivo.")
    exit()

if df.empty:
    print(f"El archivo de datos '{NOMBRE_ARCHIVO_DATOS}' está vacío.")
    exit()

# --- 2. Preprocesamiento Básico ---
df.dropna(inplace=True) 
if df.empty:
    print(f"El archivo de datos quedó vacío después de eliminar NaNs.")
    exit()

print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")

# --- 3. Preparar Datos y Entrenar MODELO HORIZONTAL (Red Neuronal) ---
print("\n--- Entrenando Red Neuronal para Movimiento Horizontal ---")

features_horizontal = [
    'jugador_x_centro',
    'bala_v_activa',
    'bala_v_x_centro',
    'bala_v_y_centro',
    'dist_jugador_bala_v_x',
    'dist_jugador_bala_v_y'
]
target_horizontal = 'accion_horizontal_comandada'

try:
    X_h = df[features_horizontal]
    y_h = df[target_horizontal]
except KeyError as e:
    print(f"Error: Columnas para modelo horizontal no encontradas: {e}")
    print(f"Columnas disponibles: {df.columns.tolist()}")
    exit()

if X_h.empty or y_h.empty or len(y_h.unique()) < 2:
    print("No hay suficientes datos o clases para el modelo horizontal.")
    exit()

# División de datos
X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
    X_h, y_h, test_size=0.25, random_state=42, stratify=y_h
)

# Escalado de características (importante para redes neuronales)
scaler_h = StandardScaler()
X_h_train_scaled = scaler_h.fit_transform(X_h_train)
X_h_test_scaled = scaler_h.transform(X_h_test)

# Crear y entrenar la red neuronal
mlp_horizontal = MLPClassifier(
    hidden_layer_sizes=(50, 30),  # Dos capas ocultas con 50 y 30 neuronas
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

mlp_horizontal.fit(X_h_train_scaled, y_h_train)

# Evaluar modelo horizontal
y_h_pred = mlp_horizontal.predict(X_h_test_scaled)
accuracy_h = accuracy_score(y_h_test, y_h_pred)
print(f"Precisión del Modelo Horizontal (prueba): {accuracy_h:.4f}")
print("Reporte Clasificación Horizontal (prueba):\n", classification_report(y_h_test, y_h_pred, zero_division=0))

# Guardar modelo y scaler horizontal
path_modelo_h = os.path.join(CARPETA_MODELOS, 'red_neuronal_horizontal_model.joblib')
path_scaler_h = os.path.join(CARPETA_MODELOS, 'red_neuronal_horizontal_scaler.joblib')
joblib.dump(mlp_horizontal, path_modelo_h)
joblib.dump(scaler_h, path_scaler_h)
print(f"Modelo Horizontal guardado en: {path_modelo_h}")
print(f"Scaler Horizontal guardado en: {path_scaler_h}")

# --- 4. Preparar Datos y Entrenar MODELO DE SALTO (Red Neuronal) ---
print("\n--- Entrenando Red Neuronal para Salto ---")

features_salto = [
    'en_suelo',
    'bala_h_activa',
    'dist_jugador_bala_h_x',
    'dist_jugador_bala_h_y',
    'bala_h_velocidad_x',
    'jugador_x_centro'
]
target_salto = 'accion_salto_comandado'

try:
    X_s = df[features_salto]
    y_s = df[target_salto]
except KeyError as e:
    print(f"Error: Columnas para modelo de salto no encontradas: {e}")
    print(f"Columnas disponibles: {df.columns.tolist()}")
    exit()

if X_s.empty or y_s.empty or len(y_s.unique()) < 2:
    print("No hay suficientes datos o clases para el modelo de salto.")
    exit()

# División de datos
X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
    X_s, y_s, test_size=0.25, random_state=42, stratify=y_s
)

# Escalado de características
scaler_s = StandardScaler()
X_s_train_scaled = scaler_s.fit_transform(X_s_train)
X_s_test_scaled = scaler_s.transform(X_s_test)

# Crear y entrenar la red neuronal para salto
mlp_salto = MLPClassifier(
    hidden_layer_sizes=(40, 20),  # Capas más pequeñas para el modelo de salto
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

mlp_salto.fit(X_s_train_scaled, y_s_train)

# Evaluar modelo de salto
y_s_pred = mlp_salto.predict(X_s_test_scaled)
accuracy_s = accuracy_score(y_s_test, y_s_pred)
print(f"Precisión del Modelo de Salto (prueba): {accuracy_s:.4f}")
print("Reporte Clasificación Salto (prueba):\n", classification_report(y_s_test, y_s_pred, zero_division=0))

# Guardar modelo y scaler de salto
path_modelo_s = os.path.join(CARPETA_MODELOS, 'red_neuronal_jump_model.joblib')
path_scaler_s = os.path.join(CARPETA_MODELOS, 'red_neuronal_jump_scaler.joblib')
joblib.dump(mlp_salto, path_modelo_s)
joblib.dump(scaler_s, path_scaler_s)
print(f"Modelo de Salto guardado en: {path_modelo_s}")
print(f"Scaler de Salto guardado en: {path_scaler_s}")

print("\n¡Entrenamiento de Red Neuronal completado!")
print("Distribución de clases en datos originales:")
print("Horizontal:", y_h.value_counts().sort_index())
print("Salto:", y_s.value_counts().sort_index())