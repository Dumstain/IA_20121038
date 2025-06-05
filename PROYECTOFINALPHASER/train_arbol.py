import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import joblib
import os
import graphviz

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

# --- 2. Preprocesamiento y Análisis ---
df.dropna(inplace=True) 
if df.empty:
    print(f"El archivo de datos quedó vacío después de eliminar NaNs.")
    exit()

print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
print("\n--- ANÁLISIS DE DISTRIBUCIÓN DE CLASES ---")
print("Movimiento Horizontal:")
print(df['accion_horizontal_comandada'].value_counts().sort_index())
print("\nSalto:")
print(df['accion_salto_comandado'].value_counts().sort_index())

# --- 3. FUNCIÓN PARA BALANCEAR DATOS ---
def balancear_datos(X, y, estrategia='oversample'):
    """
    Balancea los datos usando oversampling o undersampling
    """
    df_combined = X.copy()
    df_combined['target'] = y
    
    # Separar por clases
    clases = df_combined['target'].unique()
    class_counts = df_combined['target'].value_counts()
    
    print(f"\nDistribución original: {dict(class_counts)}")
    
    if estrategia == 'oversample':
        # Oversample: duplicar las clases minoritarias hasta llegar a la mayoría
        max_count = class_counts.max()
        
        df_balanceado = pd.DataFrame()
        for clase in clases:
            df_clase = df_combined[df_combined['target'] == clase]
            
            if len(df_clase) < max_count:
                # Oversample esta clase
                df_resampled = resample(df_clase, 
                                      replace=True, 
                                      n_samples=max_count, 
                                      random_state=42)
                df_balanceado = pd.concat([df_balanceado, df_resampled])
            else:
                df_balanceado = pd.concat([df_balanceado, df_clase])
    
    elif estrategia == 'undersample':
        # Undersample: reducir la clase mayoritaria
        min_count = class_counts.min()
        
        df_balanceado = pd.DataFrame()
        for clase in clases:
            df_clase = df_combined[df_combined['target'] == clase]
            df_resampled = resample(df_clase, 
                                  replace=False, 
                                  n_samples=min_count, 
                                  random_state=42)
            df_balanceado = pd.concat([df_balanceado, df_resampled])
    
    # Mezclar los datos
    df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_balanced = df_balanceado.drop('target', axis=1)
    y_balanced = df_balanceado['target']
    
    print(f"Distribución balanceada: {dict(y_balanced.value_counts().sort_index())}")
    
    return X_balanced, y_balanced

# --- 4. MODELO HORIZONTAL CON BALANCEO ---
print("\n--- Entrenando Modelo de Movimiento Horizontal (BALANCEADO) ---")

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
    exit()

# Balancear datos horizontales
X_h_balanced, y_h_balanced = balancear_datos(X_h, y_h, estrategia='oversample')

# División de datos balanceados
X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
    X_h_balanced, y_h_balanced, test_size=0.25, random_state=42, stratify=y_h_balanced
)

# Entrenar árbol con parámetros optimizados para clases desbalanceadas
clf_horizontal = DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
    min_samples_leaf=5,
    min_samples_split=10,
    class_weight='balanced',  # Importante para manejo de clases desbalanceadas
    criterion='gini'
)

clf_horizontal.fit(X_h_train, y_h_train)

# Evaluar
y_h_pred = clf_horizontal.predict(X_h_test)
accuracy_h = accuracy_score(y_h_test, y_h_pred)
print(f"Precisión del Modelo Horizontal: {accuracy_h:.4f}")
print("Reporte Clasificación Horizontal:\n", classification_report(y_h_test, y_h_pred, zero_division=0))

# Guardar modelo horizontal
path_modelo_h = os.path.join(CARPETA_MODELOS, 'arbol_horizontal_model.joblib')
joblib.dump(clf_horizontal, path_modelo_h)
print(f"Modelo Horizontal guardado en: {path_modelo_h}")

# --- 5. MODELO DE SALTO CON BALANCEO ---
print("\n--- Entrenando Modelo de Salto (BALANCEADO) ---")

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
    exit()

# Balancear datos de salto
X_s_balanced, y_s_balanced = balancear_datos(X_s, y_s, estrategia='oversample')

# División de datos balanceados
X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
    X_s_balanced, y_s_balanced, test_size=0.25, random_state=42, stratify=y_s_balanced
)

# Entrenar árbol de salto
clf_salto = DecisionTreeClassifier(
    max_depth=8,
    random_state=42,
    min_samples_leaf=5,
    min_samples_split=10,
    class_weight='balanced',
    criterion='gini'
)

clf_salto.fit(X_s_train, y_s_train)

# Evaluar
y_s_pred = clf_salto.predict(X_s_test)
accuracy_s = accuracy_score(y_s_test, y_s_pred)
print(f"Precisión del Modelo de Salto: {accuracy_s:.4f}")
print("Reporte Clasificación Salto:\n", classification_report(y_s_test, y_s_pred, zero_division=0))

# Guardar modelo de salto
path_modelo_s = os.path.join(CARPETA_MODELOS, 'arbol_jump_model.joblib')
joblib.dump(clf_salto, path_modelo_s)
print(f"Modelo de Salto guardado en: {path_modelo_s}")

# --- 6. ANÁLISIS DE CARACTERÍSTICAS IMPORTANTES ---
print("\n--- IMPORTANCIA DE CARACTERÍSTICAS ---")
print("Modelo Horizontal:")
feature_importance_h = pd.DataFrame({
    'feature': features_horizontal,
    'importance': clf_horizontal.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance_h)

print("\nModelo Salto:")
feature_importance_s = pd.DataFrame({
    'feature': features_salto,
    'importance': clf_salto.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance_s)

# --- 7. GENERAR VISUALIZACIONES ---
try:
    # Árbol horizontal
    dot_data_h = export_graphviz(clf_horizontal, out_file=None,
                                feature_names=features_horizontal,
                                class_names=['0_Quieto', '1_Izquierda', '2_Derecha'],
                                filled=True, rounded=True, special_characters=True, max_depth=3)
    graph_h = graphviz.Source(dot_data_h)
    graph_h.render(os.path.join(CARPETA_MODELOS, "arbol_horizontal_mejorado"), format="pdf", cleanup=True)
    
    # Árbol de salto
    dot_data_s = export_graphviz(clf_salto, out_file=None,
                                feature_names=features_salto,
                                class_names=['0_NoSaltar', '1_Saltar'],
                                filled=True, rounded=True, special_characters=True, max_depth=3)
    graph_s = graphviz.Source(dot_data_s)
    graph_s.render(os.path.join(CARPETA_MODELOS, "arbol_salto_mejorado"), format="pdf", cleanup=True)
    
    print("Visualizaciones guardadas en formato PDF")
except Exception as e:
    print(f"No se pudieron generar visualizaciones: {e}")

print("\n✅ ¡Entrenamiento de Árboles BALANCEADOS completado!")
print("Los modelos ahora deberían ser mucho mejores para detectar cuándo moverse y saltar.")