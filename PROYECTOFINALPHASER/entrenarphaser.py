import pandas as pd
# from sklearn.model_selection import train_test_split # Ya no es necesario si usamos archivos separados
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score # Para medir la precisión
import graphviz
import joblib # Para guardar/cargar el modelo

# --- PASO 1: Define los nombres de tus archivos CSV ---
# Asegúrate de que estos nombres coincidan con los archivos que generaste.
training_data_path = 'phaser.csv'  # Cambia esto por el nombre de tu archivo de entrenamiento (ej: D1.csv)
validation_data_path1 = 'phaser2.csv' # Cambia esto por el nombre de tu primer archivo de validación (ej: D2.csv)
validation_data_path2 = 'phaser3.csv' # Cambia esto por el nombre de tu segundo archivo de validación (ej: D3.csv)
model_filename = 'decision_tree_model.joblib' # Nombre para guardar el modelo

# --- PASO 2: Cargar el dataset de ENTRENAMIENTO ---
try:
    dataset_train = pd.read_csv(training_data_path)
    dataset_train.dropna(inplace=True) # Limpiar datos faltantes
    if dataset_train.empty:
        raise ValueError(f"El archivo de entrenamiento {training_data_path} está vacío.")
    print(f"Dataset de entrenamiento '{training_data_path}' cargado.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de entrenamiento {training_data_path}.")
    exit()
except Exception as e:
    print(f"Error al cargar el archivo de entrenamiento: {e}")
    exit()

# --- PASO 3: Preparar datos de ENTRENAMIENTO ---
# Asumiendo que las columnas son: velocidad_bala, distancia, salto
try:
    X_train = dataset_train[['velocidad_bala', 'distancia']] # Características
    y_train = dataset_train['salto']                         # Etiquetas
except KeyError as e:
    print(f"Error: Falta la columna '{e}' en {training_data_path}. Asegúrate de que las columnas se llamen 'velocidad_bala', 'distancia', 'salto'.")
    exit()

if len(X_train) < 10 or len(y_train.unique()) < 2:
     print(f"Error: No hay suficientes datos o clases en {training_data_path} para entrenar.")
     exit()

print("\nEntrenando el modelo...")
# --- PASO 4: Crear y Entrenar el modelo (SOLO con datos de entrenamiento) ---
# Puedes ajustar parámetros como max_depth
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
print("Modelo entrenado.")

# --- PASO 5: Guardar el modelo entrenado ---
try:
    joblib.dump(clf, model_filename)
    print(f"Modelo entrenado guardado como '{model_filename}'")
except Exception as e:
    print(f"Error al guardar el modelo: {e}")

# --- PASO 6: Validar con el PRIMER dataset de prueba ---
print(f"\nValidando con '{validation_data_path1}'...")
try:
    dataset_val1 = pd.read_csv(validation_data_path1)
    dataset_val1.dropna(inplace=True)
    if dataset_val1.empty:
        raise ValueError(f"El archivo de validación {validation_data_path1} está vacío.")

    # Preparar datos de validación 1
    X_val1 = dataset_val1[['velocidad_bala', 'distancia']]
    y_val1 = dataset_val1['salto']

    # Predecir con el modelo ENTRENADO
    y_pred1 = clf.predict(X_val1)

    # Calcular y mostrar la precisión
    accuracy1 = accuracy_score(y_val1, y_pred1)
    print(f"Precisión en {validation_data_path1}: {accuracy1:.4f}")

except FileNotFoundError:
    print(f"Error: No se encontró el archivo de validación {validation_data_path1}.")
except KeyError as e:
    print(f"Error: Falta la columna '{e}' en {validation_data_path1}.")
except Exception as e:
    print(f"Error al validar con {validation_data_path1}: {e}")

# --- PASO 7: Validar con el SEGUNDO dataset de prueba ---
print(f"\nValidando con '{validation_data_path2}'...")
try:
    dataset_val2 = pd.read_csv(validation_data_path2)
    dataset_val2.dropna(inplace=True)
    if dataset_val2.empty:
        raise ValueError(f"El archivo de validación {validation_data_path2} está vacío.")

    # Preparar datos de validación 2
    X_val2 = dataset_val2[['velocidad_bala', 'distancia']]
    y_val2 = dataset_val2['salto']

    # Predecir con el modelo ENTRENADO
    y_pred2 = clf.predict(X_val2)

    # Calcular y mostrar la precisión
    accuracy2 = accuracy_score(y_val2, y_pred2)
    print(f"Precisión en {validation_data_path2}: {accuracy2:.4f}")

except FileNotFoundError:
    print(f"Error: No se encontró el archivo de validación {validation_data_path2}.")
except KeyError as e:
    print(f"Error: Falta la columna '{e}' en {validation_data_path2}.")
except Exception as e:
    print(f"Error al validar con {validation_data_path2}: {e}")

# --- PASO 8: Visualizar el árbol entrenado (Guardar como PDF) ---
print("\nGenerando visualización del árbol en PDF (puede tardar)...")
try:
    dot_data = export_graphviz(clf, out_file=None,
                               feature_names=['velocidad_bala', 'distancia'],
                               class_names=['No Saltar', 'Saltar'], # Asegúrate que Clase 0 sea No Saltar, Clase 1 sea Saltar
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)

    # --- Modificación para generar PDF ---
    # Especifica el nombre base del archivo, el formato y si quieres que se abra (view=False)
    # cleanup=True eliminará el archivo intermedio .gv
    nombre_archivo_base = 'arbol_decision'
    graph.render(nombre_archivo_base, format='pdf', view=False, cleanup=True)
    # --- Fin Modificación ---

    print(f"Visualización del árbol guardada como '{nombre_archivo_base}.pdf'")

except Exception as e:
    print(f"Error al generar la visualización del árbol (Graphviz puede no estar instalado o configurado): {e}")
    print("Asegúrate de haber seguido los pasos para instalar Graphviz y añadirlo al PATH si el error persiste.")