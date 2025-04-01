import cv2
import mediapipe as mp
import numpy as np
import json
import os

# Inicialización de MediaPipe Face Mesh (limitamos a una cara para simplificar)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                  max_num_faces=1, 
                                  min_detection_confidence=0.5, 
                                  min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Índices de landmarks: ojos, boca, nariz y pómulos
selected_points = [33, 133, 362, 263, 61, 291, 1, 234, 454]

def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def load_stored_vector(filename):
    """Carga el vector guardado desde un archivo JSON."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def save_stored_vector(vector, filename):
    """Guarda el vector en un archivo JSON."""
    with open(filename, 'w') as f:
        json.dump(vector, f)
    print(f"Vector guardado en: {filename}")

def compare_vectors(vec1, vec2, threshold=0.2):
    """
    Compara dos vectores de proporciones normalizadas.
    Se consideran similares si la diferencia en cada parámetro es menor al umbral.
    """
    differences = {}
    similar = True
    for key in vec1:
        if key in vec2 and vec1[key] is not None and vec2[key] is not None:
            diff = abs(vec1[key] - vec2[key])
            differences[key] = diff
            if diff > threshold:
                similar = False
    return similar, differences

# Archivo para almacenar el vector de la persona (se guarda una sola vez)
stored_vector_filename = "persona_guardada.json"
stored_vector = load_stored_vector(stored_vector_filename)

# Umbral para determinar si la cara está cerca o lejos (en píxeles de la distancia interocular)
umbral_cerca = 60  # Ajusta según tu cámara y resolución

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Espejo para mayor naturalidad
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_vector = {}
    face_scale = None  # Distancia interocular en píxeles (escala)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]  # Solo usamos la primera cara detectada
        puntos = {}
        # Extraer las coordenadas de los landmarks seleccionados
        for idx in selected_points:
            x = int(face_landmarks.landmark[idx].x * frame.shape[1])
            y = int(face_landmarks.landmark[idx].y * frame.shape[0])
            puntos[idx] = (x, y)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Calcular la distancia interocular (referencia) entre puntos 33 y 133
        if 33 in puntos and 133 in puntos:
            interocular = distancia(puntos[33], puntos[133])
            face_scale = interocular  # Medida absoluta de la cara en píxeles
            if interocular > 0:
                # Normalizamos usando la distancia interocular
                current_vector["distancia_ojos"] = 1.0
                current_vector["distancia_ojo_boca"] = distancia(puntos[133], puntos[61]) / interocular if 61 in puntos else None
                current_vector["distancia_nariz_izq"] = distancia(puntos[1], puntos[234]) / interocular if (1 in puntos and 234 in puntos) else None
                current_vector["distancia_nariz_der"] = distancia(puntos[1], puntos[454]) / interocular if (1 in puntos and 454 in puntos) else None
                current_vector["distancia_cheeks"] = distancia(puntos[234], puntos[454]) / interocular if (234 in puntos and 454 in puntos) else None

                # Eliminamos posibles entradas nulas
                current_vector = {k: v for k, v in current_vector.items() if v is not None}

        # Determinar si la cara está cerca o lejos usando la distancia interocular (scale)
        if face_scale is not None:
            if face_scale >= umbral_cerca:
                scale_status = "Cara cerca"
            else:
                scale_status = "Cara lejos"
            cv2.putText(frame, scale_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Si ya se guardó un vector, comparar el vector actual con el guardado (usando solo las proporciones)
        if stored_vector is not None and current_vector:
            similar, diffs = compare_vectors(current_vector, stored_vector, threshold=0.2)
            status_text = "Misma persona" if similar else "Persona distinta"
            # Mostrar el resultado en el frame (usamos el landmark 33 para posicionar el texto)
            if 33 in puntos:
                cv2.putText(frame, status_text, (puntos[33][0], puntos[33][1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            # Imprimir diferencias en consola (opcional)
            print("Diferencias en proporciones:", diffs)
        else:
            cv2.putText(frame, "No hay vector guardado", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mostrar los valores del vector normalizado en pantalla
        y0 = 50
        dy = 20
        for i, (key, value) in enumerate(current_vector.items()):
            cv2.putText(frame, f"{key}: {value:.2f}", (10, y0 + i * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Reconocimiento Facial con Proporciones", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Presiona "s" para guardar el vector actual (una sola vez)
    elif key == ord('s'):
        if current_vector:
            stored_vector = current_vector
            save_stored_vector(stored_vector, stored_vector_filename)

cap.release()
cv2.destroyAllWindows()
