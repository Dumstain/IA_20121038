import pygame
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier  # NUEVO: K-Vecinos
import graphviz
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Proyectiles Verticales")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
ROJO = (255, 0, 0)

# Variables del jugador, bala, nave, fondo, etc.
jugador = None
bala = None
fondo = None
nave = None
menu = None

# Variables de salto (sistema existente)
salto = False
salto_altura = 15
gravedad = 1
en_suelo = True

# Variables de movimiento horizontal (nuevo sistema)
movimiento_izquierda = False
movimiento_derecha = False
velocidad_movimiento = 3
posicion_centro = 50  # Spawn del jugador
posicion_min = 10     # Extremo izquierdo
posicion_max = 70     # Extremo derecho

# Variables de proyectiles verticales (nuevo sistema)
proyectil_vertical = None
proyectil_vertical_activo = False
velocidad_proyectil_vertical = 5
ultimo_disparo_vertical = 0
intervalo_disparo_vertical = 3000  # 3 segundos
pos_proyectil_izq = 35  # Donde cae proyectil izquierdo
pos_proyectil_der = 65  # Donde cae proyectil derecho

# Variables de pausa y menú
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False
mlp_clf = None
modo_mlp = False 
modo_knn = False  # NUEVO: modo K-Vecinos

# Listas para guardar los datos de los modelos
datos_modelo = []  # Para salto (sistema existente)
datos_movimiento = []  # Para movimiento (nuevo sistema)

# NUEVAS VARIABLES GLOBALES PARA ENTRENAMIENTO ÚNICO
# Estados de entrenamiento de modelos
modelos_entrenados = {
    'arboles_salto': False,
    'arboles_movimiento': False,
    'mlp_salto': False,
    'mlp_movimiento': False,
    'knn_salto': False,
    'knn_movimiento': False
}

# Modelos entrenados (para guardar referencias)
modelos_guardados = {
    'clf': None,
    'clf_movimiento': None,
    'mlp_model': None,
    'mlp_model_movimiento': None,
    'knn_salto': None,
    'knn_movimiento': None,
    'scaler_salto': None,
    'scaler_movimiento': None
}

# Cargar las imágenes
jugador_frames = [
    pygame.image.load('assets/sprites/mono_frame_1.png'),
    pygame.image.load('assets/sprites/mono_frame_2.png'),
    pygame.image.load('assets/sprites/mono_frame_3.png'),
    pygame.image.load('assets/sprites/mono_frame_4.png')
]

bala_img = pygame.image.load('assets/sprites/purple_ball.png')
fondo_img = pygame.image.load('assets/game/fondo2.png')
nave_img = pygame.image.load('assets/game/ufo.png')
menu_img = pygame.image.load('assets/game/menu.png')

# Escalar la imagen de fondo para que coincida con el tamaño de la pantalla
fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear el rectángulo del jugador y de la bala
jugador = pygame.Rect(posicion_centro, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10 
frame_count = 0

# Variables para la bala horizontal (sistema existente)
velocidad_bala = -10 
bala_disparada = False

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

#Variables para Scaler
scaler_salto = None
scaler_movimiento = None

# Variables globales para cache de predicciones (optimización)
ultima_prediccion_movimiento = 2
frames_sin_cambio_prediccion = 0

# NUEVAS FUNCIONES PARA MANEJO DE MODELOS
def limpiar_modelos_completamente():
    """Limpia todos los modelos entrenados y resetea sus estados"""
    global modelos_entrenados, modelos_guardados
    global clf, clf_movimiento, mlp_model, mlp_model_movimiento
    global knn_salto, knn_movimiento, scaler_salto, scaler_movimiento
    
    print("🧹 Limpiando todos los modelos entrenados...")
    
    # Resetear estados de entrenamiento
    for key in modelos_entrenados:
        modelos_entrenados[key] = False
    
    # Limpiar modelos
    clf = None
    clf_movimiento = None
    mlp_model = None
    mlp_model_movimiento = None
    knn_salto = None
    knn_movimiento = None
    scaler_salto = None
    scaler_movimiento = None
    
    # Limpiar modelos guardados
    for key in modelos_guardados:
        modelos_guardados[key] = None
    
    print("✅ Modelos limpiados completamente")

def guardar_modelos():
    """Guarda las referencias de los modelos entrenados"""
    global modelos_guardados
    modelos_guardados['clf'] = clf
    modelos_guardados['clf_movimiento'] = clf_movimiento
    modelos_guardados['mlp_model'] = mlp_model
    modelos_guardados['mlp_model_movimiento'] = mlp_model_movimiento
    modelos_guardados['knn_salto'] = knn_salto
    modelos_guardados['knn_movimiento'] = knn_movimiento
    modelos_guardados['scaler_salto'] = scaler_salto
    modelos_guardados['scaler_movimiento'] = scaler_movimiento

def cargar_modelos():
    """Carga los modelos previamente entrenados"""
    global clf, clf_movimiento, mlp_model, mlp_model_movimiento
    global knn_salto, knn_movimiento, scaler_salto, scaler_movimiento
    
    clf = modelos_guardados['clf']
    clf_movimiento = modelos_guardados['clf_movimiento']
    mlp_model = modelos_guardados['mlp_model']
    mlp_model_movimiento = modelos_guardados['mlp_model_movimiento']
    knn_salto = modelos_guardados['knn_salto']
    knn_movimiento = modelos_guardados['knn_movimiento']
    scaler_salto = modelos_guardados['scaler_salto']
    scaler_movimiento = modelos_guardados['scaler_movimiento']

# Función para disparar la bala horizontal (sistema existente)
def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-8, -3)  
        bala_disparada = True

# Función para reiniciar la posición de la bala horizontal
def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50 
    bala_disparada = False

# Función para disparar proyectil vertical (nuevo sistema)
def disparar_proyectil_vertical():
    global proyectil_vertical, proyectil_vertical_activo, ultimo_disparo_vertical
    tiempo_actual = pygame.time.get_ticks()
    
    if tiempo_actual - ultimo_disparo_vertical > intervalo_disparo_vertical:
        if not proyectil_vertical_activo:
            # CAMBIO: Siempre cae en el centro donde spawnea el jugador
            proyectil_vertical = pygame.Rect(posicion_centro, -20, 16, 16)
            proyectil_vertical_activo = True
            ultimo_disparo_vertical = tiempo_actual

# Función para mover proyectil vertical
def mover_proyectil_vertical():
    global proyectil_vertical, proyectil_vertical_activo
    
    if proyectil_vertical_activo:
        proyectil_vertical.y += velocidad_proyectil_vertical
        
        # Si el proyectil sale de la pantalla, reiniciarlo
        if proyectil_vertical.y > h:
            proyectil_vertical_activo = False

# Función para manejar el movimiento horizontal (nuevo sistema)
def manejar_movimiento_horizontal():
    global jugador, movimiento_izquierda, movimiento_derecha
    
    # CAMBIO: Convertir posiciones a enteros para evitar problemas de comparación
    if movimiento_izquierda and jugador.x > posicion_min:
        jugador.x -= velocidad_movimiento
        jugador.x = max(jugador.x, posicion_min)  # Asegurar que no pase el límite
    elif movimiento_derecha and jugador.x < posicion_max:
        jugador.x += velocidad_movimiento
        jugador.x = min(jugador.x, posicion_max)  # Asegurar que no pase el límite
        
    # Solo regresar al centro si NO se está presionando ninguna tecla
    if not movimiento_izquierda and not movimiento_derecha:
        # Movimiento automático hacia el centro cuando no se presiona nada
        velocidad_retorno = 0.8  # Aumenté un poco la velocidad para que sea más notorio
        if jugador.x < posicion_centro - 2:  # Margen de 2 pixels
            jugador.x += velocidad_retorno
            jugador.x = min(jugador.x, posicion_centro)  # No sobrepasar el centro
        elif jugador.x > posicion_centro + 2:
            jugador.x -= velocidad_retorno
            jugador.x = max(jugador.x, posicion_centro)  # No sobrepasar el centro
        else:
            jugador.x = posicion_centro  # Centrarlo exactamente cuando está muy cerca

# Función para manejar el salto (sistema existente)
def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura  
        salto_altura -= gravedad  

        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 15  
            en_suelo = True

# Función para actualizar el juego
def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2

    # Mover el fondo
    fondo_x1 -= 1
    fondo_x2 -= 1

    if fondo_x1 <= -w:
        fondo_x1 = w
    if fondo_x2 <= -w:
        fondo_x2 = w

    # Dibujar los fondos
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0

    # Dibujar el jugador con la animación
    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))

    # Dibujar la nave
    pantalla.blit(nave_img, (nave.x, nave.y))

    # Mover y dibujar la bala horizontal
    if bala_disparada:
        bala.x += velocidad_bala

    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    # Manejar proyectiles verticales
    disparar_proyectil_vertical()
    mover_proyectil_vertical()
    
    # Dibujar proyectil vertical si está activo
    if proyectil_vertical_activo:
        pygame.draw.rect(pantalla, ROJO, proyectil_vertical)

    # Colisión entre la bala horizontal y el jugador
    if jugador.colliderect(bala):
        print("Colisión con bala horizontal!")
        reiniciar_juego()

    # Colisión entre proyectil vertical y el jugador
    if proyectil_vertical_activo and jugador.colliderect(proyectil_vertical):
        print("Colisión con proyectil vertical!")
        reiniciar_juego()

# Función para guardar datos del modelo en modo manual
def guardar_datos():
    global jugador, bala, velocidad_bala, salto, proyectil_vertical, proyectil_vertical_activo
    global movimiento_izquierda, movimiento_derecha
    
    # Datos para salto (sistema existente)
    distancia_bala = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0
    datos_modelo.append((velocidad_bala, distancia_bala, salto_hecho))
    
    # Datos para movimiento horizontal (nuevo sistema)
    if proyectil_vertical_activo:
        distancia_proyectil_vertical = abs(jugador.y - proyectil_vertical.y)
        posicion_proyectil_x = proyectil_vertical.x
        posicion_jugador_x = jugador.x
        
        # Determinar movimiento realizado
        movimiento_hecho = 2  # Sin movimiento por defecto
        if movimiento_izquierda:
            movimiento_hecho = 0  # Izquierda
        elif movimiento_derecha:
            movimiento_hecho = 1  # Derecha
            
        datos_movimiento.append((velocidad_proyectil_vertical, distancia_proyectil_vertical, 
                               posicion_proyectil_x, posicion_jugador_x, movimiento_hecho))

# Función para pausar el juego y guardar los datos
def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado.")
        print("Datos salto:", len(datos_modelo))
        print("Datos movimiento:", len(datos_movimiento))
    else:
        print("Juego reanudado.")

# FUNCIONES DE ENTRENAMIENTO MODIFICADAS PARA ENTRENAMIENTO ÚNICO

# Función decision Tree para salto MODIFICADA
def arbolDec():
    global clf, modelos_entrenados
    
    # Solo entrenar si no está entrenado
    if not modelos_entrenados['arboles_salto']:
        print("🌳 Entrenando Árbol de Decisión para SALTO por primera vez...")
        dataset = pd.DataFrame(datos_modelo, columns=['velocidad_bala', 'distancia', 'salto_hecho'])
        
        # 🔧 CAMBIO CLAVE: Usar .values para entrenar SIN nombres de columna
        X = dataset.iloc[:, :2].values  # Convierte a numpy array sin nombres
        y = dataset.iloc[:, 2].values   # Convierte a numpy array
        
        clf = DecisionTreeClassifier(random_state=42)  # Semilla para consistencia
        clf.fit(X, y)
        modelos_entrenados['arboles_salto'] = True
        guardar_modelos()
        print("✅ Árbol de salto entrenado y guardado (sin warnings)")
    else:
        print("🔄 Cargando Árbol de Decisión para salto ya entrenado...")
        cargar_modelos()

# Función decision Tree para movimiento MODIFICADA
def arbolDec_movimiento():
    global clf_movimiento, modelos_entrenados
    
    # Solo entrenar si no está entrenado
    if not modelos_entrenados['arboles_movimiento']:
        print("🌳 Entrenando Árbol de Decisión para MOVIMIENTO por primera vez...")
        dataset = pd.DataFrame(datos_movimiento, columns=['velocidad_proyectil', 'distancia_vertical', 
                                                        'posicion_proyectil_x', 'posicion_jugador_x', 'movimiento_hecho'])
        
        # 🔧 CAMBIO CLAVE: Usar .values para entrenar SIN nombres de columna
        X = dataset.iloc[:, :4].values  # Convierte a numpy array sin nombres
        y = dataset.iloc[:, 4].values   # Convierte a numpy array
        
        clf_movimiento = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=3,    # Menos restrictivo para captar patrones
            min_samples_leaf=1,     # Permite hojas individuales
            random_state=42         # Semilla para consistencia
        )
        clf_movimiento.fit(X, y)
        modelos_entrenados['arboles_movimiento'] = True
        guardar_modelos()
        print("✅ Árbol de movimiento entrenado y guardado (sin warnings)")
    else:
        print("🔄 Cargando Árbol de Decisión para movimiento ya entrenado...")
        cargar_modelos()

# Función redes neuronales para salto MODIFICADA
def entrenar_mlp():
    global mlp_model, scaler_salto, modelos_entrenados
    
    if not modelos_entrenados['mlp_salto']:
        if len(datos_modelo) == 0:
            print("No hay datos suficientes para entrenar el modelo MLP de salto.")
            return

        print(f"🧠 Entrenando MLP OPTIMIZADO para salto con {len(datos_modelo)} ejemplos...")
        dataset = pd.DataFrame(datos_modelo, columns=['velocidad_bala', 'distancia', 'salto_hecho'])
        X = dataset.iloc[:, :2].values  
        y = dataset.iloc[:, 2].values

        # 🔥 NORMALIZACIÓN - ¡Esto es CRUCIAL!
        scaler_salto = StandardScaler()
        X_normalized = scaler_salto.fit_transform(X)
        print("✅ Datos normalizados - ahora todas las características están en la misma escala")

        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.25, random_state=42)
        print(f"Datos de entrenamiento: {len(X_train)}, Datos de prueba: {len(X_test)}")

        # 🧠 ARQUITECTURA OPTIMIZADA
        mlp_model = Sequential([
            Dense(8, input_dim=2, activation='relu'),  # ReLU es más estable que swish
            Dense(4, activation='relu'),               # Capa oculta más pequeña
            Dense(1, activation='sigmoid')             # Salida binaria
        ])
        
        # 🎯 OPTIMIZADOR MEJORADO
        mlp_model.compile(
            optimizer=Adam(learning_rate=0.001),  # Learning rate más conservador
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        print("🚀 Modelo MLP optimizado compilado. Iniciando entrenamiento...")

        # 🔄 ENTRENAMIENTO OPTIMIZADO
        history = mlp_model.fit(
            X_train, y_train, 
            epochs=50,              # Menos épocas, más eficiente
            batch_size=16,          # Batch más pequeño
            verbose=1, 
            validation_split=0.2
        )

        # 🧪 EVALUACIÓN
        loss, accuracy = mlp_model.evaluate(X_test, y_test, verbose=1)
        
        modelos_entrenados['mlp_salto'] = True
        guardar_modelos()
        print(f"✅ MLP salto entrenado por primera vez - Precisión: {accuracy:.3f}")
    else:
        print("🔄 Cargando MLP para salto ya entrenado...")
        cargar_modelos()

# Función redes neuronales para movimiento MODIFICADA
def entrenar_mlp_movimiento():
    global mlp_model_movimiento, scaler_movimiento, modelos_entrenados
    
    if not modelos_entrenados['mlp_movimiento']:
        if len(datos_movimiento) == 0:
            print("No hay datos suficientes para entrenar el modelo MLP de movimiento.")
            return

        print(f"🧠 Entrenando MLP OPTIMIZADO para movimiento con {len(datos_movimiento)} ejemplos...")
        dataset = pd.DataFrame(datos_movimiento, columns=['velocidad_proyectil', 'distancia_vertical', 
                                                        'posicion_proyectil_x', 'posicion_jugador_x', 'movimiento_hecho'])
        
        # 🎯 FEATURE ENGINEERING - Crear características más útiles
        dataset['diferencia_posicion'] = dataset['posicion_proyectil_x'] - dataset['posicion_jugador_x']
        dataset['distancia_normalizada'] = dataset['distancia_vertical'] / 400.0  # Normalizar manualmente
        dataset['urgencia'] = np.where(dataset['distancia_vertical'] < 50, 1, 0)  # ¿Emergencia?
        
        # Usar solo las características útiles (eliminamos velocidad_proyectil constante)
        X = dataset[['distancia_normalizada', 'diferencia_posicion', 'urgencia']].values
        y = dataset['movimiento_hecho'].values
        
        print("🔧 Features creadas: distancia_normalizada, diferencia_posicion, urgencia")

        # 🔥 NORMALIZACIÓN
        scaler_movimiento = StandardScaler()
        X_normalized = scaler_movimiento.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.25, random_state=42)
        print(f"Datos de entrenamiento: {len(X_train)}, Datos de prueba: {len(X_test)}")

        # 🧠 ARQUITECTURA OPTIMIZADA PARA CLASIFICACIÓN MULTICLASE
        mlp_model_movimiento = Sequential([
            Dense(12, input_dim=3, activation='relu'),  # Más neuronas para 3 clases
            Dense(8, activation='relu'),                
            Dense(4, activation='relu'),                # Capa adicional
            Dense(3, activation='softmax')              # 3 clases: izq, der, centro
        ])
        
        mlp_model_movimiento.compile(
            optimizer=Adam(learning_rate=0.001), 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        print("🚀 Modelo MLP movimiento optimizado compilado. Iniciando entrenamiento...")

        history = mlp_model_movimiento.fit(
            X_train, y_train, 
            epochs=60,              # Más épocas para clasificación multiclase
            batch_size=16, 
            verbose=1, 
            validation_split=0.2
        )

        loss, accuracy = mlp_model_movimiento.evaluate(X_test, y_test, verbose=1)
        
        modelos_entrenados['mlp_movimiento'] = True
        guardar_modelos()
        print(f"✅ MLP movimiento entrenado por primera vez - Precisión: {accuracy:.3f}")
    else:
        print("🔄 Cargando MLP para movimiento ya entrenado...")
        cargar_modelos()

# NUEVO: Función K-Vecinos para salto MODIFICADA
def entrenar_knn():
    global knn_salto, modelos_entrenados
    
    if not modelos_entrenados['knn_salto']:
        if len(datos_modelo) < 10:
            print("No hay suficientes datos para entrenar K-Vecinos de salto.")
            return
        
        print("🎯 Entrenando K-Vecinos para SALTO por primera vez...")
        dataset = pd.DataFrame(datos_modelo, columns=['velocidad_bala', 'distancia', 'salto_hecho'])
        X = dataset.iloc[:, :2].values  
        y = dataset.iloc[:, 2].values
        
        # Usar K=5 vecinos, pero ajustar si hay pocos datos
        k_optimo = min(5, len(datos_modelo) // 3)  # K no mayor a 1/3 de los datos
        if k_optimo < 3:
            k_optimo = 3
        
        knn_salto = KNeighborsClassifier(
            n_neighbors=k_optimo,
            weights='distance',  # Dar más peso a vecinos más cercanos
            metric='euclidean'   # Distancia euclidiana estándar
        )
        
        knn_salto.fit(X, y)
        modelos_entrenados['knn_salto'] = True
        guardar_modelos()
        print(f"✅ K-Vecinos salto entrenado con K={k_optimo}")
    else:
        print("🔄 Cargando K-Vecinos para salto ya entrenado...")
        cargar_modelos()

# NUEVO: Función K-Vecinos para movimiento MODIFICADA
def entrenar_knn_movimiento():
    global knn_movimiento, modelos_entrenados
    
    if not modelos_entrenados['knn_movimiento']:
        if len(datos_movimiento) < 10:
            print("No hay suficientes datos para entrenar K-Vecinos de movimiento.")
            return
        
        print("🎯 Entrenando K-Vecinos para MOVIMIENTO por primera vez...")
        dataset = pd.DataFrame(datos_movimiento, columns=['velocidad_proyectil', 'distancia_vertical', 
                                                        'posicion_proyectil_x', 'posicion_jugador_x', 'movimiento_hecho'])
        X = dataset.iloc[:, :4].values  
        y = dataset.iloc[:, 4].values
        
        # Usar K=7 vecinos para movimiento (más opciones, más vecinos)
        k_optimo = min(7, len(datos_movimiento) // 3)
        if k_optimo < 3:
            k_optimo = 3
        
        knn_movimiento = KNeighborsClassifier(
            n_neighbors=k_optimo,
            weights='distance',  # Dar más peso a vecinos más cercanos
            metric='euclidean'   # Distancia euclidiana estándar
        )
        
        knn_movimiento.fit(X, y)
        modelos_entrenados['knn_movimiento'] = True
        guardar_modelos()
        print(f"✅ K-Vecinos movimiento entrenado con K={k_optimo}")
    else:
        print("🔄 Cargando K-Vecinos para movimiento ya entrenado...")
        cargar_modelos()

# Función para mostrar el menú MODIFICADA
def mostrar_menu():
    global menu_activo, modo_auto, datos_modelo, datos_movimiento, clf, modo_mlp, modo_knn
    pantalla.fill(NEGRO)
    lineas_menu = [
        "Menu:",
        "'A' para auto con arboles",
        "'R' para auto con redes",
        "'K' para auto con K-vecinos",
        "'M' para Manual",
        "'G' para gráfica",
        "'S' Dentro del juego para regresar al menu",
        "'Q' para Salir"
    ]
    
    x_inicial = w // 4
    y_inicial = h // 3 - 20
    espaciado = 25
    for i, linea in enumerate(lineas_menu):
        texto = fuente.render(linea, True, BLANCO)
        pantalla.blit(texto, (x_inicial, y_inicial + i * espaciado))
    
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    if len(datos_modelo) == 0 or len(datos_movimiento) == 0:
                        print("No hay suficientes datos para entrenar los modelos.")
                        menu_activo = True  
                    else:
                        modo_auto = True
                        menu_activo = False
                        modo_mlp = False
                        modo_knn = False
                        arbolDec()
                        arbolDec_movimiento()
                        jugador.x = posicion_centro
                        jugador.y = h - 100 
                        salto = False 
                        en_suelo = True  
                        print("Modo Automático: Usando árboles de decisión...")
                
                elif evento.key == pygame.K_r:
                    if len(datos_modelo) == 0 or len(datos_movimiento) == 0:
                        print("No hay suficientes datos para entrenar los modelos.")
                        menu_activo = True
                    else:
                        modo_auto = True
                        modo_mlp = True
                        modo_knn = False
                        menu_activo = False
                        entrenar_mlp()
                        entrenar_mlp_movimiento()
                        jugador.x = posicion_centro
                        jugador.y = h - 100
                        print("Modo Automático (MLP): Usando redes neuronales...")
                
                elif evento.key == pygame.K_k:
                    if len(datos_modelo) < 10 or len(datos_movimiento) < 10:
                        print("No hay suficientes datos para K-Vecinos (mínimo 10 ejemplos cada uno).")
                        menu_activo = True
                    else:
                        modo_auto = True
                        modo_mlp = False
                        modo_knn = True
                        menu_activo = False
                        entrenar_knn()
                        entrenar_knn_movimiento()
                        jugador.x = posicion_centro
                        jugador.y = h - 100
                        print("Modo Automático (K-Vecinos): Usando vecinos más cercanos...")
                        
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                    # LIMPIAR MODELOS COMPLETAMENTE cuando se elige manual
                    limpiar_modelos_completamente()
                    # Limpiar datos para empezar fresco
                    datos_modelo = []
                    datos_movimiento = []
                    jugador.x = posicion_centro
                    jugador.y = h - 100
                    salto = False
                    en_suelo = True
                    print("🎮 Modo Manual seleccionado - Modelos limpiados, recolectando nuevos datos...")
                    
                elif evento.key == pygame.K_q:
                    print("Juego terminado.")
                    pygame.quit()
                    exit()
                elif evento.key == pygame.K_g:
                    graficar()

# Función para graficar
def graficar():
    if len(datos_modelo) > 0:
        df_salto = pd.DataFrame(datos_modelo, columns=['velocidad_bala', 'distancia', 'salto_hecho'])
        
        fig = plt.figure(figsize=(15, 5))
        
        # Gráfico 1: Datos de salto
        ax1 = fig.add_subplot(131, projection="3d")
        ax1.scatter(df_salto["velocidad_bala"], df_salto["distancia"], df_salto["salto_hecho"])
        ax1.set_xlabel("Velocidad Bala")
        ax1.set_ylabel("Distancia")
        ax1.set_zlabel("Salto")
        ax1.set_title("Datos Salto")
    
    if len(datos_movimiento) > 0:
        df_mov = pd.DataFrame(datos_movimiento, columns=['velocidad_proyectil', 'distancia_vertical', 
                                                       'posicion_proyectil_x', 'posicion_jugador_x', 'movimiento_hecho'])
        
        # Gráfico 2: Posiciones
        ax2 = fig.add_subplot(132)
        ax2.scatter(df_mov["posicion_proyectil_x"], df_mov["posicion_jugador_x"], c=df_mov["movimiento_hecho"])
        ax2.set_xlabel("Posición Proyectil X")
        ax2.set_ylabel("Posición Jugador X")
        ax2.set_title("Posiciones y Movimientos")
        
        # Gráfico 3: Distancias
        ax3 = fig.add_subplot(133, projection="3d")
        ax3.scatter(df_mov["distancia_vertical"], df_mov["posicion_proyectil_x"], df_mov["movimiento_hecho"])
        ax3.set_xlabel("Distancia Vertical")
        ax3.set_ylabel("Posición Proyectil")
        ax3.set_zlabel("Movimiento")
        ax3.set_title("Datos Movimiento")
    
    plt.tight_layout()
    plt.show()

# Función para reiniciar el juego tras la colisión MODIFICADA
def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo, modo_auto
    global proyectil_vertical_activo, movimiento_izquierda, movimiento_derecha
    
    menu_activo = True
    jugador.x, jugador.y = posicion_centro, h - 100
    bala.x = w - 50
    nave.x, nave.y = w - 100, h - 100
    bala_disparada = False
    proyectil_vertical_activo = False
    salto = False
    en_suelo = True
    movimiento_izquierda = False
    movimiento_derecha = False
    
    if not modo_auto: 
        print("Datos salto:", len(datos_modelo))
        print("Datos movimiento:", len(datos_movimiento))
    
    # NO limpiar escaladores aquí para mantener modelos entrenados
    mostrar_menu()

# FUNCIONES DE PREDICCIÓN

# Predicción para salto (existente)
def arbol_decision_predict():
    global clf, velocidad_bala, jugador, bala  # 🔧 AGREGAR variables globales
    if clf is None:
        return 0
    
    datos_actuales = [velocidad_bala, abs(jugador.x - bala.x)]
    prediccion = clf.predict([datos_actuales])
    return int(prediccion[0])

def mlp_predict():
    global mlp_model, scaler_salto, velocidad_bala, jugador, bala  # 🔧 AGREGAR variables globales
    if mlp_model is None or scaler_salto is None:
        return 0
    
    datos_actuales = np.array([[velocidad_bala, abs(jugador.x - bala.x)]])
    datos_normalizados = scaler_salto.transform(datos_actuales)
    prediccion = mlp_model.predict(datos_normalizados, verbose=0)
    
    probabilidad_salto = prediccion[0][0]
    
    # 🎯 DEBUG: Descomenta para ver probabilidades
    print(f"🧠 Salto - Probabilidad: {probabilidad_salto:.3f}")
    
    if probabilidad_salto < 0.3:
        return 0
    elif probabilidad_salto > 0.7:
        return 1
    else:
        return 0 if probabilidad_salto < 0.5 else 1

# NUEVO: Predicción K-Vecinos para salto

# REEMPLAZA estas funciones de predicción en tu código:

# 1. FUNCIÓN ARBOL SALTO - CORREGIDA
def arbol_decision_predict():
    global clf, velocidad_bala, jugador, bala  # 🔧 AGREGAR variables globales
    if clf is None:
        return 0
    
    datos_actuales = [velocidad_bala, abs(jugador.x - bala.x)]
    prediccion = clf.predict([datos_actuales])
    return int(prediccion[0])

# 2. FUNCIÓN ARBOL MOVIMIENTO - CORREGIDA  
def arbol_decision_predict_movimiento():
    global clf_movimiento, proyectil_vertical, proyectil_vertical_activo
    global ultima_prediccion_movimiento, frames_sin_cambio_prediccion
    global velocidad_proyectil_vertical, jugador  # 🔧 AGREGAR variables globales
    
    if not proyectil_vertical_activo or clf_movimiento is None:
        return 2
    
    frames_sin_cambio_prediccion += 1
    if frames_sin_cambio_prediccion < 5:
        return ultima_prediccion_movimiento
    
    frames_sin_cambio_prediccion = 0
    
    try:
        distancia_vertical = abs(jugador.y - proyectil_vertical.y)
        
        datos_actuales = [
            velocidad_proyectil_vertical,  
            distancia_vertical,            
            proyectil_vertical.x,         
            jugador.x                     
        ]
        
        prediccion = clf_movimiento.predict([datos_actuales])
        ultima_prediccion_movimiento = int(prediccion[0])
        
        # 🎯 DEBUG: Descomenta para ver decisiones
        decision_texto = ["IZQUIERDA", "DERECHA", "QUIETO"][ultima_prediccion_movimiento]
        print(f"🌳 Árbol decide: {decision_texto}")
        
        return ultima_prediccion_movimiento
        
    except Exception as e:
        print(f"🌳 Error en predicción: {e}")
        return 2

# 3. FUNCIÓN MLP SALTO - CORREGIDA
def mlp_predict():
    global mlp_model, scaler_salto, velocidad_bala, jugador, bala  # 🔧 AGREGAR variables globales
    if mlp_model is None or scaler_salto is None:
        return 0
    
    datos_actuales = np.array([[velocidad_bala, abs(jugador.x - bala.x)]])
    datos_normalizados = scaler_salto.transform(datos_actuales)
    prediccion = mlp_model.predict(datos_normalizados, verbose=0)
    
    probabilidad_salto = prediccion[0][0]
    
    # 🎯 DEBUG: Descomenta para ver probabilidades
    print(f"🧠 Salto - Probabilidad: {probabilidad_salto:.3f}")
    
    if probabilidad_salto < 0.3:
        return 0
    elif probabilidad_salto > 0.7:
        return 1
    else:
        return 0 if probabilidad_salto < 0.5 else 1

# 4. FUNCIÓN MLP MOVIMIENTO - CORREGIDA
def mlp_predict_movimiento():
    global mlp_model_movimiento, scaler_movimiento, proyectil_vertical, proyectil_vertical_activo
    global jugador  # 🔧 AGREGAR variables globales
    
    if mlp_model_movimiento is None or not proyectil_vertical_activo or scaler_movimiento is None:
        return 2
    
    distancia_normalizada = abs(jugador.y - proyectil_vertical.y) / 400.0
    diferencia_posicion = proyectil_vertical.x - jugador.x
    urgencia = 1 if abs(jugador.y - proyectil_vertical.y) < 50 else 0
    
    datos_actuales = np.array([[distancia_normalizada, diferencia_posicion, urgencia]])
    datos_normalizados = scaler_movimiento.transform(datos_actuales)
    prediccion = mlp_model_movimiento.predict(datos_normalizados, verbose=0)
    
    prob_izquierda = prediccion[0][0]
    prob_derecha = prediccion[0][1] 
    prob_quieto = prediccion[0][2]
    
    # 🎯 DEBUG: Descomenta para ver probabilidades
    print(f"🧠 Movimiento - Izq={prob_izquierda:.3f}, Der={prob_derecha:.3f}, Quieto={prob_quieto:.3f}")
    
    max_prob = max(prob_izquierda, prob_derecha, prob_quieto)
    
    if prob_quieto > 0.6:
        print("🧠 Red muy segura: QUEDARSE QUIETO")
        return 2
    elif max_prob < 0.4:
        print("🧠 Red insegura: QUEDARSE QUIETO por defecto")
        return 2
    elif prob_izquierda > 0.6:
        print("🧠 Red segura: MOVER IZQUIERDA")
        return 0
    elif prob_derecha > 0.6:
        print("🧠 Red segura: MOVER DERECHA") 
        return 1
    else:
        print("🧠 Red insegura sobre dirección: QUEDARSE QUIETO")
        return 2

# 5. FUNCIÓN KNN SALTO - CORREGIDA
def knn_predict():
    global knn_salto, velocidad_bala, jugador, bala  # 🔧 AGREGAR variables globales
    if knn_salto is None:
        return 0
    
    datos_actuales = [[velocidad_bala, abs(jugador.x - bala.x)]]
    prediccion = knn_salto.predict(datos_actuales)
    
    # 🎯 DEBUG: Descomenta para ver decisiones
    decision = int(prediccion[0])
    print(f"🎯 KNN Salto decide: {'SALTAR' if decision == 1 else 'NO SALTAR'}")
    
    return decision

# PREDICCIÓN OPTIMIZADA PARA MOVIMIENTO
def mlp_predict_movimiento():
    global mlp_model_movimiento, scaler_movimiento, proyectil_vertical, proyectil_vertical_activo
    global jugador  # 🔧 AGREGAR variables globales
    
    if mlp_model_movimiento is None or not proyectil_vertical_activo or scaler_movimiento is None:
        return 2
    
    distancia_normalizada = abs(jugador.y - proyectil_vertical.y) / 400.0
    diferencia_posicion = proyectil_vertical.x - jugador.x
    urgencia = 1 if abs(jugador.y - proyectil_vertical.y) < 50 else 0
    
    datos_actuales = np.array([[distancia_normalizada, diferencia_posicion, urgencia]])
    datos_normalizados = scaler_movimiento.transform(datos_actuales)
    prediccion = mlp_model_movimiento.predict(datos_normalizados, verbose=0)
    
    prob_izquierda = prediccion[0][0]
    prob_derecha = prediccion[0][1] 
    prob_quieto = prediccion[0][2]
    
    # 🎯 DEBUG: Descomenta para ver probabilidades
    print(f"🧠 Movimiento - Izq={prob_izquierda:.3f}, Der={prob_derecha:.3f}, Quieto={prob_quieto:.3f}")
    
    max_prob = max(prob_izquierda, prob_derecha, prob_quieto)
    
    if prob_quieto > 0.6:
        print("🧠 Red muy segura: QUEDARSE QUIETO")
        return 2
    elif max_prob < 0.4:
        print("🧠 Red insegura: QUEDARSE QUIETO por defecto")
        return 2
    elif prob_izquierda > 0.6:
        print("🧠 Red segura: MOVER IZQUIERDA")
        return 0
    elif prob_derecha > 0.6:
        print("🧠 Red segura: MOVER DERECHA") 
        return 1
    else:
        print("🧠 Red insegura sobre dirección: QUEDARSE QUIETO")
        return 2



def arbol_decision_predict_movimiento():
    global clf_movimiento, proyectil_vertical, proyectil_vertical_activo
    global ultima_prediccion_movimiento, frames_sin_cambio_prediccion
    global velocidad_proyectil_vertical, jugador  # 🔧 AGREGAR variables globales
    
    if not proyectil_vertical_activo or clf_movimiento is None:
        return 2
    
    frames_sin_cambio_prediccion += 1
    if frames_sin_cambio_prediccion < 5:
        return ultima_prediccion_movimiento
    
    frames_sin_cambio_prediccion = 0
    
    try:
        distancia_vertical = abs(jugador.y - proyectil_vertical.y)
        
        datos_actuales = [
            velocidad_proyectil_vertical,  
            distancia_vertical,            
            proyectil_vertical.x,         
            jugador.x                     
        ]
        
        prediccion = clf_movimiento.predict([datos_actuales])
        ultima_prediccion_movimiento = int(prediccion[0])
        
        # 🎯 DEBUG: Descomenta para ver decisiones
        decision_texto = ["IZQUIERDA", "DERECHA", "QUIETO"][ultima_prediccion_movimiento]
        print(f"🌳 Árbol decide: {decision_texto}")
        
        return ultima_prediccion_movimiento
        
    except Exception as e:
        print(f"🌳 Error en predicción: {e}")
        return 2
    
def inspeccionar_arbol():
    global clf_movimiento, datos_movimiento
    
    if clf_movimiento is None:
        print("🌳 No hay árbol entrenado para inspeccionar")
        return
    
    print("🌳 INSPECCIÓN DEL ÁRBOL:")
    print(f"   - Número de nodos: {clf_movimiento.tree_.node_count}")
    print(f"   - Profundidad máxima: {clf_movimiento.tree_.max_depth}")
    print(f"   - Número de hojas: {clf_movimiento.get_n_leaves()}")
    print(f"   - Clases detectadas: {clf_movimiento.classes_}")
    
    # Mostrar distribución de clases en datos de entrenamiento
    if len(datos_movimiento) > 0:
        df = pd.DataFrame(datos_movimiento, columns=['velocidad_proyectil', 'distancia_vertical', 
                                                   'posicion_proyectil_x', 'posicion_jugador_x', 'movimiento_hecho'])
        distribucion = df['movimiento_hecho'].value_counts().sort_index()
        print(f"🌳 Distribución de clases en entrenamiento:")
        for clase, cantidad in distribucion.items():
            nombre_clase = ["Izquierda", "Derecha", "Quieto"][clase]
            print(f"   - Clase {clase} ({nombre_clase}): {cantidad} ejemplos")
    
    # Hacer una predicción de prueba
    print(f"🌳 Predicción de prueba con datos neutros:")
    datos_prueba = [5, 200, 50, 50]  # Datos del centro
    prediccion_prueba = clf_movimiento.predict([datos_prueba])
    print(f"   - Input: {datos_prueba}")
    print(f"   - Output: {prediccion_prueba[0]} ({['Izquierda', 'Derecha', 'Quieto'][prediccion_prueba[0]]})")

# VERSIÓN PARA DEBUGGING COMPLETO
def arbol_decision_predict_movimiento_debug():
    global clf_movimiento, proyectil_vertical, proyectil_vertical_activo
    
    if not proyectil_vertical_activo:
        print("🌳 No hay proyectil vertical activo")
        return 2
    
    if clf_movimiento is None:
        print("🌳 ⚠️ No hay árbol entrenado")
        return 2
    
    # Preparar datos exactamente como en entrenamiento
    distancia_vertical = abs(jugador.y - proyectil_vertical.y)
    datos_actuales = [velocidad_proyectil_vertical, distancia_vertical,
                     proyectil_vertical.x, jugador.x]
    
    print(f"🌳 INPUT al árbol:")
    print(f"   - velocidad_proyectil_vertical: {velocidad_proyectil_vertical}")
    print(f"   - distancia_vertical: {distancia_vertical}")
    print(f"   - proyectil_x: {proyectil_vertical.x}")
    print(f"   - jugador_x: {jugador.x}")
    
    try:
        prediccion = clf_movimiento.predict([datos_actuales])
        resultado = int(prediccion[0])
        
        decisiones = ["🔴 IZQUIERDA", "🔵 DERECHA", "⚪ QUIETO"]
        print(f"🌳 OUTPUT del árbol: {decisiones[resultado]} (clase {resultado})")
        
        # Mostrar probabilidades de cada clase si es posible
        try:
            probabilidades = clf_movimiento.predict_proba([datos_actuales])[0]
            print(f"🌳 Probabilidades: Izq={probabilidades[0]:.3f}, Der={probabilidades[1]:.3f}, Quieto={probabilidades[2]:.3f}")
        except:
            pass
        
        return resultado
        
    except Exception as e:
        print(f"🌳 ❌ Error en predicción: {e}")
        return 2

# NUEVO: Predicción K-Vecinos para movimiento
def knn_predict_movimiento():
    global knn_movimiento, proyectil_vertical, proyectil_vertical_activo
    global velocidad_proyectil_vertical, jugador  # 🔧 AGREGAR variables globales
    
    if knn_movimiento is None or not proyectil_vertical_activo:
        return 2
    
    datos_actuales = [[velocidad_proyectil_vertical, abs(jugador.y - proyectil_vertical.y),
                      proyectil_vertical.x, jugador.x]]
    prediccion = knn_movimiento.predict(datos_actuales)
    
    # 🎯 DEBUG: Descomenta para ver decisiones
    decision = int(prediccion[0])
    decision_texto = ["IZQUIERDA", "DERECHA", "QUIETO"][decision]
    print(f"🎯 KNN Movimiento decide: {decision_texto}")
    
    return decision


def reiniciar_escaladores():
    """NO hacer nada para mantener los modelos entrenados"""
    pass  # No limpiar escaladores para mantener modelos

def main():
    global salto, en_suelo, bala_disparada, modo_auto, clf, modo_mlp, modo_knn
    global movimiento_izquierda, movimiento_derecha

    reloj = pygame.time.Clock()
    mostrar_menu()
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_a or evento.key == pygame.K_LEFT:
                    movimiento_izquierda = True
                if evento.key == pygame.K_d or evento.key == pygame.K_RIGHT:
                    movimiento_derecha = True
                if evento.key == pygame.K_p:
                    pausa_juego()
                if evento.key == pygame.K_q:
                    print("Juego terminado.")
                    pygame.quit()
                    exit()
                if evento.key == pygame.K_s:
                    reiniciar_juego()
            
            if evento.type == pygame.KEYUP:
                if evento.key == pygame.K_a or evento.key == pygame.K_LEFT:
                    movimiento_izquierda = False
                if evento.key == pygame.K_d or evento.key == pygame.K_RIGHT:
                    movimiento_derecha = False

        if not pausa:
            if modo_auto:
                # Predicción para salto según el algoritmo elegido
                if modo_mlp:
                    prediccion_salto = mlp_predict()
                    prediccion_movimiento = mlp_predict_movimiento()
                elif modo_knn:  # NUEVO: K-Vecinos
                    prediccion_salto = knn_predict()
                    prediccion_movimiento = knn_predict_movimiento()
                else:  # Árboles de decisión
                    prediccion_salto = arbol_decision_predict()
                    prediccion_movimiento = arbol_decision_predict_movimiento()

                # Ejecutar salto
                if prediccion_salto == 1 and en_suelo:
                    salto = True
                    en_suelo = False

                # Ejecutar movimiento
                movimiento_izquierda = (prediccion_movimiento == 0)
                movimiento_derecha = (prediccion_movimiento == 1)

                if salto:
                    manejar_salto()
            else:
                if salto:
                    manejar_salto()

            # Manejar movimiento horizontal
            manejar_movimiento_horizontal()

            # Guardar datos si estamos en modo manual
            if not modo_auto:
                guardar_datos()

            # Actualizar el juego
            if not bala_disparada:
                disparar_bala()
            update()

        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()