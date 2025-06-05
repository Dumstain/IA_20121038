import pygame
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
import graphviz
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import time

# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Proyectiles Verticales - MEJORADO")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
ROJO = (255, 0, 0)
VERDE = (0, 255, 0)
AZUL = (0, 0, 255)

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
modo_knn = False

# 🔥 NUEVAS VARIABLES PARA RECOLECCIÓN INTELIGENTE DE DATOS
ultimo_guardado_salto = 0
ultimo_guardado_movimiento = 0
intervalo_guardado = 200  # Guardar cada 200ms máximo
ultima_accion_jugador = 0
situacion_peligrosa_activa = False

# Listas para guardar los datos de los modelos (MEJORADAS)
datos_modelo = []  # Para salto (sistema existente)
datos_movimiento = []  # Para movimiento (nuevo sistema)

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

# Variables para Scaler
scaler_salto = None
scaler_movimiento = None

# Variables globales para modelos
clf = None
clf_movimiento = None
mlp_model = None
mlp_model_movimiento = None
knn_salto = None
knn_movimiento = None

# 🔥 FUNCIONES MEJORADAS PARA FEATURES INTELIGENTES

def crear_features_salto_inteligentes():
    """Crea características más informativas para el salto"""
    global velocidad_bala, jugador, bala
    
    distancia_bala = abs(jugador.x - bala.x)
    
    # ⚡ NUEVAS FEATURES INTELIGENTES
    tiempo_hasta_impacto = distancia_bala / abs(velocidad_bala) if velocidad_bala != 0 else 999
    peligro_inminente = 1 if tiempo_hasta_impacto < 2.5 else 0
    distancia_normalizada = distancia_bala / w  # Normalizar por ancho pantalla
    velocidad_normalizada = abs(velocidad_bala) / 10  # Normalizar velocidad
    posicion_jugador_normalizada = jugador.x / w
    
    return [
        velocidad_normalizada,      # Velocidad normalizada
        distancia_normalizada,      # Distancia normalizada  
        tiempo_hasta_impacto,       # Tiempo crítico
        peligro_inminente,          # Situación de emergencia
        posicion_jugador_normalizada # Posición del jugador
    ]

def crear_features_movimiento_inteligentes():
    """Crea características más informativas para el movimiento"""
    global proyectil_vertical, proyectil_vertical_activo, jugador, velocidad_proyectil_vertical
    
    if not proyectil_vertical_activo:
        return None
    
    distancia_vertical = abs(jugador.y - proyectil_vertical.y)
    diferencia_horizontal = proyectil_vertical.x - jugador.x
    
    # ⚡ NUEVAS FEATURES INTELIGENTES
    tiempo_hasta_impacto = distancia_vertical / velocidad_proyectil_vertical if velocidad_proyectil_vertical > 0 else 999
    peligro_inminente = 1 if tiempo_hasta_impacto < 3.0 else 0
    diferencia_normalizada = diferencia_horizontal / w
    distancia_vertical_normalizada = distancia_vertical / h
    necesita_moverse = 1 if abs(diferencia_horizontal) < 25 else 0  # Zona de peligro
    direccion_necesaria = 1 if diferencia_horizontal > 0 else -1 if diferencia_horizontal < 0 else 0
    
    return [
        distancia_vertical_normalizada,  # Distancia vertical normalizada
        diferencia_normalizada,          # Diferencia horizontal normalizada
        tiempo_hasta_impacto,            # Tiempo hasta impacto
        peligro_inminente,               # Emergencia
        necesita_moverse,                # ¿Está en zona de peligro?
        direccion_necesaria              # Dirección recomendada
    ]

def detectar_situacion_peligrosa():
    """Detecta si hay una situación que requiere acción inmediata"""
    global situacion_peligrosa_activa, bala, jugador, proyectil_vertical, proyectil_vertical_activo
    
    peligro_bala = abs(jugador.x - bala.x) < 150 and bala_disparada
    peligro_proyectil = (proyectil_vertical_activo and 
                        abs(jugador.y - proyectil_vertical.y) < 120 and
                        abs(jugador.x - proyectil_vertical.x) < 30)
    
    situacion_peligrosa_activa = peligro_bala or peligro_proyectil
    return situacion_peligrosa_activa

def accion_realizada_por_jugador():
    """Detecta si el jugador realizó alguna acción recientemente"""
    global ultima_accion_jugador, salto, movimiento_izquierda, movimiento_derecha
    
    tiempo_actual = pygame.time.get_ticks()
    accion_reciente = (salto or movimiento_izquierda or movimiento_derecha)
    
    if accion_reciente:
        ultima_accion_jugador = tiempo_actual
        return True
    
    # Considerar "acción reciente" si fue hace menos de 500ms
    return tiempo_actual - ultima_accion_jugador < 500

# 🔥 FUNCIONES MEJORADAS PARA MANEJO DE MODELOS

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
            proyectil_vertical = pygame.Rect(posicion_centro, -20, 16, 16)
            proyectil_vertical_activo = True
            ultimo_disparo_vertical = tiempo_actual

# Función para mover proyectil vertical
def mover_proyectil_vertical():
    global proyectil_vertical, proyectil_vertical_activo
    
    if proyectil_vertical_activo:
        proyectil_vertical.y += velocidad_proyectil_vertical
        
        if proyectil_vertical.y > h:
            proyectil_vertical_activo = False

# Función para manejar el movimiento horizontal (nuevo sistema)
def manejar_movimiento_horizontal():
    global jugador, movimiento_izquierda, movimiento_derecha
    
    if movimiento_izquierda and jugador.x > posicion_min:
        jugador.x -= velocidad_movimiento
        jugador.x = max(jugador.x, posicion_min)
    elif movimiento_derecha and jugador.x < posicion_max:
        jugador.x += velocidad_movimiento
        jugador.x = min(jugador.x, posicion_max)
        
    if not movimiento_izquierda and not movimiento_derecha:
        velocidad_retorno = 10
        if jugador.x < posicion_centro - 2:
            jugador.x += velocidad_retorno
            jugador.x = min(jugador.x, posicion_centro)
        elif jugador.x > posicion_centro + 2:
            jugador.x -= velocidad_retorno
            jugador.x = max(jugador.x, posicion_centro)
        else:
            jugador.x = posicion_centro

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

    # 🔥 MOSTRAR INFORMACIÓN DE DEBUG EN PANTALLA
    if modo_auto:
        mostrar_info_debug()

    # Colisión entre la bala horizontal y el jugador
    if jugador.colliderect(bala):
        print("Colisión con bala horizontal!")
        reiniciar_juego()

    # Colisión entre proyectil vertical y el jugador
    if proyectil_vertical_activo and jugador.colliderect(proyectil_vertical):
        print("Colisión con proyectil vertical!")
        reiniciar_juego()

def mostrar_info_debug():
    """Muestra información de debug en pantalla durante modo automático"""
    global pantalla, fuente
    
    # Detectar situación actual
    detectar_situacion_peligrosa()
    
    # Información de estado
    y_offset = 10
    textos_debug = [
        f"Datos Salto: {len(datos_modelo)}",
        f"Datos Movimiento: {len(datos_movimiento)}",
        f"Peligro: {'SÍ' if situacion_peligrosa_activa else 'NO'}",
        f"Algoritmo: {'MLP' if modo_mlp else 'KNN' if modo_knn else 'Árbol'}"
    ]
    
    for texto in textos_debug:
        superficie_texto = fuente.render(texto, True, BLANCO)
        pantalla.blit(superficie_texto, (10, y_offset))
        y_offset += 25

# 🔥 FUNCIÓN MEJORADA PARA GUARDAR DATOS CON LÓGICA INTELIGENTE
def guardar_datos_inteligente():
    """Guarda datos solo cuando es relevante y útil"""
    global ultimo_guardado_salto, ultimo_guardado_movimiento
    global datos_modelo, datos_movimiento
    
    tiempo_actual = pygame.time.get_ticks()
    
    # 🎯 GUARDAR DATOS DE SALTO
    if (detectar_situacion_peligrosa() or accion_realizada_por_jugador() or 
        tiempo_actual - ultimo_guardado_salto > intervalo_guardado):
        
        features_salto = crear_features_salto_inteligentes()
        if features_salto:
            salto_hecho = 1 if salto else 0
            datos_modelo.append(features_salto + [salto_hecho])
            ultimo_guardado_salto = tiempo_actual
    
    # 🎯 GUARDAR DATOS DE MOVIMIENTO  
    if proyectil_vertical_activo and (detectar_situacion_peligrosa() or accion_realizada_por_jugador() or
                                     tiempo_actual - ultimo_guardado_movimiento > intervalo_guardado):
        
        features_movimiento = crear_features_movimiento_inteligentes()
        if features_movimiento:
            # Determinar movimiento realizado
            movimiento_hecho = 2  # Sin movimiento por defecto
            if movimiento_izquierda:
                movimiento_hecho = 0  # Izquierda
            elif movimiento_derecha:
                movimiento_hecho = 1  # Derecha
                
            datos_movimiento.append(features_movimiento + [movimiento_hecho])
            ultimo_guardado_movimiento = tiempo_actual

# Función para pausar el juego y guardar los datos
def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("🔄 Juego pausado.")
        diagnosticar_datos()
    else:
        print("▶️ Juego reanudado.")

def diagnosticar_datos():
    """Diagnóstica la calidad de los datos recolectados"""
    print("\n📊 DIAGNÓSTICO DE DATOS:")
    print(f"   Datos de salto: {len(datos_modelo)}")
    print(f"   Datos de movimiento: {len(datos_movimiento)}")
    
    if len(datos_movimiento) > 0:
        df_mov = pd.DataFrame(datos_movimiento)
        if len(df_mov.columns) > 6:  # Última columna es la acción
            distribucion = df_mov.iloc[:, -1].value_counts().sort_index()
            print("   Distribución de acciones de movimiento:")
            acciones = ["Izquierda", "Derecha", "Quieto"]
            for i, cantidad in distribucion.items():
                if i < len(acciones):
                    print(f"      {acciones[int(i)]}: {cantidad} ({cantidad/len(datos_movimiento)*100:.1f}%)")

# 🔥 FUNCIONES DE ENTRENAMIENTO MEJORADAS CON TODAS LAS OPTIMIZACIONES

def arbolDec():
    """Entrena árbol de decisión OPTIMIZADO para salto"""
    global clf, modelos_entrenados
    
    if not modelos_entrenados['arboles_salto']:
        if len(datos_modelo) < 5:
            print("❌ Necesitas al menos 20 ejemplos para entrenar el árbol de salto")
            return
            
        print("🌳 Entrenando Árbol de Decisión OPTIMIZADO para SALTO...")
        
        # Preparar datos con features mejoradas
        dataset = pd.DataFrame(datos_modelo)
        X = dataset.iloc[:, :-1].values  # Todas las features menos la última
        y = dataset.iloc[:, -1].values   # Última columna es el target
        
        print(f"   📊 Datos: {len(X)} ejemplos con {X.shape[1]} características")
        
        # 🎯 CONFIGURACIÓN OPTIMIZADA DEL ÁRBOL
        clf = DecisionTreeClassifier(
            max_depth=6,                    # Menos profundidad = menos overfitting
            min_samples_split=max(10, len(X)//10),  # Mínimo 10 ejemplos para dividir
            min_samples_leaf=max(3, len(X)//20),    # Mínimo 3 ejemplos por hoja
            class_weight='balanced',        # 🔥 BALANCEO AUTOMÁTICO DE CLASES
            random_state=42,
            criterion='gini'
        )
        
        # Entrenar con validación
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluar rendimiento
        precision_train = clf.score(X_train, y_train)
        precision_test = clf.score(X_test, y_test)
        
        print(f"   ✅ Precisión entrenamiento: {precision_train:.3f}")
        print(f"   ✅ Precisión validación: {precision_test:.3f}")
        print(f"   🌿 Profundidad real: {clf.get_depth()}")
        print(f"   🍃 Número de hojas: {clf.get_n_leaves()}")
        
        modelos_entrenados['arboles_salto'] = True
        guardar_modelos()
        print("   💾 Árbol de salto entrenado y guardado")
    else:
        print("🔄 Cargando Árbol de Decisión para salto ya entrenado...")
        cargar_modelos()

def arbolDec_movimiento():
    """Entrena árbol de decisión OPTIMIZADO para movimiento"""
    global clf_movimiento, modelos_entrenados
    
    if not modelos_entrenados['arboles_movimiento']:
        if len(datos_movimiento) < 30:
            print("❌ Necesitas al menos 30 ejemplos para entrenar el árbol de movimiento")
            return
            
        print("🌳 Entrenando Árbol de Decisión OPTIMIZADO para MOVIMIENTO...")
        
        # Preparar datos con features mejoradas
        dataset = pd.DataFrame(datos_movimiento)
        X = dataset.iloc[:, :-1].values  # Todas las features menos la última
        y = dataset.iloc[:, -1].values   # Última columna es el target
        
        print(f"   📊 Datos: {len(X)} ejemplos con {X.shape[1]} características")
        
        # Verificar distribución de clases
        unique, counts = np.unique(y, return_counts=True)
        print("   📈 Distribución de clases:")
        acciones = ["Izquierda", "Derecha", "Quieto"]
        for clase, cantidad in zip(unique, counts):
            if clase < len(acciones):
                print(f"      {acciones[int(clase)]}: {cantidad} ({cantidad/len(y)*100:.1f}%)")
        
                # 🎯 CONFIGURACIÓN OPTIMIZADA DEL ÁRBOL
        clf_movimiento = DecisionTreeClassifier(
            max_depth=6,  # Un poco más profundo
            min_samples_split=5,  # Menos restrictivo
            min_samples_leaf=1,   # Permite decisiones más específicas
            class_weight={0: 1.5, 1: 1.5, 2: 0.7}  # Favorecer movimiento sobre quieto
        )
        
        # Entrenar con validación
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        clf_movimiento.fit(X_train, y_train)
        
        # Evaluar rendimiento
        precision_train = clf_movimiento.score(X_train, y_train)
        precision_test = clf_movimiento.score(X_test, y_test)
        
        print(f"   ✅ Precisión entrenamiento: {precision_train:.3f}")
        print(f"   ✅ Precisión validación: {precision_test:.3f}")
        print(f"   🌿 Profundidad real: {clf_movimiento.get_depth()}")
        print(f"   🍃 Número de hojas: {clf_movimiento.get_n_leaves()}")
        
        # Mostrar importancia de características
        if hasattr(clf_movimiento, 'feature_importances_'):
            importancias = clf_movimiento.feature_importances_
            print("   🔍 Importancia de características:")
            nombres_features = ["Dist_Vert", "Diff_Horiz", "Tiempo", "Peligro", "Necesita_Mover", "Direccion"]
            for i, imp in enumerate(importancias):
                if i < len(nombres_features):
                    print(f"      {nombres_features[i]}: {imp:.3f}")
        
        modelos_entrenados['arboles_movimiento'] = True
        guardar_modelos()
        print("   💾 Árbol de movimiento entrenado y guardado")
    else:
        print("🔄 Cargando Árbol de Decisión para movimiento ya entrenado...")
        cargar_modelos()

def entrenar_mlp():
    """Entrena MLP OPTIMIZADO para salto"""
    global mlp_model, scaler_salto, modelos_entrenados
    
    if not modelos_entrenados['mlp_salto']:
        if len(datos_modelo) < 5:
            print("❌ Necesitas al menos 50 ejemplos para entrenar MLP de salto")
            return

        print(f"🧠 Entrenando MLP OPTIMIZADO para salto con {len(datos_modelo)} ejemplos...")
        
        dataset = pd.DataFrame(datos_modelo)
        X = dataset.iloc[:, :-1].values  
        y = dataset.iloc[:, -1].values

        # 🔥 NORMALIZACIÓN MEJORADA
        scaler_salto = StandardScaler()
        X_normalized = scaler_salto.fit_transform(X)
        
        print(f"   📊 Features normalizadas: {X.shape[1]} características")

        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.25, random_state=42)

        # 🧠 ARQUITECTURA OPTIMIZADA
        mlp_model = Sequential([
            Dense(16, input_dim=X.shape[1], activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        mlp_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )

        history = mlp_model.fit(
            X_train, y_train, 
            epochs=20,
            batch_size=min(16, len(X_train)//4),
            verbose=1,
            validation_split=0.2
        )

        # 🧪 EVALUACIÓN
        loss, accuracy = mlp_model.evaluate(X_test, y_test, verbose=1)
        
        modelos_entrenados['mlp_salto'] = True
        guardar_modelos()
        print(f"   ✅ MLP salto entrenado - Precisión: {accuracy:.3f}")
    else:
        print("🔄 Cargando MLP para salto ya entrenado...")
        cargar_modelos()

def entrenar_mlp_movimiento():
    """Entrena MLP OPTIMIZADO para movimiento"""
    global mlp_model_movimiento, scaler_movimiento, modelos_entrenados
    
    if not modelos_entrenados['mlp_movimiento']:
        if len(datos_movimiento) < 60:
            print("❌ Necesitas al menos 60 ejemplos para entrenar MLP de movimiento")
            return

        print(f"🧠 Entrenando MLP OPTIMIZADO para movimiento con {len(datos_movimiento)} ejemplos...")
        
        dataset = pd.DataFrame(datos_movimiento)
        X = dataset.iloc[:, :-1].values  
        y = dataset.iloc[:, -1].values
        
        print(f"   📊 Features: {X.shape[1]} características")

        # 🔥 NORMALIZACIÓN
        scaler_movimiento = StandardScaler()
        X_normalized = scaler_movimiento.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.25, random_state=42)

        # 🧠 ARQUITECTURA OPTIMIZADA PARA MULTICLASE
        mlp_model_movimiento = Sequential([
            Dense(20, input_dim=X.shape[1], activation='relu'),
            Dense(12, activation='relu'),
            Dense(6, activation='relu'),
            Dense(3, activation='softmax')  # 3 clases: izq, der, centro
        ])
        
        mlp_model_movimiento.compile(
            optimizer=Adam(learning_rate=0.001), 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

        history = mlp_model_movimiento.fit(
            X_train, y_train, 
            epochs=10,
            batch_size=min(16, len(X_train)//4), 
            verbose=1, 
            validation_split=0.2
        )

        loss, accuracy = mlp_model_movimiento.evaluate(X_test, y_test, verbose=1)
        
        modelos_entrenados['mlp_movimiento'] = True
        guardar_modelos()
        print(f"   ✅ MLP movimiento entrenado - Precisión: {accuracy:.3f}")
    else:
        print("🔄 Cargando MLP para movimiento ya entrenado...")
        cargar_modelos()

def entrenar_knn():
    """Entrena K-Vecinos OPTIMIZADO para salto"""
    global knn_salto, modelos_entrenados
    
    if not modelos_entrenados['knn_salto']:
        if len(datos_modelo) < 5:
            print("❌ Necesitas al menos 15 ejemplos para entrenar K-Vecinos de salto")
            return
        
        print("🎯 Entrenando K-Vecinos OPTIMIZADO para SALTO...")
        
        dataset = pd.DataFrame(datos_modelo)
        X = dataset.iloc[:, :-1].values  
        y = dataset.iloc[:, -1].values
        
        # 🎯 K OPTIMIZADO SEGÚN CANTIDAD DE DATOS
        k_optimo = max(3, min(9, len(datos_modelo) // 5))
        if k_optimo % 2 == 0:  # Asegurar que K sea impar para evitar empates
            k_optimo += 1
        
        knn_salto = KNeighborsClassifier(
            n_neighbors=k_optimo,
            weights='distance',  # Dar más peso a vecinos más cercanos
            metric='euclidean',
            algorithm='auto'     # Dejar que sklearn elija el mejor algoritmo
        )
        
        knn_salto.fit(X, y)
        modelos_entrenados['knn_salto'] = True
        guardar_modelos()
        print(f"   ✅ K-Vecinos salto entrenado con K={k_optimo}")
    else:
        print("🔄 Cargando K-Vecinos para salto ya entrenado...")
        cargar_modelos()

def entrenar_knn_movimiento():
    """Entrena K-Vecinos OPTIMIZADO para movimiento"""
    global knn_movimiento, modelos_entrenados
    
    if not modelos_entrenados['knn_movimiento']:
        if len(datos_movimiento) < 21:
            print("❌ Necesitas al menos 21 ejemplos para entrenar K-Vecinos de movimiento")
            return
        
        print("🎯 Entrenando K-Vecinos OPTIMIZADO para MOVIMIENTO...")
        
        dataset = pd.DataFrame(datos_movimiento)
        X = dataset.iloc[:, :-1].values  
        y = dataset.iloc[:, -1].values
        
        # 🎯 K OPTIMIZADO PARA MULTICLASE
        k_optimo = max(5, min(11, len(datos_movimiento) // 6))
        if k_optimo % 2 == 0:
            k_optimo += 1
        
        knn_movimiento = KNeighborsClassifier(
            n_neighbors=k_optimo,
            weights='distance',
            metric='euclidean',
            algorithm='auto'
        )
        
        knn_movimiento.fit(X, y)
        modelos_entrenados['knn_movimiento'] = True
        guardar_modelos()
        print(f"   ✅ K-Vecinos movimiento entrenado con K={k_optimo}")
    else:
        print("🔄 Cargando K-Vecinos para movimiento ya entrenado...")
        cargar_modelos()

# 🔥 FUNCIONES DE PREDICCIÓN MEJORADAS Y OPTIMIZADAS

def arbol_decision_predict():
    """Predicción OPTIMIZADA con árbol para salto"""
    global clf
    
    if clf is None:
        return 0
    
    features = crear_features_salto_inteligentes()
    if not features:
        return 0
    
    try:
        prediccion = clf.predict([features])
        probabilidades = clf.predict_proba([features])[0]
        
        # 🎯 DEBUG OPCIONAL
        if random.random() < 0.1:  # Solo mostrar 10% de las veces
            print(f"🌳 Salto - Prob No Saltar: {probabilidades[0]:.3f}, Prob Saltar: {probabilidades[1]:.3f}")
        
        return int(prediccion[0])
    except Exception as e:
        print(f"🌳 Error en predicción salto: {e}")
        return 0

def arbol_decision_predict_movimiento():
    """Predicción OPTIMIZADA con árbol para movimiento"""
    global clf_movimiento, proyectil_vertical_activo
    
    if clf_movimiento is None or not proyectil_vertical_activo:
        return 2
    
    features = crear_features_movimiento_inteligentes()
    if not features:
        return 2
    
    try:
        prediccion = clf_movimiento.predict([features])
        resultado = int(prediccion[0])
        
        # 🎯 DEBUG OPCIONAL
        if random.random() < 0.15:  # Solo mostrar 15% de las veces
            decisiones = ["🔴 IZQUIERDA", "🔵 DERECHA", "⚪ QUIETO"]
            print(f"🌳 Movimiento: {decisiones[resultado]}")
        
        return resultado
        
    except Exception as e:
        print(f"🌳 Error en predicción movimiento: {e}")
        return 2

def mlp_predict():
    """Predicción OPTIMIZADA con MLP para salto"""
    global mlp_model, scaler_salto
    
    if mlp_model is None or scaler_salto is None:
        return 0
    
    features = crear_features_salto_inteligentes()
    if not features:
        return 0
    
    try:
        datos_normalizados = scaler_salto.transform([features])
        prediccion = mlp_model.predict(datos_normalizados, verbose=0)
        probabilidad_salto = prediccion[0][0]
        
        # 🎯 LÓGICA DE DECISIÓN MEJORADA
        if probabilidad_salto > 0.45:
            decision = 1
        elif probabilidad_salto < 0.3:
            decision = 0
        else:
            decision = 1 if probabilidad_salto > 0.5 else 0
        
        # 🎯 DEBUG OPCIONAL
        if random.random() < 0.1:
            print(f"🧠 Salto MLP - Probabilidad: {probabilidad_salto:.3f} → {'SALTAR' if decision else 'NO SALTAR'}")
        
        return decision
    except Exception as e:
        print(f"🧠 Error en MLP salto: {e}")
        return 0

def mlp_predict_movimiento():
    """Predicción OPTIMIZADA con MLP para movimiento"""
    global mlp_model_movimiento, scaler_movimiento, proyectil_vertical_activo
    
    if mlp_model_movimiento is None or scaler_movimiento is None or not proyectil_vertical_activo:
        return 2
    
    features = crear_features_movimiento_inteligentes()
    if not features:
        return 2
    
    try:
        datos_normalizados = scaler_movimiento.transform([features])
        prediccion = mlp_model_movimiento.predict(datos_normalizados, verbose=0)
        
        probabilidades = prediccion[0]
        prob_izquierda, prob_derecha, prob_quieto = probabilidades
        
        # 🎯 LÓGICA DE DECISIÓN MEJORADA
        max_prob = max(probabilidades)
        decision = np.argmax(probabilidades)
        
        # Solo actuar si hay confianza alta
        if max_prob < 0.4:
            decision = 2  # Quedarse quieto si no hay confianza
        
        # 🎯 DEBUG OPCIONAL
        if random.random() < 0.15:
            decisiones = ["🔴 IZQUIERDA", "🔵 DERECHA", "⚪ QUIETO"]
            print(f"🧠 Mov MLP - {decisiones[decision]} (conf: {max_prob:.3f})")
        
        return int(decision)
    except Exception as e:
        print(f"🧠 Error en MLP movimiento: {e}")
        return 2

def knn_predict():
    """Predicción OPTIMIZADA con K-Vecinos para salto"""
    global knn_salto
    
    if knn_salto is None:
        return 0
    
    features = crear_features_salto_inteligentes()
    if not features:
        return 0
    
    try:
        prediccion = knn_salto.predict([features])
        probabilidades = knn_salto.predict_proba([features])[0]
        
        decision = int(prediccion[0])
        confianza = max(probabilidades)
        
        # 🎯 DEBUG OPCIONAL
        if random.random() < 0.1:
            print(f"🎯 KNN Salto - {'SALTAR' if decision else 'NO SALTAR'} (conf: {confianza:.3f})")
        
        return decision
    except Exception as e:
        print(f"🎯 Error en KNN salto: {e}")
        return 0

def knn_predict_movimiento():
    """Predicción OPTIMIZADA con K-Vecinos para movimiento"""
    global knn_movimiento, proyectil_vertical_activo
    
    if knn_movimiento is None or not proyectil_vertical_activo:
        return 2
    
    features = crear_features_movimiento_inteligentes()
    if not features:
        return 2
    
    try:
        prediccion = knn_movimiento.predict([features])
        probabilidades = knn_movimiento.predict_proba([features])[0]
        
        decision = int(prediccion[0])
        confianza = max(probabilidades)
        
        # Solo actuar si hay confianza razonable
        if confianza < 0.35:
            decision = 2
        
        # 🎯 DEBUG OPCIONAL
        if random.random() < 0.15:
            decisiones = ["🔴 IZQUIERDA", "🔵 DERECHA", "⚪ QUIETO"]
            print(f"🎯 KNN Mov - {decisiones[decision]} (conf: {confianza:.3f})")
        
        return decision
    except Exception as e:
        print(f"🎯 Error en KNN movimiento: {e}")
        return 2

# Función para mostrar el menú MEJORADA
def mostrar_menu():
    global menu_activo, modo_auto, datos_modelo, datos_movimiento, modo_mlp, modo_knn
    
    pantalla.fill(NEGRO)
    
    # 🎨 MENÚ MEJORADO CON MÁS INFORMACIÓN
    lineas_menu = [
        "🎮 MENÚ PRINCIPAL - JUEGO IA MEJORADO",
        "",
        "📊 DATOS RECOLECTADOS:",
        f"   Salto: {len(datos_modelo)} ejemplos",
        f"   Movimiento: {len(datos_movimiento)} ejemplos",
        "",
        "🤖 ALGORITMOS DISPONIBLES:",
        "'A' - Automático con Árboles de Decisión",
        "'R' - Automático con Redes Neuronales (MLP)",
        "'K' - Automático con K-Vecinos Más Cercanos",
        "",
        "🎮 OTROS CONTROLES:",
        "'M' - Modo Manual (recolectar datos)",
        "'G' - Mostrar gráficas de datos",
        "'D' - Diagnóstico de modelos",
        "'S' - Volver al menú (durante juego)",
        "'Q' - Salir del juego"
    ]
    
    x_inicial = w // 6
    y_inicial = 30
    espaciado = 22
    
    for i, linea in enumerate(lineas_menu):
        if linea.startswith("🎮") or linea.startswith("📊") or linea.startswith("🤖"):
            color = VERDE
        elif linea.startswith("'"):
            color = AZUL
        else:
            color = BLANCO
            
        texto = fuente.render(linea, True, color)
        pantalla.blit(texto, (x_inicial, y_inicial + i * espaciado))
    
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    if len(datos_modelo) < 10 or len(datos_movimiento) < 10:
                        print("❌ Datos insuficientes para árboles (necesitas 20+ salto, 30+ movimiento)")
                        continue
                    modo_auto = True
                    menu_activo = False
                    modo_mlp = False
                    modo_knn = False
                    arbolDec()
                    arbolDec_movimiento()
                    resetear_posiciones()
                    print("🌳 Modo Automático: Usando árboles de decisión optimizados...")
                
                elif evento.key == pygame.K_r:
                    if len(datos_modelo) < 5 or len(datos_movimiento) < 5:
                        print("❌ Datos insuficientes para MLP (necesitas 50+ salto, 60+ movimiento)")
                        continue
                    modo_auto = True
                    modo_mlp = True
                    modo_knn = False
                    menu_activo = False
                    entrenar_mlp()
                    entrenar_mlp_movimiento()
                    resetear_posiciones()
                    print("🧠 Modo Automático: Usando redes neuronales optimizadas...")
                
                elif evento.key == pygame.K_k:
                    if len(datos_modelo) < 15 or len(datos_movimiento) < 21:
                        print("❌ Datos insuficientes para K-Vecinos (necesitas 15+ salto, 21+ movimiento)")
                        continue
                    modo_auto = True
                    modo_mlp = False
                    modo_knn = True
                    menu_activo = False
                    entrenar_knn()
                    entrenar_knn_movimiento()
                    resetear_posiciones()
                    print("🎯 Modo Automático: Usando K-Vecinos optimizados...")
                        
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                    limpiar_modelos_completamente()
                    datos_modelo = []
                    datos_movimiento = []
                    resetear_posiciones()
                    print("🎮 Modo Manual - Recolectando nuevos datos con lógica inteligente...")
                    
                elif evento.key == pygame.K_d:
                    diagnosticar_modelos_completo()
                    
                elif evento.key == pygame.K_q:
                    print("👋 ¡Gracias por jugar!")
                    pygame.quit()
                    exit()
                elif evento.key == pygame.K_g:
                    graficar_mejorado()

def resetear_posiciones():
    """Resetea las posiciones del juego"""
    global jugador, salto, en_suelo, movimiento_izquierda, movimiento_derecha
    jugador.x = posicion_centro
    jugador.y = h - 100
    salto = False
    en_suelo = True
    movimiento_izquierda = False
    movimiento_derecha = False

def diagnosticar_modelos_completo():
    """Diagnóstico completo de todos los modelos"""
    print("\n🔍 DIAGNÓSTICO COMPLETO DE MODELOS:")
    print("=" * 50)
    
    # Datos recolectados
    print(f"📊 DATOS RECOLECTADOS:")
    print(f"   Ejemplos de salto: {len(datos_modelo)}")
    print(f"   Ejemplos de movimiento: {len(datos_movimiento)}")
    
    # Estado de modelos
    print(f"\n🤖 ESTADO DE MODELOS:")
    for modelo, entrenado in modelos_entrenados.items():
        estado = "✅ ENTRENADO" if entrenado else "❌ NO ENTRENADO"
        print(f"   {modelo}: {estado}")
    
    # Distribución de clases si hay datos
    if len(datos_movimiento) > 0:
        df_mov = pd.DataFrame(datos_movimiento)
        if len(df_mov.columns) > 6:
            distribucion = df_mov.iloc[:, -1].value_counts().sort_index()
            print(f"\n📈 DISTRIBUCIÓN DE MOVIMIENTOS:")
            acciones = ["Izquierda", "Derecha", "Quieto"]
            for i, cantidad in distribucion.items():
                if i < len(acciones):
                    porcentaje = cantidad/len(datos_movimiento)*100
                    print(f"   {acciones[int(i)]}: {cantidad} ({porcentaje:.1f}%)")
    
    print("=" * 50)

# Función para graficar MEJORADA
def graficar_mejorado():
    """Gráficas mejoradas con más información"""
    if len(datos_modelo) == 0 and len(datos_movimiento) == 0:
        print("❌ No hay datos para graficar")
        return
        
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('🎮 ANÁLISIS DE DATOS DEL JUEGO IA - VERSIÓN MEJORADA', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Datos de salto mejorados
    if len(datos_modelo) > 0:
        df_salto = pd.DataFrame(datos_modelo)
        
        ax1 = fig.add_subplot(231, projection="3d")
        colores = ['red' if x == 1 else 'blue' for x in df_salto.iloc[:, -1]]
        ax1.scatter(df_salto.iloc[:, 1], df_salto.iloc[:, 2], df_salto.iloc[:, -1], c=colores, alpha=0.7)
        ax1.set_xlabel("Distancia Normalizada")
        ax1.set_ylabel("Tiempo hasta Impacto")
        ax1.set_zlabel("Acción (0=No Saltar, 1=Saltar)")
        ax1.set_title("🦘 Datos de Salto 3D")
        
        # Histograma de acciones de salto
        ax2 = fig.add_subplot(232)
        salto_counts = df_salto.iloc[:, -1].value_counts()
        ax2.bar(['No Saltar', 'Saltar'], [salto_counts.get(0, 0), salto_counts.get(1, 0)], 
                color=['lightcoral', 'lightgreen'])
        ax2.set_title("📊 Distribución Acciones Salto")
        ax2.set_ylabel("Cantidad")
    
    # Gráficos de movimiento mejorados
    if len(datos_movimiento) > 0:
        df_mov = pd.DataFrame(datos_movimiento)
        
        # Scatter plot de posiciones
        ax3 = fig.add_subplot(233)
        colores_mov = ['red', 'blue', 'green']
        for accion in [0, 1, 2]:
            mask = df_mov.iloc[:, -1] == accion
            if mask.any():
                ax3.scatter(df_mov.loc[mask, df_mov.columns[1]], 
                           df_mov.loc[mask, df_mov.columns[0]], 
                           c=colores_mov[accion], 
                           label=['Izquierda', 'Derecha', 'Quieto'][accion],
                           alpha=0.7)
        ax3.set_xlabel("Diferencia Horizontal")
        ax3.set_ylabel("Distancia Vertical")
        ax3.set_title("🎯 Decisiones de Movimiento")
        ax3.legend()
        
        # Histograma de acciones de movimiento
        ax4 = fig.add_subplot(234)
        mov_counts = df_mov.iloc[:, -1].value_counts().sort_index()
        acciones = ['Izquierda', 'Derecha', 'Quieto']
        colores_barras = ['red', 'blue', 'green']
        bars = ax4.bar([acciones[int(i)] for i in mov_counts.index], 
                      mov_counts.values, 
                      color=[colores_barras[int(i)] for i in mov_counts.index])
        ax4.set_title("📊 Distribución Acciones Movimiento")
        ax4.set_ylabel("Cantidad")
        
        # Agregar porcentajes en las barras
        total = len(df_mov)
        for bar, count in zip(bars, mov_counts.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{count}\n({count/total*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        # Timeline de acciones
        ax5 = fig.add_subplot(235)
        ax5.plot(range(len(df_mov)), df_mov.iloc[:, -1], 'o-', markersize=3, alpha=0.7)
        ax5.set_xlabel("Momento en el tiempo")
        ax5.set_ylabel("Acción (0=Izq, 1=Der, 2=Quieto)")
        ax5.set_title("⏱️ Timeline de Decisiones")
        ax5.set_yticks([0, 1, 2])
        ax5.set_yticklabels(['Izq', 'Der', 'Quieto'])
        ax5.grid(True, alpha=0.3)
    
    # Estadísticas generales
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    stats_text = f"""
📈 ESTADÍSTICAS GENERALES

🦘 Datos de Salto: {len(datos_modelo)}
🎯 Datos de Movimiento: {len(datos_movimiento)}

🤖 Modelos Entrenados:
"""
    for modelo, entrenado in modelos_entrenados.items():
        if entrenado:
            stats_text += f"✅ {modelo.replace('_', ' ').title()}\n"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Función para reiniciar el juego tras la colisión MEJORADA
def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo, modo_auto
    global proyectil_vertical_activo, movimiento_izquierda, movimiento_derecha
    
    menu_activo = True
    resetear_posiciones()
    bala.x = w - 50
    nave.x, nave.y = w - 100, h - 100
    bala_disparada = False
    proyectil_vertical_activo = False
    
    if not modo_auto: 
        print(f"\n🎮 COLISIÓN - Regresando al menú")
        diagnosticar_datos()
    
    mostrar_menu()

def main():
    global salto, en_suelo, bala_disparada, modo_auto, modo_mlp, modo_knn
    global movimiento_izquierda, movimiento_derecha

    reloj = pygame.time.Clock()
    mostrar_menu()
    correr = True

    print("🚀 Juego iniciado con todas las mejoras implementadas!")

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
                    print("👋 ¡Gracias por jugar!")
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
                # 🎯 PREDICCIÓN OPTIMIZADA SEGÚN ALGORITMO ELEGIDO
                if modo_mlp:
                    prediccion_salto = mlp_predict()
                    prediccion_movimiento = mlp_predict_movimiento()
                elif modo_knn:
                    prediccion_salto = knn_predict()
                    prediccion_movimiento = knn_predict_movimiento()
                else:  # Árboles de decisión
                    prediccion_salto = arbol_decision_predict()
                    prediccion_movimiento = arbol_decision_predict_movimiento()

                # Ejecutar predicciones
                if prediccion_salto == 1 and en_suelo:
                    salto = True
                    en_suelo = False

                movimiento_izquierda = (prediccion_movimiento == 0)
                movimiento_derecha = (prediccion_movimiento == 1)

                if salto:
                    manejar_salto()
            else:
                # Modo manual
                if salto:
                    manejar_salto()
                
                # 🔥 GUARDAR DATOS CON LÓGICA INTELIGENTE
                guardar_datos_inteligente()

            # Manejar movimiento horizontal
            manejar_movimiento_horizontal()

            # Actualizar el juego
            if not bala_disparada:
                disparar_bala()
            update()

        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()