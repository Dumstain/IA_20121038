import pygame
import random
import csv  # Importar la librería CSV
import joblib # Importar la librería joblib

# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# Variables del jugador, bala, nave, fondo, etc.
jugador = None
bala = None
fondo = None
nave = None
menu = None

# Variables de salto
salto = False
salto_altura = 15  # Velocidad inicial de salto
gravedad = 1
en_suelo = True

# Variables de pausa y menú
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False  # Indica si el modo de juego es automático

# Lista para guardar los datos de velocidad, distancia y salto (target)
datos_modelo = []

# --- INICIO CÓDIGO AÑADIDO: Cargar Modelo ---
modelo_cargado = None
nombre_modelo = 'decision_tree_model.joblib'
try:
    modelo_cargado = joblib.load(nombre_modelo)
    print(f"Modelo de ML '{nombre_modelo}' cargado exitosamente.")
except FileNotFoundError:
    print(f"Advertencia: No se encontró el archivo del modelo ({nombre_modelo}). El modo automático no funcionará.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
# --- FIN CÓDIGO AÑADIDO ---

# Cargar las imágenes
try:
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

except pygame.error as e:
    print(f"Error al cargar las imágenes: {e}")
    print("Asegúrate de que la carpeta 'assets' esté en el mismo directorio que game.py y contenga las imágenes necesarias.")
    pygame.quit()
    exit()


# Escalar la imagen de fondo para que coincida con el tamaño de la pantalla
fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear el rectángulo del jugador y de la bala
jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)  # Tamaño del menú

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0

# Variables para la bala
velocidad_bala = -10  # Velocidad de la bala hacia la izquierda
bala_disparada = False

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

# --- INICIO CÓDIGO AÑADIDO: Función para guardar datos en CSV ---
def guardar_csv(nombre_archivo='phaser3.csv', modo_escritura='w'):
    """Guarda la lista datos_modelo en un archivo CSV."""
    global datos_modelo
    if not datos_modelo: # No guardar si no hay datos
        print("No hay datos nuevos para guardar.")
        return
    try:
        with open(nombre_archivo, modo_escritura, newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            # Escribir encabezado solo si se sobrescribe ('w') o el archivo está vacío
            # Nota: Una comprobación más robusta si el archivo existe y está vacío sería mejor para el modo 'a'
            if modo_escritura == 'w':
                escritor_csv.writerow(['velocidad_bala', 'distancia', 'salto'])
            # Escribir los datos
            escritor_csv.writerows(datos_modelo)
        print(f"Datos guardados exitosamente en {nombre_archivo}")
        # Limpiar datos después de guardar (importante si no quieres acumular entre sesiones)
        datos_modelo = []
    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")
# --- FIN CÓDIGO AÑADIDO ---

# Función para disparar la bala
def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        # Asegúrate que la velocidad sea negativa
        velocidad_bala = random.randint(-15, -5) # Rango de velocidad más amplio? Ajusta si es necesario
        bala_disparada = True

# Función para reiniciar la posición de la bala
def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50  # Reiniciar la posición de la bala
    bala_disparada = False

# Función para manejar el salto
def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura  # Mover al jugador hacia arriba
        salto_altura -= gravedad  # Aplicar gravedad (reduce la velocidad del salto)

        # Si el jugador llega al suelo, detener el salto
        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 15  # Restablecer la velocidad de salto
            en_suelo = True

# Función para actualizar el juego
def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2

    # Mover el fondo
    fondo_x1 -= 1
    fondo_x2 -= 1

    # Si el primer fondo sale de la pantalla, lo movemos detrás del segundo
    if fondo_x1 <= -w:
        fondo_x1 = w

    # Si el segundo fondo sale de la pantalla, lo movemos detrás del primero
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

    # Mover y dibujar la bala
    if bala_disparada:
        bala.x += velocidad_bala

    # Si la bala sale de la pantalla, reiniciar su posición
    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    # Colisión entre la bala y el jugador
    if jugador.colliderect(bala):
        print("Colisión detectada!")
        # --- MODIFICADO: Guardar datos antes de reiniciar ---
        if not modo_auto: # Solo guardar si estábamos en modo manual
            print("Guardando datos antes de reiniciar...")
            guardar_csv('phaser3.csv', 'w') # Usar 'w' para sobrescribir, 'a' para añadir
        # --- FIN MODIFICADO ---
        reiniciar_juego()  # Mostrar menú y reiniciar estado

# Función para recolectar datos del modelo en modo manual (cambié el nombre)
def recolectar_datos():
    global jugador, bala, velocidad_bala, salto
    # Solo recolectar si la bala está disparada (tiene velocidad y posición relevantes)
    if bala_disparada:
        distancia = abs(jugador.x - bala.x)
        # Determinar si el salto está ACTIVO en este frame preciso.
        # Si se quiere registrar si el jugador DECIDIÓ saltar, la lógica sería diferente (basada en el input)
        salto_hecho = 1 if salto else 0
        # Guardar velocidad de la bala, distancia al jugador y si está saltando o no
        datos_modelo.append([velocidad_bala, distancia, salto_hecho])

# Función para pausar el juego
def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado.")
        # Opcional: Guardar datos al pausar si estás en modo manual
        # if not modo_auto:
        #    guardar_csv('phaser3.csv', 'a') # Usar 'a' para añadir datos al pausar
    else:
        print("Juego reanudado.")

# Función para mostrar el menú y seleccionar el modo de juego
def mostrar_menu():
    global menu_activo, modo_auto
    pantalla.fill(NEGRO) # Limpiar pantalla para el menú
    texto = fuente.render("Presiona 'A' para Auto, 'M' para Manual, o 'Q' para Salir", True, BLANCO)
    pantalla.blit(texto, (w // 4, h // 2))
    pygame.display.flip()

    menu_activo_local = True
    while menu_activo_local:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                # --- MODIFICADO: Guardar datos al salir desde el menú ---
                if not modo_auto:
                    guardar_csv('phaser3.csv', 'w') # Guardar datos de la última sesión manual
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    if modelo_cargado: # Solo permitir modo auto si el modelo cargó
                        modo_auto = True
                        menu_activo = False
                        menu_activo_local = False
                        print("Modo Automático seleccionado.")
                    else:
                        print("El modelo no está cargado. No se puede iniciar el modo Automático.")
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                    menu_activo_local = False
                    print("Modo Manual seleccionado.")
                elif evento.key == pygame.K_q:
                    # --- MODIFICADO: Guardar datos al salir desde el menú ---
                    if not modo_auto:
                        guardar_csv('phaser3.csv', 'w') # Guardar datos de la última sesión manual
                    print("Juego terminado.")
                    pygame.quit()
                    exit()

# Función para reiniciar el juego tras la colisión
def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo
    menu_activo = True  # Activar de nuevo el menú
    # Reiniciar estado del juego
    jugador.x, jugador.y = 50, h - 100
    bala.x = w - 50
    nave.x, nave.y = w - 100, h - 100
    bala_disparada = False
    salto = False
    salto_altura = 15 # Restablecer velocidad de salto
    en_suelo = True
    # Los datos ya se guardaron antes de llamar a esta función
    mostrar_menu()  # Mostrar el menú de nuevo para seleccionar modo

def main():
    global salto, en_suelo, bala_disparada, datos_modelo

    reloj = pygame.time.Clock()
    mostrar_menu()  # Mostrar el menú al inicio para la primera selección
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                # Salto manual solo si no está en modo automático y no está pausado
                if evento.key == pygame.K_SPACE and en_suelo and not pausa and not modo_auto:
                    salto = True
                    en_suelo = False
                    # --- AÑADIDO: Registrar el intento de salto manual para datos ---
                    # Es importante decidir QUÉ quieres registrar: la acción (presionar espacio)
                    # o el estado (estar saltando). Aquí registramos la acción.
                    # La función recolectar_datos() registrará el estado.
                    # Podrías añadir otra columna si quieres ambos.
                if evento.key == pygame.K_p:
                    pausa_juego()
                if evento.key == pygame.K_q:
                    correr = False

        if not pausa and not menu_activo: # Solo correr lógica si no está pausado NI en el menú

            # --- Lógica de Decisión y Recolección ---
            if modo_auto:
                # --- INICIO LÓGICA MODO AUTO ---
                if modelo_cargado and bala_disparada and en_suelo:
                    velocidad_actual = velocidad_bala
                    distancia_actual = abs(jugador.x - bala.x)
                    caracteristicas_actuales = [[velocidad_actual, distancia_actual]]

                    try:
                        prediccion = modelo_cargado.predict(caracteristicas_actuales)
                        if prediccion[0] == 1: # Predecir 'Saltar'
                            salto = True
                            en_suelo = False
                    except Exception as e:
                        print(f"Error durante la predicción: {e}")
                # --- FIN LÓGICA MODO AUTO ---
            else: # Modo Manual
                recolectar_datos() # Recolectar datos solo en modo manual

            # --- Lógica Común (Movimiento, Disparo, Colisión) ---
            if salto: # Actualizar estado del salto (si está activo)
                manejar_salto()

            if not bala_disparada: # Disparar si no hay bala
                disparar_bala()

            update() # Actualizar posiciones, dibujar, manejar colisiones

            # --- Lógica de Dibujo ---
            # (El dibujo se hace dentro de update() en este código)

        elif menu_activo:
            # Si el menú está activo, no hacer nada más que esperar input en mostrar_menu()
            # (La llamada a mostrar_menu() ahora maneja su propio bucle de eventos)
            pass # Podrías dibujar un estado de 'menú' aquí si no fuera bloqueante

        # Actualizar la pantalla fuera del if not pausa
        pygame.display.flip()
        reloj.tick(30)  # Limitar el juego a 30 FPS

    # --- Guardar datos al salir del bucle principal ---
    if not modo_auto: # Solo guardar si la última sesión fue manual
        guardar_csv('phaser3.csv', 'w') # O 'a' si prefieres añadir a datos existentes
    print("Juego terminado. Datos finales:", datos_modelo) # Muestra datos si no se guardaron
    pygame.quit()

if __name__ == "__main__":
    main()