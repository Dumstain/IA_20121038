import pygame
import random
import csv
import os

# Inicializar Pygame
pygame.init()
pygame.mixer.init()

# Dimensiones de la pantalla
ANCHO_PANTALLA, ALTO_PANTALLA = 800, 400
pantalla = pygame.display.set_mode((ANCHO_PANTALLA, ALTO_PANTALLA))
pygame.display.set_caption("🎮 MODO RECOLECCIÓN DE DATOS - Juega para entrenar IA")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
AZUL_CLARO = (173, 216, 230)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
AMARILLO = (255, 255, 0)

# --- Constantes del Juego ---
JUGADOR_VELOCIDAD_HORIZONTAL = 5
RANGO_MOVIMIENTO_JUGADOR_RADIO = 35
BALA_VERTICAL_VELOCIDAD = 4
DESPLAZAMIENTO_X_BALA_VERTICAL = 15
ALTURA_SALTO_INICIAL = 15
GRAVEDAD = 1
POS_Y_TOP_JUGADOR_SUELO = ALTO_PANTALLA - 100
POS_Y_TOP_BALA_HORIZONTAL = ALTO_PANTALLA - 90

# --- Variables Globales del Juego ---
jugador_rect = None; jugador_frames = []
current_frame_jugador = 0; frame_speed_jugador = 10; frame_count_jugador = 0
jugador_velocidad_x_actual = 0
X_SPAWN_JUGADOR_CENTRO_INICIAL = 0
LIMITE_JUGADOR_CENTRO_IZQ = 0; LIMITE_JUGADOR_CENTRO_DER = 0

# Salto
salto_activo = False; salto_altura_actual = ALTURA_SALTO_INICIAL; en_suelo = True

# Bala Horizontal
bala_horizontal_rect = None; bala_horizontal_img_asset = None
velocidad_bala_horizontal = -10; bala_horizontal_disparada = False

# Bala Vertical
bala_vertical_rect = None; bala_vertical_img_asset = None; bala_vertical_disparada = False

# UFO
ufo_rect = None; ufo_img_asset = None

# Otros elementos
fondo_img = None; fuente_menu = pygame.font.SysFont('Arial', 28); fuente_info = pygame.font.SysFont('Arial', 20)

# UI y Estado del Juego
pausa = False; corriendo = True; musica_iniciada = False

# Datos para IA
datos_para_csv = []
CABECERAS_CSV = [
    'jugador_x_centro', 'jugador_y_top', 'jugador_velocidad_x', 'en_suelo',
    'bala_h_activa', 'bala_h_x_centro', 'bala_h_y_centro', 'bala_h_velocidad_x',
    'dist_jugador_bala_h_x', 'dist_jugador_bala_h_y',
    'bala_v_activa', 'bala_v_x_centro', 'bala_v_y_centro', 'bala_v_velocidad_y',
    'dist_jugador_bala_v_x', 'dist_jugador_bala_v_y',
    'accion_horizontal_comandada', 'accion_salto_comandado'
]
accion_horizontal_comandada_manual = 0
accion_salto_comandado_manual = 0

# Estadísticas para mejor recolección
stats = {
    'movimientos_izq': 0,
    'movimientos_der': 0, 
    'saltos': 0,
    'frames_totales': 0,
    'colisiones': 0
}

# Cargar las imágenes (con fallback a rectángulos de colores)
try:
    jugador_frames_assets = [pygame.image.load(f'assets/sprites/mono_frame_{i+1}.png') for i in range(4)]
    jugador_frames = jugador_frames_assets
    bala_horizontal_img_asset = pygame.image.load('assets/sprites/purple_ball.png')
    bala_vertical_img_asset = pygame.image.load('assets/sprites/purple_ball.png')
    ufo_img_asset = pygame.image.load('assets/game/ufo.png')
    ufo_img_asset = pygame.transform.scale(ufo_img_asset, (64, 32))
    fondo_img_asset = pygame.image.load('assets/game/fondo2.png')
    fondo_img = pygame.transform.scale(fondo_img_asset, (ANCHO_PANTALLA, ALTO_PANTALLA))
    assets_cargados = True
except:
    print("⚠️  No se pudieron cargar las imágenes, usando formas simples")
    assets_cargados = False
    # Crear sprites simples
    jugador_frames = [pygame.Surface((40, 60)) for _ in range(4)]
    for frame in jugador_frames:
        frame.fill(VERDE)
    bala_horizontal_img_asset = pygame.Surface((20, 20))
    bala_horizontal_img_asset.fill(ROJO)
    bala_vertical_img_asset = pygame.Surface((20, 20))
    bala_vertical_img_asset.fill(ROJO)
    ufo_img_asset = pygame.Surface((64, 32))
    ufo_img_asset.fill(AMARILLO)
    fondo_img = pygame.Surface((ANCHO_PANTALLA, ALTO_PANTALLA))
    fondo_img.fill(AZUL_CLARO)

# --- Inicialización ---
jugador_sprite_inicial = jugador_frames[0]
jugador_rect = jugador_sprite_inicial.get_rect(topleft=(50, POS_Y_TOP_JUGADOR_SUELO))
X_SPAWN_JUGADOR_CENTRO_INICIAL = jugador_rect.centerx
LIMITE_JUGADOR_CENTRO_IZQ = X_SPAWN_JUGADOR_CENTRO_INICIAL - RANGO_MOVIMIENTO_JUGADOR_RADIO
LIMITE_JUGADOR_CENTRO_DER = X_SPAWN_JUGADOR_CENTRO_INICIAL + RANGO_MOVIMIENTO_JUGADOR_RADIO

bala_horizontal_rect = bala_horizontal_img_asset.get_rect(topleft=(ANCHO_PANTALLA - 50, POS_Y_TOP_BALA_HORIZONTAL))
bala_vertical_rect = bala_vertical_img_asset.get_rect(bottom=0)
ufo_rect = ufo_img_asset.get_rect()
ufo_rect.centery = POS_Y_TOP_BALA_HORIZONTAL + bala_horizontal_rect.height // 2
ufo_rect.centerx = ANCHO_PANTALLA - 30

fondo_x1 = 0; fondo_x2 = ANCHO_PANTALLA

# --- Funciones del Juego ---
def guardar_datos_csv_ia(nombre_archivo='datos_combinados_ia.csv', modo_escritura='a'):
    global datos_para_csv
    if not datos_para_csv: return
    try:
        archivo_existe_y_tiene_contenido = False
        try:
            with open(nombre_archivo, 'r') as f_check:
                if f_check.readline(): archivo_existe_y_tiene_contenido = True
        except FileNotFoundError: pass
        with open(nombre_archivo, modo_escritura, newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            if modo_escritura == 'w' or not archivo_existe_y_tiene_contenido:
                escritor_csv.writerow(CABECERAS_CSV)
            escritor_csv.writerows(datos_para_csv)
        print(f"💾 Datos guardados: {len(datos_para_csv)} filas en {nombre_archivo}")
        datos_para_csv = []
    except Exception as e: print(f"❌ Error al guardar CSV: {e}")

def disparar_bala_horizontal():
    global bala_horizontal_disparada, velocidad_bala_horizontal, bala_horizontal_rect
    if not bala_horizontal_disparada:
        velocidad_bala_horizontal = random.randint(-15, -7)
        bala_horizontal_rect.x = ANCHO_PANTALLA - 50
        bala_horizontal_rect.top = POS_Y_TOP_BALA_HORIZONTAL
        ufo_rect.centerx = bala_horizontal_rect.centerx + ufo_rect.width / 2 + 5
        bala_horizontal_disparada = True

def reset_bala_horizontal(): 
    global bala_horizontal_disparada; bala_horizontal_disparada = False

def disparar_bala_vertical():
    global bala_vertical_disparada, bala_vertical_rect
    if not bala_vertical_disparada:
        modificador_direccion = random.choice([-1, 1])
        bala_vertical_rect.centerx = X_SPAWN_JUGADOR_CENTRO_INICIAL + (modificador_direccion * DESPLAZAMIENTO_X_BALA_VERTICAL)
        bala_vertical_rect.bottom = 0
        bala_vertical_disparada = True

def reset_bala_vertical(): 
    global bala_vertical_disparada; bala_vertical_disparada = False

def actualizar_posicion_jugador():
    global jugador_rect, salto_activo, salto_altura_actual, en_suelo, jugador_velocidad_x_actual
    jugador_rect.centerx += jugador_velocidad_x_actual
    if jugador_rect.centerx < LIMITE_JUGADOR_CENTRO_IZQ: jugador_rect.centerx = LIMITE_JUGADOR_CENTRO_IZQ
    if jugador_rect.centerx > LIMITE_JUGADOR_CENTRO_DER: jugador_rect.centerx = LIMITE_JUGADOR_CENTRO_DER
    if salto_activo:
        jugador_rect.y -= salto_altura_actual; salto_altura_actual -= GRAVEDAD
        if jugador_rect.top >= POS_Y_TOP_JUGADOR_SUELO:
            jugador_rect.top = POS_Y_TOP_JUGADOR_SUELO; salto_activo = False; en_suelo = True

def actualizar_estado_general():
    global current_frame_jugador, frame_count_jugador, fondo_x1, fondo_x2, stats

    stats['frames_totales'] += 1
    
    if assets_cargados:
        fondo_x1 -= 1; fondo_x2 -= 1
        if fondo_x1 <= -ANCHO_PANTALLA: fondo_x1 = ANCHO_PANTALLA
        if fondo_x2 <= -ANCHO_PANTALLA: fondo_x2 = ANCHO_PANTALLA
        frame_count_jugador += 1
        if frame_count_jugador >= frame_speed_jugador:
            current_frame_jugador = (current_frame_jugador + 1) % len(jugador_frames)
            frame_count_jugador = 0

    if bala_horizontal_disparada:
        bala_horizontal_rect.x += velocidad_bala_horizontal
        if bala_horizontal_rect.right < 0: reset_bala_horizontal()
    else: disparar_bala_horizontal()
    
    if bala_vertical_disparada:
        bala_vertical_rect.y += BALA_VERTICAL_VELOCIDAD
        if bala_vertical_rect.top > ALTO_PANTALLA: reset_bala_vertical()
    else: disparar_bala_vertical()
    
    # Colisiones
    if bala_horizontal_disparada and jugador_rect.colliderect(bala_horizontal_rect):
        print("💥 Colisión HORIZONTAL!")
        stats['colisiones'] += 1
        guardar_datos_csv_ia()
        mostrar_estadisticas()
        reiniciar_partida()
    if bala_vertical_disparada and jugador_rect.colliderect(bala_vertical_rect):
        print("💥 Colisión VERTICAL!")
        stats['colisiones'] += 1
        guardar_datos_csv_ia()
        mostrar_estadisticas()
        reiniciar_partida()

def dibujar_elementos():
    if assets_cargados:
        pantalla.blit(fondo_img, (fondo_x1, 0))
        pantalla.blit(fondo_img, (fondo_x2, 0))
        pantalla.blit(jugador_frames[current_frame_jugador], jugador_rect)
        pantalla.blit(ufo_img_asset, ufo_rect)
        if bala_horizontal_disparada: pantalla.blit(bala_horizontal_img_asset, bala_horizontal_rect)
        if bala_vertical_disparada: pantalla.blit(bala_vertical_img_asset, bala_vertical_rect)
    else:
        pantalla.blit(fondo_img, (0, 0))
        pantalla.blit(jugador_frames[current_frame_jugador], jugador_rect)  
        pantalla.blit(ufo_img_asset, ufo_rect)
        if bala_horizontal_disparada: pantalla.blit(bala_horizontal_img_asset, bala_horizontal_rect)
        if bala_vertical_disparada: pantalla.blit(bala_vertical_img_asset, bala_vertical_rect)
    
    # Dibujar HUD con estadísticas
    dibujar_hud()
    
    if pausa:
        texto_pausa = fuente_menu.render("⏸️  PAUSA", True, BLANCO)
        pantalla.blit(texto_pausa, texto_pausa.get_rect(center=(ANCHO_PANTALLA//2, ALTO_PANTALLA//2)))

def dibujar_hud():
    """Dibuja información útil en pantalla"""
    y_offset = 10
    
    # Título
    titulo = fuente_info.render("🎮 MODO RECOLECCIÓN DE DATOS", True, BLANCO)
    pantalla.blit(titulo, (10, y_offset))
    y_offset += 30
    
    # Estadísticas de acciones
    texto_mov = fuente_info.render(f"⬅️ {stats['movimientos_izq']} | ➡️ {stats['movimientos_der']} | ⬆️ {stats['saltos']}", True, BLANCO)
    pantalla.blit(texto_mov, (10, y_offset))
    y_offset += 25
    
    # Datos recolectados
    datos_count = len(datos_para_csv)
    texto_datos = fuente_info.render(f"📊 Datos: {datos_count} | Colisiones: {stats['colisiones']}", True, BLANCO)
    pantalla.blit(texto_datos, (10, y_offset))
    y_offset += 25
    
    # Instrucciones
    instrucciones = [
        "⬅️➡️ Moverse | ESPACIO Saltar | P Pausa | ESC Reiniciar | Q Salir"
    ]
    for instr in instrucciones:
        texto_instr = pygame.font.SysFont('Arial', 16).render(instr, True, AMARILLO)
        pantalla.blit(texto_instr, (10, y_offset))
        y_offset += 20

def recolectar_datos_para_ia():
    global datos_para_csv, jugador_rect, jugador_velocidad_x_actual, en_suelo
    global bala_horizontal_rect, bala_horizontal_disparada, velocidad_bala_horizontal
    global bala_vertical_rect, bala_vertical_disparada, BALA_VERTICAL_VELOCIDAD
    global accion_horizontal_comandada_manual, accion_salto_comandado_manual

    px_c = jugador_rect.centerx; py_t = jugador_rect.top; pvx = jugador_velocidad_x_actual
    p_en_s_feature = 1 if en_suelo else 0
    
    bh_a = 1 if bala_horizontal_disparada else 0
    bhx_c = bala_horizontal_rect.centerx if bh_a else -1000; bhy_c = bala_horizontal_rect.centery if bh_a else -1000
    bh_vx = velocidad_bala_horizontal if bh_a else 0
    d_j_bh_x = bhx_c - px_c if bh_a else ANCHO_PANTALLA; d_j_bh_y = bhy_c - py_t if bh_a else ALTO_PANTALLA
    
    bv_a = 1 if bala_vertical_disparada else 0
    bvx_c = bala_vertical_rect.centerx if bv_a else -1000; bvy_c = bala_vertical_rect.centery if bv_a else -1000
    bv_vy = BALA_VERTICAL_VELOCIDAD if bv_a else 0
    d_j_bv_x = bvx_c - px_c if bv_a else ANCHO_PANTALLA; d_j_bv_y = bvy_c - py_t if bv_a else ALTO_PANTALLA
    
    fila_datos = [
        px_c, py_t, pvx, p_en_s_feature,
        bh_a, bhx_c, bhy_c, bh_vx, d_j_bh_x, d_j_bh_y,
        bv_a, bvx_c, bvy_c, bv_vy, d_j_bv_x, d_j_bv_y,
        accion_horizontal_comandada_manual, accion_salto_comandado_manual
    ]
    datos_para_csv.append(fila_datos)

def mostrar_estadisticas():
    """Muestra estadísticas detalladas de la sesión"""
    total_acciones = stats['movimientos_izq'] + stats['movimientos_der'] + stats['saltos']
    if total_acciones > 0:
        print(f"\n📊 ESTADÍSTICAS DE LA SESIÓN:")
        print(f"⬅️  Movimientos izquierda: {stats['movimientos_izq']} ({stats['movimientos_izq']/total_acciones*100:.1f}%)")
        print(f"➡️  Movimientos derecha: {stats['movimientos_der']} ({stats['movimientos_der']/total_acciones*100:.1f}%)")
        print(f"⬆️  Saltos: {stats['saltos']} ({stats['saltos']/total_acciones*100:.1f}%)")
        print(f"💥 Colisiones totales: {stats['colisiones']}")
        print(f"🎮 Frames jugados: {stats['frames_totales']}")
        print(f"📊 Datos recolectados: {len(datos_para_csv)}")

def reiniciar_partida():
    """Reinicia una partida manteniendo las estadísticas"""
    global jugador_rect, bala_horizontal_disparada, bala_vertical_disparada, ufo_rect
    global salto_activo, en_suelo, salto_altura_actual, jugador_velocidad_x_actual
    
    jugador_rect.x = 50; jugador_rect.top = POS_Y_TOP_JUGADOR_SUELO; jugador_velocidad_x_actual = 0
    salto_activo = False; en_suelo = True; salto_altura_actual = ALTURA_SALTO_INICIAL
    reset_bala_horizontal(); reset_bala_vertical()
    ufo_rect.centery = POS_Y_TOP_BALA_HORIZONTAL + bala_horizontal_img_asset.get_rect().height // 2
    ufo_rect.centerx = ANCHO_PANTALLA - 30
    
    print("🔄 Partida reiniciada - ¡Sigue jugando para recolectar más datos!")

def gestionar_pausa():
    global pausa
    pausa = not pausa
    if pausa: 
        print("⏸️  Juego pausado - Presiona P para continuar")
    else: 
        print("▶️  Juego reanudado")

def main_loop():
    global salto_activo, en_suelo, salto_altura_actual, jugador_velocidad_x_actual
    global pausa, corriendo, stats
    global accion_horizontal_comandada_manual, accion_salto_comandado_manual

    reloj = pygame.time.Clock()
    
    print("🎮 MODO RECOLECCIÓN DE DATOS INICIADO")
    print("Objetivo: Jugar para generar datos balanceados de entrenamiento")
    print("- Muévete MUCHO (izquierda/derecha) para esquivar balas verticales")
    print("- Salta FRECUENTEMENTE para esquivar balas horizontales")
    print("- Intenta sobrevivir el mayor tiempo posible\n")
    
    while corriendo:
        # Reset de acciones manuales cada frame
        accion_horizontal_comandada_manual = 0
        accion_salto_comandado_manual = 0
        
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT: 
                corriendo = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_LEFT: 
                    jugador_velocidad_x_actual = -JUGADOR_VELOCIDAD_HORIZONTAL
                    accion_horizontal_comandada_manual = 1
                    stats['movimientos_izq'] += 1
                elif evento.key == pygame.K_RIGHT: 
                    jugador_velocidad_x_actual = JUGADOR_VELOCIDAD_HORIZONTAL
                    accion_horizontal_comandada_manual = 2
                    stats['movimientos_der'] += 1
                elif evento.key == pygame.K_SPACE and en_suelo:
                    salto_activo = True; en_suelo = False; salto_altura_actual = ALTURA_SALTO_INICIAL
                    accion_salto_comandado_manual = 1
                    stats['saltos'] += 1
                elif evento.key == pygame.K_p: 
                    gestionar_pausa()
                elif evento.key == pygame.K_ESCAPE: 
                    reiniciar_partida()
                elif evento.key == pygame.K_q:
                    corriendo = False
            if evento.type == pygame.KEYUP:
                if (evento.key == pygame.K_LEFT and jugador_velocidad_x_actual < 0) or \
                   (evento.key == pygame.K_RIGHT and jugador_velocidad_x_actual > 0):
                    jugador_velocidad_x_actual = 0
        
        if not pausa:
            actualizar_posicion_jugador()
            recolectar_datos_para_ia()
            actualizar_estado_general()
        
        dibujar_elementos()
        pygame.display.flip()
        reloj.tick(30)
    
    # Guardar datos finales al salir
    if datos_para_csv:
        guardar_datos_csv_ia()
        mostrar_estadisticas()
    
    print("\n🏁 Sesión de recolección finalizada")
    print("Ahora puedes entrenar los modelos con train_arbol_mejorado.py")
    pygame.quit()

if __name__ == "__main__":
    main_loop()