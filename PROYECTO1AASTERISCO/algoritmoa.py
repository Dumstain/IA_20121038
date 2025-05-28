import pygame
import sys
import heapq 
import math

# --- Constantes ---
# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)       # Obstáculos
GRIS_CLARO = (200, 200, 200) # Líneas
GRIS_MEDIO = (160, 160, 160) # Input Inactivo
GRIS_OSCURO = (100, 100, 100) # Panel UI
AZUL_CIELO = (135, 206, 235) # Input Activo
VERDE_BOTON = (50, 205, 50)
VERDE_BOTON_HOVER = (34, 139, 34)
NARANJA = (255, 165, 0)  # Nodo Inicio
TURQUESA = (64, 224, 208) # Nodo Fin
ROJO_CLARO = (255, 100, 100) # Nodo en Conjunto Cerrado
VERDE_CLARO = (100, 255, 100) # Nodo en Conjunto Abierto
PURPURA = (128, 0, 128) # Nodo del Camino
AMARILLO_TOOLTIP_BG = (255, 255, 224) # Fondo para el tooltip
NEGRO_TOOLTIP_TEXTO = (50,50,50) # Texto del tooltip

# Dimensiones de la ventana y UI
ANCHO_VENTANA = 1000
ALTO_VENTANA = 750
ANCHO_PANEL_DERECHO = 250
ANCHO_GRID = ANCHO_VENTANA - ANCHO_PANEL_DERECHO
ALTO_GRID = ALTO_VENTANA

# Configuración de la cuadrícula y A*
NUM_FILAS_DEFAULT = 20
NUM_COLUMNAS_DEFAULT = 20
COSTO_CARDINAL = 10 
COSTO_DIAGONAL = 14 
UMBRAL_FILAS_COLS_TOOLTIP = 10 # Si filas O cols >= esto, usar tooltip

# Fuentes
pygame.font.init()
FUENTE_UI_ETIQUETA = pygame.font.Font(None, 28)
FUENTE_UI_INPUT = pygame.font.Font(None, 32)
FUENTE_UI_BOTON = pygame.font.Font(None, 30)
FUENTE_INSTRUCCIONES = pygame.font.Font(None, 20) 
FUENTE_SCORES_NODO = pygame.font.Font(None, 18) 
FUENTE_SCORES_TOOLTIP = pygame.font.Font(None, 22)


# --- Clase Nodo ---
class Nodo:
    def __init__(self, fila, col, ancho_celda, alto_celda, total_filas, total_columnas):
        self.fila = fila
        self.col = col
        self.ancho_celda = ancho_celda
        self.alto_celda = alto_celda
        self.total_filas = total_filas
        self.total_columnas = total_columnas
        self.x = col * ancho_celda
        self.y = fila * alto_celda
        self.color = BLANCO
        
        self.vecinos = []
        self.g_cost = float('inf')
        self.h_cost = float('inf')
        self.f_cost = float('inf')
        self.padre = None
        
        self.es_obstaculo = False
        self.es_inicio = False
        self.es_fin = False

    def dibujar(self, ventana, dibujar_scores_en_celda=False):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho_celda, self.alto_celda))
        
        if dibujar_scores_en_celda and self.f_cost != float('inf'):
            # Asegurar que g_cost, h_cost, f_cost no sean inf antes de convertir a int
            g_display = str(int(self.g_cost)) if self.g_cost != float('inf') else "inf"
            h_display = str(int(self.h_cost)) if self.h_cost != float('inf') else "inf"
            f_display = str(int(self.f_cost)) if self.f_cost != float('inf') else "inf"

            g_text = FUENTE_SCORES_NODO.render(f"G:{g_display}", True, NEGRO_TOOLTIP_TEXTO if self.color != NEGRO else BLANCO)
            h_text = FUENTE_SCORES_NODO.render(f"H:{h_display}", True, NEGRO_TOOLTIP_TEXTO if self.color != NEGRO else BLANCO)
            f_text = FUENTE_SCORES_NODO.render(f"F:{f_display}", True, NEGRO_TOOLTIP_TEXTO if self.color != NEGRO else BLANCO)
            
            ventana.blit(g_text, (self.x + 2, self.y + 2))
            ventana.blit(h_text, (self.x + self.ancho_celda - h_text.get_width() - 2, self.y + 2))
            ventana.blit(f_text, (self.x + (self.ancho_celda - f_text.get_width())//2, self.y + self.alto_celda - f_text.get_height() - 2))


    def obtener_pos(self): return self.fila, self.col
    def esta_clickeado(self, pos_mouse): return self.x <= pos_mouse[0] < self.x + self.ancho_celda and self.y <= pos_mouse[1] < self.y + self.alto_celda

    def resetear_estado_algoritmo(self):
        if not self.es_inicio and not self.es_fin and not self.es_obstaculo: self.color = BLANCO 
        self.g_cost = float('inf'); self.h_cost = float('inf'); self.f_cost = float('inf')
        self.padre = None
    def resetear_completo(self):
        self.color = BLANCO; self.es_obstaculo = False; self.es_inicio = False; self.es_fin = False
        self.resetear_estado_algoritmo(); self.vecinos = []
    def hacer_inicio(self): self.color = NARANJA; self.es_inicio = True; self.es_fin = False; self.es_obstaculo = False
    def hacer_fin(self): self.color = TURQUESA; self.es_fin = True; self.es_inicio = False; self.es_obstaculo = False
    def hacer_obstaculo(self): self.color = NEGRO; self.es_obstaculo = True; self.es_inicio = False; self.es_fin = False
    def hacer_abierto(self): 
        if not self.es_inicio and not self.es_fin: self.color = VERDE_CLARO
    def hacer_cerrado(self): 
        if not self.es_inicio and not self.es_fin: self.color = ROJO_CLARO
    def hacer_camino(self): 
        if not self.es_inicio and not self.es_fin: self.color = PURPURA
    def calcular_heuristica(self, nodo_fin):
        dx = abs(self.fila - nodo_fin.fila); dy = abs(self.col - nodo_fin.col)
        self.h_cost = COSTO_DIAGONAL * min(dx, dy) + COSTO_CARDINAL * (abs(dx - dy))
        return self.h_cost
    def actualizar_vecinos(self, grid_nodos):
        self.vecinos = []
        posibles_movimientos = [
            (0, 1, False), (0, -1, False), (1, 0, False), (-1, 0, False), 
            (1, 1, True), (1, -1, True), (-1, 1, True), (-1, -1, True)]
        for dr, dc, es_diag in posibles_movimientos:
            cf, cc = self.fila + dr, self.col + dc
            if 0 <= cf < self.total_filas and 0 <= cc < self.total_columnas:
                vec_pot = grid_nodos[cf][cc]
                if not vec_pot.es_obstaculo:
                    if es_diag: # Lógica para no cortar esquinas (si una de las celdas cardinales intermedias es obstáculo)
                        obstaculo_intermedio1 = grid_nodos[self.fila + dr][self.col].es_obstaculo
                        obstaculo_intermedio2 = grid_nodos[self.fila][self.col + dc].es_obstaculo
                        if not obstaculo_intermedio1 or not obstaculo_intermedio2 : # Permite si al menos una ruta cardinal está libre
                             self.vecinos.append(vec_pot)
                    else: self.vecinos.append(vec_pot)
    def __lt__(self, other): return self.f_cost < other.f_cost

# --- Clases InputBox y Button ---
class InputBox:
    def __init__(self, x, y, w, h, etiqueta="", max_len=3):
        self.rect = pygame.Rect(x, y, w, h); self.etiqueta = etiqueta; self.texto = ""
        self.activo = False; self.color_activo = AZUL_CIELO; self.color_inactivo = GRIS_MEDIO
        self.color_borde = self.color_inactivo; self.max_len = max_len
    def manejar_evento(self, evento):
        if evento.type == pygame.MOUSEBUTTONDOWN:
            self.activo = self.rect.collidepoint(evento.pos) # True si click dentro, False si click fuera.
            self.color_borde = self.color_activo if self.activo else self.color_inactivo
        if evento.type == pygame.KEYDOWN and self.activo:
            if evento.key == pygame.K_BACKSPACE: self.texto = self.texto[:-1]
            elif evento.unicode.isdigit() and len(self.texto) < self.max_len: self.texto += evento.unicode
    def dibujar(self, ventana):
        et_surf = FUENTE_UI_ETIQUETA.render(self.etiqueta, True, BLANCO)
        ventana.blit(et_surf, (self.rect.x - et_surf.get_width() - 5, self.rect.y + (self.rect.height - et_surf.get_height()) // 2))
        pygame.draw.rect(ventana, self.color_borde, self.rect, 2) # Borde coloreado
        pygame.draw.rect(ventana, BLANCO, self.rect.inflate(-4,-4)) # Fondo blanco
        txt_surf = FUENTE_UI_INPUT.render(self.texto, True, NEGRO)
        ventana.blit(txt_surf, (self.rect.x + 5, self.rect.y + (self.rect.height - txt_surf.get_height()) // 2))
    def obtener_valor(self):
        try: return int(self.texto)
        except ValueError: return 0

class Button:
    def __init__(self, x, y, w, h, texto, color_base=VERDE_BOTON, color_hover=VERDE_BOTON_HOVER):
        self.rect = pygame.Rect(x, y, w, h); self.texto_str = texto
        self.color_base = color_base; self.color_hover = color_hover; self.color_actual = color_base
    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color_actual, self.rect, border_radius=5)
        txt_surf = FUENTE_UI_BOTON.render(self.texto_str, True, BLANCO)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        ventana.blit(txt_surf, txt_rect)
    def manejar_evento(self, evento, pos_mouse):
        self.color_actual = self.color_hover if self.rect.collidepoint(pos_mouse) else self.color_base
        if self.rect.collidepoint(pos_mouse) and evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1: return True
        return False

# --- Funciones Auxiliares y A* ---
def crear_grid_nodos(filas, columnas, ancho_total_grid, alto_total_grid):
    grid = []; filas = max(1, filas); columnas = max(1, columnas)
    ancho_celda = ancho_total_grid // columnas; alto_celda = alto_total_grid // filas
    ancho_celda = max(1, ancho_celda); alto_celda = max(1, alto_celda)
    for i in range(filas):
        fila_nodos = [Nodo(i, j, ancho_celda, alto_celda, filas, columnas) for j in range(columnas)]
        grid.append(fila_nodos)
    return grid, ancho_celda, alto_celda

def actualizar_vecinos_de_grid(grid_nodos): 
    if grid_nodos: 
        for fila in grid_nodos: 
            for nodo in fila: nodo.actualizar_vecinos(grid_nodos)

def resetear_estado_algoritmo_grid(grid_nodos, nodo_inicio_obj, nodo_fin_obj): 
    if grid_nodos:
        for fila in grid_nodos: 
            for nodo in fila: nodo.resetear_estado_algoritmo()
        if nodo_inicio_obj: nodo_inicio_obj.hacer_inicio()
        if nodo_fin_obj: nodo_fin_obj.hacer_fin()

def dibujar_lineas_grid(ventana, filas, columnas, ancho_total_grid, alto_total_grid, ancho_celda, alto_celda): 
    if not all([filas, columnas, ancho_celda, alto_celda]): return
    for i in range(filas + 1):
        y = i * alto_celda; pygame.draw.line(ventana, GRIS_CLARO, (0, y), (columnas * ancho_celda, y))
    for j in range(columnas + 1):
        x = j * ancho_celda; pygame.draw.line(ventana, GRIS_CLARO, (x, 0), (x, filas * alto_celda))

def obtener_nodo_desde_pos_mouse(pos_mouse, ancho_grid_area, alto_grid_area, ancho_celda, alto_celda, num_filas, num_columnas): 
    if not all([ancho_celda, alto_celda]): return None, None
    if not (0 <= pos_mouse[0] < ancho_grid_area and 0 <= pos_mouse[1] < alto_grid_area): return None, None
    col = pos_mouse[0] // ancho_celda; fila = pos_mouse[1] // alto_celda
    if 0 <= fila < num_filas and 0 <= col < num_columnas: return fila, col
    return None, None

def reconstruir_camino(nodo_actual, funcion_dibujo): 
    costo_total = nodo_actual.g_cost; path_len = 0
    # El nodo final ya tiene su color, empezamos a pintar desde su padre
    if nodo_actual.padre:
        nodo_actual = nodo_actual.padre # Mover al padre para empezar a pintar el camino
        while nodo_actual.padre: # Continuar hasta que el padre sea el nodo de inicio
            nodo_actual.hacer_camino(); 
            nodo_actual = nodo_actual.padre; path_len +=1
            funcion_dibujo(); pygame.time.delay(25)
        # El último nodo antes del inicio (el padre del bucle anterior es el inicio) también es camino
        if nodo_actual.es_inicio == False : # Asegurarse de no repintar el nodo de inicio con color de camino
             nodo_actual.hacer_camino() 
             path_len +=1
             funcion_dibujo(); pygame.time.delay(25)
    print(f"Costo total del camino: {costo_total}, Pasos (sin contar inicio): {path_len}")


def algoritmo_astar(funcion_dibujo, grid_nodos, inicio, fin): 
    cont_id = 0; pq = []; inicio.g_cost = 0; inicio.h_cost = inicio.calcular_heuristica(fin)
    inicio.f_cost = inicio.g_cost + inicio.h_cost; heapq.heappush(pq, (inicio.f_cost, cont_id, inicio))
    abierto_hash = {inicio}; cerrado_hash = set()
    while pq:
        for ev in pygame.event.get(): 
            if ev.type == pygame.QUIT: pygame.quit(); sys.exit()
        _, _, actual = heapq.heappop(pq)
        if actual in cerrado_hash: continue
        # Si se saca de abierto_hash antes de verificar si está en cerrado, se pueden procesar nodos duplicados
        # que se añadieron a la cola antes de que una ruta más corta los hiciera obsoletos.
        # Es mejor quitarlo después de confirmar que no está en cerrado o es el objetivo.
        
        if actual == fin:
            # Quitar de abierto_hash aquí, aunque ya no se usará para este nodo.
            if actual in abierto_hash : abierto_hash.remove(actual) 
            cerrado_hash.add(actual) # Asegurar que el nodo final esté en cerrado para consistencia
            reconstruir_camino(fin, funcion_dibujo); fin.hacer_fin(); inicio.hacer_inicio(); return True 
        
        # Mover la lógica de remover de abierto_hash y añadir a cerrado_hash después de la comprobación de 'fin'
        # y antes de procesar vecinos, si no es 'fin'.
        if actual in abierto_hash : abierto_hash.remove(actual)
        cerrado_hash.add(actual)

        for vec in actual.vecinos:
            if vec in cerrado_hash: continue
            es_diag = abs(actual.fila - vec.fila) == 1 and abs(actual.col - vec.col) == 1
            costo_mov = COSTO_DIAGONAL if es_diag else COSTO_CARDINAL
            g_tent = actual.g_cost + costo_mov
            if g_tent < vec.g_cost:
                vec.padre = actual; vec.g_cost = g_tent; vec.h_cost = vec.calcular_heuristica(fin)
                vec.f_cost = vec.g_cost + vec.h_cost
                if vec not in abierto_hash: # Solo añadir si no está ya en la cola (o para actualizar si se permite re-add)
                    cont_id += 1; heapq.heappush(pq, (vec.f_cost, cont_id, vec))
                    abierto_hash.add(vec); vec.hacer_abierto()
                # No hay 'else' para actualizar en heapq, simplemente se añade otra vez.
                # El heapq sacará el que tenga menor f_cost.
        if actual != inicio: actual.hacer_cerrado() # No colorear el nodo de inicio como cerrado
        funcion_dibujo(); pygame.time.delay(1) # Delay muy corto para visualización fluida
    return False 

def dibujar_panel_ui(ventana, input_filas, input_cols, boton_aplicar): 
    panel_rect = pygame.Rect(ANCHO_GRID, 0, ANCHO_PANEL_DERECHO, ALTO_VENTANA)
    pygame.draw.rect(ventana, GRIS_OSCURO, panel_rect)
    input_filas.dibujar(ventana); input_cols.dibujar(ventana); boton_aplicar.dibujar(ventana)
    y_sep = boton_aplicar.rect.bottom + 20
    pygame.draw.line(ventana, GRIS_CLARO, (ANCHO_GRID + 10, y_sep), (ANCHO_VENTANA - 10, y_sep), 2)
    instrucciones = [
        "Controles:",
        "Click Izq Grid: Inicio > Fin > Muros",
        "Click Der Grid: Borrar nodo",
        "Panel Der: Cambiar dims y Aplicar",
        "[Espacio]: Iniciar Algoritmo A*",
        "[R]: Reiniciar Cuadrícula Completa",
        "[C]: Limpiar Camino y Estados A*",
        "Scores GHF: En celda (<10 F/C),",
        "  o Tooltip con mouse si >=10 F/C."
    ]
    y_txt = y_sep + 15
    for i, linea in enumerate(instrucciones):
        surf = (FUENTE_UI_ETIQUETA if i == 0 else FUENTE_INSTRUCCIONES).render(linea, True, BLANCO)
        ventana.blit(surf, (ANCHO_GRID + 20, y_txt))
        y_txt += 20 if i==0 else 18

def dibujar_tooltip(ventana, nodo_hover, pos_mouse):
    if nodo_hover and nodo_hover.f_cost != float('inf'):
        g_val = f"G: {int(nodo_hover.g_cost)}" if nodo_hover.g_cost != float('inf') else "G: inf"
        h_val = f"H: {int(nodo_hover.h_cost)}" if nodo_hover.h_cost != float('inf') else "H: inf"
        f_val = f"F: {int(nodo_hover.f_cost)}" if nodo_hover.f_cost != float('inf') else "F: inf"
        
        textos_surf = [
            FUENTE_SCORES_TOOLTIP.render(g_val, True, NEGRO_TOOLTIP_TEXTO),
            FUENTE_SCORES_TOOLTIP.render(h_val, True, NEGRO_TOOLTIP_TEXTO),
            FUENTE_SCORES_TOOLTIP.render(f_val, True, NEGRO_TOOLTIP_TEXTO)
        ]
        padding = 5
        max_ancho_texto = max(s.get_width() for s in textos_surf) if textos_surf else 0
        alto_total_texto = sum(s.get_height() for s in textos_surf) + padding * (len(textos_surf) -1 if len(textos_surf)>0 else 0)
        
        tooltip_ancho = max_ancho_texto + 2 * padding
        tooltip_alto = alto_total_texto + 2 * padding
        
        tt_x, tt_y = pos_mouse[0] + 15, pos_mouse[1] + 10
        if tt_x + tooltip_ancho > ANCHO_VENTANA: tt_x = pos_mouse[0] - tooltip_ancho - 10
        if tt_y + tooltip_alto > ALTO_VENTANA: tt_y = pos_mouse[1] - tooltip_alto - 10
        
        tooltip_rect = pygame.Rect(tt_x, tt_y, tooltip_ancho, tooltip_alto)
        pygame.draw.rect(ventana, AMARILLO_TOOLTIP_BG, tooltip_rect)
        pygame.draw.rect(ventana, NEGRO_TOOLTIP_TEXTO, tooltip_rect, 1) 
        
        y_offset = tt_y + padding
        for surf in textos_surf:
            ventana.blit(surf, (tt_x + padding, y_offset))
            y_offset += surf.get_height() + padding


def dibujar_todo_wrapper(ventana, grid_nodos, filas, columnas, ancho_total_grid, alto_total_grid, ancho_celda, alto_celda, ui_elements, nodo_tooltip, pos_mouse_tooltip):
    ventana.fill(BLANCO)
    dibujar_scores_en_celda_flag = (filas < UMBRAL_FILAS_COLS_TOOLTIP and columnas < UMBRAL_FILAS_COLS_TOOLTIP)
    if grid_nodos:
        for fila_de_nodos in grid_nodos:
            for nodo in fila_de_nodos:
                nodo.dibujar(ventana, dibujar_scores_en_celda_flag) 
    dibujar_lineas_grid(ventana, filas, columnas, ancho_total_grid, alto_total_grid, ancho_celda, alto_celda)
    dibujar_panel_ui(ventana, ui_elements['input_filas'], ui_elements['input_cols'], ui_elements['boton_aplicar'])
    if nodo_tooltip and not dibujar_scores_en_celda_flag : 
        dibujar_tooltip(ventana, nodo_tooltip, pos_mouse_tooltip)
    pygame.display.update()

# --- Bucle Principal ---
def main():
    pygame.init()
    pygame.display.set_caption("Visualizador Algoritmo A* - v0.6 (Scores GHF)")
    ventana = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))

    num_filas_actual = NUM_FILAS_DEFAULT
    num_columnas_actual = NUM_COLUMNAS_DEFAULT 
    
    padding_ui = 20; ancho_input = 60; alto_input = 30
    x_base_etiqueta_input = ANCHO_GRID + padding_ui + 60 # Ajustar para que quepa "Cols:"
    input_filas = InputBox(x_base_etiqueta_input, padding_ui + 30, ancho_input, alto_input, "Filas:", 2)
    input_filas.texto = str(num_filas_actual)
    input_cols = InputBox(x_base_etiqueta_input, input_filas.rect.bottom + padding_ui, ancho_input, alto_input, "Cols:", 2)
    input_cols.texto = str(num_columnas_actual)
    boton_aplicar = Button(ANCHO_GRID + (ANCHO_PANEL_DERECHO - 100) // 2, input_cols.rect.bottom + padding_ui, 100, 40, "Aplicar")
    ui_elements = {'input_filas': input_filas, 'input_cols': input_cols, 'boton_aplicar': boton_aplicar}

    grid_nodos, ancho_celda, alto_celda = crear_grid_nodos(num_filas_actual, num_columnas_actual, ANCHO_GRID, ALTO_GRID)
    actualizar_vecinos_de_grid(grid_nodos)
    
    nodo_inicio_obj = None
    nodo_fin_obj = None
    algoritmo_corriendo = False
    nodo_para_tooltip = None 

    corriendo = True
    clock = pygame.time.Clock() # Para controlar FPS

    while corriendo:
        pos_mouse = pygame.mouse.get_pos()
        # Wrapper para la función de dibujo, útil para pasar a A*
        # No se pasa nodo_para_tooltip ni pos_mouse_tooltip a A* para no dibujarlo durante su ejecución
        funcion_dibujo_para_astar = lambda: dibujar_todo_wrapper(ventana, grid_nodos, num_filas_actual, num_columnas_actual, ANCHO_GRID, ALTO_GRID, ancho_celda, alto_celda, ui_elements, None, None)

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT: corriendo = False
            
            if not algoritmo_corriendo:
                input_filas.manejar_evento(evento)
                input_cols.manejar_evento(evento)

                if boton_aplicar.manejar_evento(evento, pos_mouse):
                    nf = input_filas.obtener_valor(); nc = input_cols.obtener_valor()
                    # Validación para evitar celdas de tamaño 0 o negativo
                    if nf > 0 and nc > 0 and \
                       (ANCHO_GRID // nc if nc > 0 else 0) > 0 and \
                       (ALTO_GRID // nf if nf > 0 else 0) > 0:
                        num_filas_actual = nf; num_columnas_actual = nc
                        grid_nodos, ancho_celda, alto_celda = crear_grid_nodos(nf, nc, ANCHO_GRID, ALTO_GRID)
                        actualizar_vecinos_de_grid(grid_nodos)
                        nodo_inicio_obj = None; nodo_fin_obj = None
                    else: 
                        input_filas.texto = str(num_filas_actual); input_cols.texto = str(num_columnas_actual)
                        print("Dimensiones inválidas o resultarían en celdas de tamaño cero.")


                if evento.type == pygame.MOUSEBUTTONDOWN:
                    if pos_mouse[0] < ANCHO_GRID: # Click en la cuadrícula
                        f_clk, c_clk = obtener_nodo_desde_pos_mouse(pos_mouse, ANCHO_GRID, ALTO_GRID, ancho_celda, alto_celda, num_filas_actual, num_columnas_actual)
                        if f_clk is not None: # Asegurarse que el click fue en una celda válida
                            nodo_clk = grid_nodos[f_clk][c_clk]
                            if evento.button == 1: # Click Izquierdo
                                if not nodo_inicio_obj and nodo_clk!=nodo_fin_obj: nodo_inicio_obj=nodo_clk; nodo_inicio_obj.hacer_inicio()
                                elif not nodo_fin_obj and nodo_clk!=nodo_inicio_obj: nodo_fin_obj=nodo_clk; nodo_fin_obj.hacer_fin()
                                elif nodo_clk!=nodo_inicio_obj and nodo_clk!=nodo_fin_obj: nodo_clk.hacer_obstaculo()
                                actualizar_vecinos_de_grid(grid_nodos) 
                            elif evento.button == 3: # Click Derecho
                                if nodo_clk==nodo_inicio_obj: nodo_inicio_obj=None
                                if nodo_clk==nodo_fin_obj: nodo_fin_obj=None
                                era_obs = nodo_clk.es_obstaculo; nodo_clk.resetear_completo()
                                if era_obs: actualizar_vecinos_de_grid(grid_nodos)
                
                if evento.type == pygame.MOUSEMOTION:
                    if pos_mouse[0] < ANCHO_GRID: 
                        f_hov, c_hov = obtener_nodo_desde_pos_mouse(pos_mouse, ANCHO_GRID, ALTO_GRID, ancho_celda, alto_celda, num_filas_actual, num_columnas_actual)
                        if f_hov is not None and (num_filas_actual >= UMBRAL_FILAS_COLS_TOOLTIP or num_columnas_actual >= UMBRAL_FILAS_COLS_TOOLTIP):
                            nodo_para_tooltip = grid_nodos[f_hov][c_hov]
                        else:
                            nodo_para_tooltip = None
                    else:
                        nodo_para_tooltip = None


                if evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_SPACE:
                        if nodo_inicio_obj and nodo_fin_obj:
                            print("Iniciando A*...")
                            resetear_estado_algoritmo_grid(grid_nodos, nodo_inicio_obj, nodo_fin_obj) 
                            actualizar_vecinos_de_grid(grid_nodos) 
                            algoritmo_corriendo = True
                            path_found = algoritmo_astar(funcion_dibujo_para_astar, grid_nodos, nodo_inicio_obj, nodo_fin_obj)
                            algoritmo_corriendo = False
                            if not path_found: print("No se encontró camino.")
                        else: print("Define Inicio y Fin.")
                    elif evento.key == pygame.K_r: 
                        print("Reiniciando cuadrícula...")
                        grid_nodos, ancho_celda, alto_celda = crear_grid_nodos(num_filas_actual, num_columnas_actual, ANCHO_GRID, ALTO_GRID)
                        actualizar_vecinos_de_grid(grid_nodos); nodo_inicio_obj=None; nodo_fin_obj=None
                    elif evento.key == pygame.K_c: 
                        print("Limpiando camino...")
                        resetear_estado_algoritmo_grid(grid_nodos, nodo_inicio_obj, nodo_fin_obj)
            
        if not algoritmo_corriendo: # Dibujar el estado actual si A* no está corriendo y dibujando
             dibujar_todo_wrapper(ventana, grid_nodos, num_filas_actual, num_columnas_actual, ANCHO_GRID, ALTO_GRID, ancho_celda, alto_celda, ui_elements, nodo_para_tooltip, pos_mouse)
        
        clock.tick(60) # Limitar a 60 FPS

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()