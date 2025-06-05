import pygame
import sys
import os

# --- CONFIGURACIÓN DEL CONTROLADOR DE ANIMACIONES ---
# Personaje
NUM_CUADROS_CORRER = 12  # Ajustado para tu juego (era 12)
NUM_CUADROS_SALTAR = 6 

PREFIJO_CORRER_SUP = "assets/sprites/upper"
PREFIJO_CORRER_INF = "assets/sprites/lower"
PREFIJO_SALTAR_SUP = "assets/sprites/upper_jump"
PREFIJO_SALTAR_INF = "assets/sprites/lower_jump"
EXTENSION_IMG = ".png"

OFFSET_X_PARTE_SUPERIOR = 8
OFFSET_X_PARTE_INFERIOR = 0
OFFSET_Y_UNION_SUPERIOR = 12
OFFSET_Y_UNION_SUPERIOR_SALTO = 3
VELOCIDAD_ANIMACION_PERSONAJE = 3

# --- CONFIGURACIÓN BUNKERCAÑON ---
SPRITE_BUNKER_BASE = "assets/game/bunker1.png"
SPRITE_BUNKER_CUBIERTA = "assets/game/bunker2.png"
PREFIJO_CANON = "assets/game/canon"
NUM_CUADROS_CANON_TOTAL = 10
PREFIJO_DISPARO = "assets/game/disparo"
NUM_CUADROS_DISPARO_TOTAL = 17

INDICE_CANON_INICIAL = 1
INDICE_CANON_LISTO = 7
INDICE_CANON_MAX_EXT = 10

VELOCIDAD_ANIM_CANON_EXTENDIENDO = 4
VELOCIDAD_ANIM_DISPARO = 2
VELOCIDAD_ANIM_CANON_CONTRAYENDO = 3

OFFSET_CANON_X = 30
OFFSET_CANON_Y = 20
OFFSET_CUBIERTA_X = -9
OFFSET_CUBIERTA_Y = 3
OFFSET_DISPARO_X = -60
OFFSET_DISPARO_Y = -20

# --- CONFIGURACIÓN PROYECTILES ---
SPRITE_PROYECTIL = "assets/sprites/proyectil.png"
VELOCIDAD_PROYECTIL = 8  # Velocidad horizontal del proyectil
GRAVEDAD_PROYECTIL = 0.3  # Gravedad que afecta al proyectil
TIEMPO_VIDA_PROYECTIL = 180  # Frames que vive el proyectil (3 segundos a 60 FPS)

# Colores
ROJO = (255, 0, 0) 
COLOR_HILO_CENTRAL = (255, 255, 0) 

class ProyectilAnimado:
    def __init__(self, x_inicial, y_inicial, velocidad_x, velocidad_y=0):
        self.x = x_inicial
        self.y = y_inicial
        self.velocidad_x = velocidad_x
        self.velocidad_y = velocidad_y
        self.activo = True
        self.tiempo_vida = TIEMPO_VIDA_PROYECTIL
        
        # Cargar el sprite del proyectil
        self.sprite = None
        if os.path.exists(SPRITE_PROYECTIL):
            self.sprite = pygame.image.load(SPRITE_PROYECTIL).convert_alpha()
        else:
            print(f"ADVERTENCIA: No se encontró {SPRITE_PROYECTIL}, usando sprite temporal")
            # Crear un sprite temporal (círculo rojo)
            self.sprite = pygame.Surface((16, 16), pygame.SRCALPHA)
            pygame.draw.circle(self.sprite, ROJO, (8, 8), 8)
        
        # Rect para colisiones - COMPATIBLE CON TU SISTEMA
        self.rect = pygame.Rect(self.x, self.y, self.sprite.get_width(), self.sprite.get_height())
    
    def actualizar(self, ancho_pantalla, alto_pantalla, altura_suelo):
        """Actualizar proyectil - compatible con tu sistema"""
        if not self.activo:
            return
        
        # Actualizar posición
        self.x += self.velocidad_x
        self.y += self.velocidad_y
        
        # Aplicar gravedad
        self.velocidad_y += GRAVEDAD_PROYECTIL
        
        # Actualizar rect para colisiones
        self.rect.x = int(self.x)
        self.rect.y = int(self.y)
        
        # Verificar límites de pantalla
        if (self.x < -50 or self.x > ancho_pantalla + 50 or 
            self.y > alto_pantalla + 50):
            self.activo = False
        
        # Verificar tiempo de vida
        self.tiempo_vida -= 1
        if self.tiempo_vida <= 0:
            self.activo = False
        
        # Verificar colisión con el suelo
        if self.y >= altura_suelo - self.sprite.get_height():
            self.activo = False
    
    def dibujar(self, superficie):
        """Dibujar proyectil"""
        if self.activo and self.sprite:
            superficie.blit(self.sprite, (int(self.x), int(self.y)))
    
    # MÉTODOS COMPATIBLES CON TU SISTEMA ORIGINAL
    def get_rect(self):
        return self.rect
    
    def is_active(self):
        return self.activo

class PersonajeAnimado:
    def __init__(self, x_inicial, y_suelo):
        self.x = x_inicial
        self.y = y_suelo
        self.y_suelo_logica = y_suelo
        self.offset_base_upper_x = OFFSET_X_PARTE_SUPERIOR
        self.offset_base_lower_x = OFFSET_X_PARTE_INFERIOR
        self.offset_y_union_superior = OFFSET_Y_UNION_SUPERIOR
        self.offset_y_union_superior_salto = OFFSET_Y_UNION_SUPERIOR_SALTO
        self.velocidad_animacion = VELOCIDAD_ANIMACION_PERSONAJE
        
        # Listas de frames de animación
        self.cuadros_correr_sup, self.cuadros_correr_inf = [], []
        self.cuadros_saltar_sup, self.cuadros_saltar_inf = [], []
        self._cargar_cuadros_individuales()
        
        # Estados de animación
        self.estado_actual = "idle"
        self.cuadro_anim_actual = 0
        self.contador_animacion = 0
        self.mirando_derecha = True
        
        # Física del salto - COMPATIBLE CON TU SISTEMA
        self.vy = 0
        self.gravedad = 1
        self.fuerza_salto = -20
        self.en_suelo = True
        
        # Crear rect de colisión - COMPATIBLE CON TU SISTEMA
        if self.cuadros_correr_sup:
            h_sup_inicial = self.cuadros_correr_sup[0].get_height()
            h_inf_inicial = self.cuadros_correr_inf[0].get_height()
            w_estimado_inicial = self.cuadros_correr_sup[0].get_width()
            alto_visual_inicial = h_sup_inicial + h_inf_inicial - self.offset_y_union_superior
            self.rect_logico = pygame.Rect(0, 0, w_estimado_inicial, alto_visual_inicial)
            self.rect_logico.midbottom = (self.x, self.y)
        else: 
            # Fallback si no hay sprites
            self.rect_logico = pygame.Rect(self.x - 25, self.y - 100, 50, 100)

    def _cargar_cuadros_individuales(self):
        """Cargar todos los frames de animación"""
        errores_carga = []
        
        def cargar_secuencia(prefijo_sup, prefijo_inf, num_cuadros, lista_sup, lista_inf, tipo_anim):
            if num_cuadros == 0: 
                return
            for i in range(1, num_cuadros + 1):
                ruta_sup = f"{prefijo_sup}{i}{EXTENSION_IMG}"
                ruta_inf = f"{prefijo_inf}{i}{EXTENSION_IMG}"
                try:
                    if not os.path.exists(ruta_sup): 
                        errores_carga.append(f"Falta: {ruta_sup} ({tipo_anim})")
                        continue
                    if not os.path.exists(ruta_inf): 
                        errores_carga.append(f"Falta: {ruta_inf} ({tipo_anim})")
                        continue
                    lista_sup.append(pygame.image.load(ruta_sup).convert_alpha())
                    lista_inf.append(pygame.image.load(ruta_inf).convert_alpha())
                except pygame.error as e: 
                    errores_carga.append(f"Error cargando {ruta_sup} o {ruta_inf}: {e}")
        
        # Cargar animaciones
        cargar_secuencia(PREFIJO_CORRER_SUP, PREFIJO_CORRER_INF, NUM_CUADROS_CORRER, 
                        self.cuadros_correr_sup, self.cuadros_correr_inf, "correr")
        cargar_secuencia(PREFIJO_SALTAR_SUP, PREFIJO_SALTAR_INF, NUM_CUADROS_SALTAR, 
                        self.cuadros_saltar_sup, self.cuadros_saltar_inf, "saltar")

        if errores_carga:
            print("\n--- ADVERTENCIAS AL CARGAR ANIMACIONES DEL PERSONAJE ---")
            for err in errores_carga: 
                print(err)
            print("Se usarán sprites de fallback si es necesario.")
            print("--------------------------------------------------------")

    def actualizar_fisica_y_movimiento(self, dx, quiere_saltar, ancho_pantalla):
        """Actualizar lógica de movimiento - COMPATIBLE CON TU SISTEMA"""
        # Lógica de estados
        if self.en_suelo:
            if quiere_saltar and self.cuadros_saltar_sup: 
                self.estado_actual = "saltando"
                self.vy = self.fuerza_salto
                self.en_suelo = False
                self.cuadro_anim_actual = 0
            elif dx != 0: 
                self.estado_actual = "corriendo"
            else: 
                self.estado_actual = "idle"
        else: 
            self.estado_actual = "saltando"

        # Movimiento horizontal
        self.x += dx
        if dx > 0: 
            self.mirando_derecha = True
        elif dx < 0: 
            self.mirando_derecha = False

        # Física del salto
        if not self.en_suelo:
            self.vy += self.gravedad
            self.y += self.vy
            if self.y >= self.y_suelo_logica: 
                self.y = self.y_suelo_logica
                self.en_suelo = True
                self.vy = 0
        
        # Límites de pantalla
        ancho_personaje_actual = self.rect_logico.width if self.rect_logico else 50
        self.x = max(0 + ancho_personaje_actual // 2, 
                    min(self.x, ancho_pantalla - ancho_personaje_actual // 2))

    def actualizar_animacion(self):
        """Actualizar frames de animación"""
        self.contador_animacion += 1
        if self.contador_animacion < self.velocidad_animacion: 
            return
        self.contador_animacion = 0
        
        cuadros_activos_sup, _ = self._obtener_listas_cuadros_actuales()
        if not cuadros_activos_sup: 
            return
        
        num_cuadros_disponibles = len(cuadros_activos_sup)

        if self.estado_actual == "idle":
            self.cuadro_anim_actual = 0
            return

        if self.estado_actual == "corriendo":
            if num_cuadros_disponibles <= 1: 
                self.cuadro_anim_actual = 0
                return
            
            self.cuadro_anim_actual += 1
            if self.cuadro_anim_actual >= num_cuadros_disponibles:
                self.cuadro_anim_actual = 0 
        
        elif self.estado_actual == "saltando":
            if self.cuadro_anim_actual < num_cuadros_disponibles - 1:
                self.cuadro_anim_actual += 1

    def _obtener_listas_cuadros_actuales(self):
        """Obtener frames según el estado actual"""
        if self.estado_actual == "saltando" and self.cuadros_saltar_sup: 
            return self.cuadros_saltar_sup, self.cuadros_saltar_inf
        if self.cuadros_correr_sup: 
            return self.cuadros_correr_sup, self.cuadros_correr_inf
        return [], []

    def dibujar(self, superficie):
        """Renderizar el personaje con animación completa"""
        lista_cuadros_sup_actual, lista_cuadros_inf_actual = self._obtener_listas_cuadros_actuales()
        if not lista_cuadros_sup_actual: 
            # Fallback: dibujar rectángulo simple
            pygame.draw.rect(superficie, (0, 255, 0), self.rect_logico)
            return

        idx_cuadro_valido = min(self.cuadro_anim_actual, len(lista_cuadros_sup_actual) - 1)
        cuadro_sup_actual = lista_cuadros_sup_actual[idx_cuadro_valido]
        cuadro_inf_actual = lista_cuadros_inf_actual[idx_cuadro_valido]

        # Flip horizontal si mira a la izquierda
        if not self.mirando_derecha:
            cuadro_sup_actual = pygame.transform.flip(cuadro_sup_actual, True, False)
            cuadro_inf_actual = pygame.transform.flip(cuadro_inf_actual, True, False)

        ancho_sup, alto_sup = cuadro_sup_actual.get_size()
        ancho_inf, alto_inf = cuadro_inf_actual.get_size()

        # Calcular offsets según dirección y estado
        factor_dir_offset = 1 if self.mirando_derecha else -1
        factor_est_offset = -1 if self.estado_actual == "saltando" else 1
        offset_x_eff_sup = self.offset_base_upper_x * factor_dir_offset * factor_est_offset
        offset_x_eff_inf = self.offset_base_lower_x * factor_dir_offset * factor_est_offset

        # Posicionar parte inferior
        pos_x_inf_final = (self.x - ancho_inf / 2) + offset_x_eff_inf
        pos_y_inf_final = self.y - alto_inf
        
        # Posicionar parte superior
        offset_y_union_final = (self.offset_y_union_superior_salto if self.estado_actual == "saltando" 
                               else self.offset_y_union_superior)
        pos_x_sup_final = (self.x - ancho_sup / 2) + offset_x_eff_sup
        pos_y_sup_final = (pos_y_inf_final - alto_sup) + offset_y_union_final

        # Dibujar las dos partes
        superficie.blit(cuadro_inf_actual, (pos_x_inf_final, pos_y_inf_final))
        superficie.blit(cuadro_sup_actual, (pos_x_sup_final, pos_y_sup_final))

        # Actualizar rect de colisión
        self.rect_logico.width = max(ancho_sup, ancho_inf)
        self.rect_logico.height = alto_sup + alto_inf - offset_y_union_final
        self.rect_logico.centerx = self.x
        self.rect_logico.bottom = self.y
        
        # Línea central de debug (opcional)
        # pygame.draw.line(superficie, COLOR_HILO_CENTRAL, (self.x, pos_y_sup_final), (self.x, self.y), 1)

    # MÉTODOS COMPATIBLES CON TU SISTEMA ORIGINAL
    def get_collision_rect(self):
        """Obtener rectángulo de colisión - COMPATIBLE"""
        return self.rect_logico
    
    def get_position(self):
        """Obtener posición actual - COMPATIBLE"""
        return (self.x, self.y)
    
    def set_position(self, x, y):
        """Establecer posición - COMPATIBLE"""
        self.x = x
        self.y = y
        self.rect_logico.centerx = self.x
        self.rect_logico.bottom = self.y
    
    def is_jumping(self):
        """Verificar si está saltando - COMPATIBLE"""
        return not self.en_suelo
    
    def is_on_ground(self):
        """Verificar si está en el suelo - COMPATIBLE"""
        return self.en_suelo

class BunkerAnimado:
    def __init__(self, x_esquina_base, y_suelo_para_base):
        self.x_base_ref = x_esquina_base 
        self.y_base_ref = y_suelo_para_base 

        # Cargar sprites del bunker
        self.sprite_bunker_base, self.sprite_bunker_cubierta = None, None
        if os.path.exists(SPRITE_BUNKER_BASE):
            self.sprite_bunker_base = pygame.image.load(SPRITE_BUNKER_BASE).convert_alpha()
            self.y_base_ref = y_suelo_para_base - self.sprite_bunker_base.get_height() 
        else: 
            print(f"ADVERTENCIA: No se encontró el sprite base del búnker: {SPRITE_BUNKER_BASE}")
        
        if os.path.exists(SPRITE_BUNKER_CUBIERTA): 
            self.sprite_bunker_cubierta = pygame.image.load(SPRITE_BUNKER_CUBIERTA).convert_alpha()
        else: 
            print(f"ADVERTENCIA: No se encontró el sprite de cubierta del búnker: {SPRITE_BUNKER_CUBIERTA}")

        # Cargar animaciones del cañón
        self.cuadros_canon, self.cuadros_disparo = [], []
        for i in range(1, NUM_CUADROS_CANON_TOTAL + 1):
            ruta = f"{PREFIJO_CANON}{i}{EXTENSION_IMG}"
            if os.path.exists(ruta): 
                self.cuadros_canon.append(pygame.image.load(ruta).convert_alpha())
            else: 
                print(f"ADVERTENCIA: No se encontró {ruta}")
                # Sprite temporal
                temp_sprite = pygame.Surface((30, 10))
                temp_sprite.fill((100, 100, 100))
                self.cuadros_canon.append(temp_sprite)
        
        for i in range(1, NUM_CUADROS_DISPARO_TOTAL + 1): 
            ruta = f"{PREFIJO_DISPARO}{i}{EXTENSION_IMG}"
            if os.path.exists(ruta): 
                self.cuadros_disparo.append(pygame.image.load(ruta).convert_alpha())
            else: 
                print(f"ADVERTENCIA: No se encontró {ruta}")
                # Sprite temporal
                temp_sprite = pygame.Surface((20, 20), pygame.SRCALPHA)
                self.cuadros_disparo.append(temp_sprite)
        
        # Estados de animación
        self.estado = "listo" 
        self.cuadro_actual_canon = INDICE_CANON_LISTO
        self.cuadro_actual_disparo = (NUM_CUADROS_DISPARO_TOTAL - 1) if self.cuadros_disparo else 0 
        
        self.contador_anim = 0
        self.disparo_animacion_activa = False
        
        # Offsets de posicionamiento
        self.offset_canon_x, self.offset_canon_y = OFFSET_CANON_X, OFFSET_CANON_Y
        self.offset_cubierta_x, self.offset_cubierta_y = OFFSET_CUBIERTA_X, OFFSET_CUBIERTA_Y
        self.offset_disparo_x, self.offset_disparo_y = OFFSET_DISPARO_X, OFFSET_DISPARO_Y
        
        self.paso_sincronizacion_disparo = 0
        self.proyectil_disparado = False

    def intentar_disparar(self):
        """Iniciar secuencia de disparo - COMPATIBLE CON TU SISTEMA"""
        if self.estado == "listo":
            self.estado = "disparando"
            self.cuadro_actual_disparo = 0 
            self.disparo_animacion_activa = True
            self.paso_sincronizacion_disparo = 0 
            self.contador_anim = 0 
            self.proyectil_disparado = False
            return True  # Indica que se inició el disparo
        return False

    def obtener_posicion_boquilla(self):
        """Retorna la posición actual de la boquilla del cañón"""
        if not self.cuadros_canon or self.cuadro_actual_canon < 1:
            # Fallback a posición fija
            return (self.x_base_ref, self.y_base_ref + 20)
        
        sprite_canon_actual = self.cuadros_canon[self.cuadro_actual_canon - 1]
        ancho_sprite = sprite_canon_actual.get_width()
        
        # Base fija del cañón
        base_canon_x = self.x_base_ref + self.offset_canon_x
        base_canon_y = self.y_base_ref + self.offset_canon_y
        
        # Posición de la boquilla (lado izquierdo del sprite)
        boquilla_x = base_canon_x - ancho_sprite
        boquilla_y = base_canon_y + sprite_canon_actual.get_height() // 2
        
        return (boquilla_x, boquilla_y)

    def actualizar_animacion(self, lista_proyectiles=None):
        """Actualizar animación del cañón y disparos"""
        self.contador_anim += 1
        
        if self.estado == "disparando":
            if not self.disparo_animacion_activa: 
                return None

            if self.contador_anim < VELOCIDAD_ANIM_DISPARO: 
                return None
            self.contador_anim = 0

            # Secuencia de animación de disparo
            if self.paso_sincronizacion_disparo == 0: 
                self.cuadro_actual_canon = INDICE_CANON_LISTO 
                self.paso_sincronizacion_disparo = 1
            elif self.paso_sincronizacion_disparo == 1: 
                self.cuadro_actual_canon = INDICE_CANON_LISTO + 1
                self.cuadro_actual_disparo = 1
                self.paso_sincronizacion_disparo = 2
            elif self.paso_sincronizacion_disparo == 2: 
                self.cuadro_actual_canon = INDICE_CANON_LISTO + 2
                self.cuadro_actual_disparo = 2
                self.paso_sincronizacion_disparo = 3
            elif self.paso_sincronizacion_disparo == 3: 
                self.cuadro_actual_canon = INDICE_CANON_MAX_EXT
                self.cuadro_actual_disparo = 3
                
                # MOMENTO EXACTO PARA DISPARAR PROYECTIL
                if not self.proyectil_disparado:
                    posicion_boquilla = self.obtener_posicion_boquilla()
                    if posicion_boquilla:
                        x_boquilla, y_boquilla = posicion_boquilla
                        # Crear proyectil
                        nuevo_proyectil = ProyectilAnimado(x_boquilla, y_boquilla, -VELOCIDAD_PROYECTIL, -2)
                        if lista_proyectiles is not None:
                            lista_proyectiles.append(nuevo_proyectil)
                        self.proyectil_disparado = True
                        return nuevo_proyectil  # Retornar el proyectil creado
                
                self.paso_sincronizacion_disparo = 4 
            
            elif self.paso_sincronizacion_disparo == 4: 
                self.cuadro_actual_canon = INDICE_CANON_MAX_EXT
                
                if self.disparo_animacion_activa: 
                    self.cuadro_actual_disparo += 1
                    if self.cuadro_actual_disparo >= NUM_CUADROS_DISPARO_TOTAL - 1 : 
                        self.cuadro_actual_disparo = NUM_CUADROS_DISPARO_TOTAL - 1 
                        self.disparo_animacion_activa = False 
                        self.estado = "contrayendo"
                        self.contador_anim = 0 
        
        elif self.estado == "contrayendo":
            if self.contador_anim < VELOCIDAD_ANIM_CANON_CONTRAYENDO: 
                return None
            self.contador_anim = 0
            if self.cuadro_actual_canon > INDICE_CANON_LISTO: 
                self.cuadro_actual_canon -= 1
            else: 
                self.cuadro_actual_canon = INDICE_CANON_LISTO
                self.estado = "listo"
        
        elif self.estado == "listo":
            self.cuadro_actual_canon = INDICE_CANON_LISTO
        
        return None

    def dibujar(self, superficie):
        """Renderizar el bunker con animaciones"""
        # Dibujar base del bunker
        if self.sprite_bunker_base: 
            superficie.blit(self.sprite_bunker_base, (self.x_base_ref, self.y_base_ref))
        else:
            # Fallback: rectángulo simple
            pygame.draw.rect(superficie, (100, 80, 60), 
                           (self.x_base_ref, self.y_base_ref, 80, 50))
        
        # Dibujar cañón
        sprite_canon_actual = None
        if self.cuadros_canon and 0 <= (self.cuadro_actual_canon - 1) < len(self.cuadros_canon):
            sprite_canon_actual = self.cuadros_canon[self.cuadro_actual_canon - 1]
        
        if sprite_canon_actual:
            # Posicionamiento del cañón
            base_canon_x = self.x_base_ref + self.offset_canon_x
            base_canon_y = self.y_base_ref + self.offset_canon_y
            
            ancho_sprite = sprite_canon_actual.get_width()
            pos_x_canon_final = base_canon_x - ancho_sprite
            pos_y_canon_final = base_canon_y
            
            superficie.blit(sprite_canon_actual, (pos_x_canon_final, pos_y_canon_final))

            # Dibujar animación de disparo
            if (self.cuadros_disparo and 0 <= self.cuadro_actual_disparo < len(self.cuadros_disparo) 
                and self.disparo_animacion_activa):
                sprite_disparo_actual = self.cuadros_disparo[self.cuadro_actual_disparo]
                boquilla_x = pos_x_canon_final
                pos_x_disparo_final = boquilla_x + self.offset_disparo_x
                pos_y_disparo_final = base_canon_y + self.offset_disparo_y
                superficie.blit(sprite_disparo_actual, (pos_x_disparo_final, pos_y_disparo_final))
        
        # Dibujar cubierta del bunker
        if self.sprite_bunker_cubierta: 
            pos_x_cubierta_final = self.x_base_ref + self.offset_cubierta_x
            pos_y_cubierta_final = self.y_base_ref + self.offset_cubierta_y
            superficie.blit(self.sprite_bunker_cubierta, (pos_x_cubierta_final, pos_y_cubierta_final))

    # MÉTODOS COMPATIBLES CON TU SISTEMA ORIGINAL
    def get_rect(self):
        """Obtener rectángulo del bunker - COMPATIBLE"""
        if self.sprite_bunker_base:
            return pygame.Rect(self.x_base_ref, self.y_base_ref, 
                             self.sprite_bunker_base.get_width(), 
                             self.sprite_bunker_base.get_height())
        else:
            return pygame.Rect(self.x_base_ref, self.y_base_ref, 80, 50)
    
    def get_position(self):
        """Obtener posición del bunker - COMPATIBLE"""
        return (self.x_base_ref, self.y_base_ref)
    
    def is_ready_to_fire(self):
        """Verificar si puede disparar - COMPATIBLE"""
        return self.estado == "listo"