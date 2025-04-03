# Modelo de Red Neuronal para Identificar Emociones a partir de Landmarks de Mediapipe

En este trabajo se describe un modelo sencillo de red neuronal para clasificar emociones usando los valores de los landmarks que genera Mediapipe. A continuación, se explica la estructura del modelo, los datos de entrada, las funciones de activación, entre otros detalles importantes.

---

## 1. Tipo de Red Neuronal y sus Partes

Se usará una **Red Neuronal Feedforward Multicapa (MLP)**. Esta red se compone de:

### a. Capa de Entrada (Input Layer)
- **Función:** Recibe el vector con todos los valores de los landmarks.
- **Detalles:**  
  - Por ejemplo, si se usan 468 landmarks en 3D (coordenadas *x*, *y* y *z*), el número total de entradas será de **468 × 3 = 1404**.
  - **Nota:** Este número puede cambiar si se usan menos landmarks o solo 2D.

### b. Capas Ocultas (Hidden Layers)
- **Función:** Estas capas procesan y transforman los datos de entrada para detectar patrones más complejos.
- **Componentes:**
  - Varias capas conectadas completamente (fully connected).
  - **Función de Activación:** Se usa **ReLU** (Rectified Linear Unit) en estas capas, lo que ayuda a que la red aprenda de forma eficiente y evita problemas como el desvanecimiento del gradiente.
- **Ejemplo de Configuración:**
  - Primera capa oculta: 512 neuronas.
  - Segunda capa oculta: 256 neuronas.
  - Tercera capa oculta: 128 neuronas (esto se puede ajustar según el problema).

### c. Capa de Salida (Output Layer)
- **Función:** Esta capa da la clasificación final de la emoción.
- **Detalles:**
  - El número de neuronas en la salida es igual al número de emociones que queremos identificar (por ejemplo, 5: alegría, tristeza, enojo, sorpresa y neutral).
  - **Función de Activación:** Se utiliza **Softmax**, que convierte la salida en probabilidades (la suma de estas es 1).

---

## 2. Patrones a Utilizar

- **Datos de Entrada:**  
  - Se usan vectores que contienen las coordenadas de cada landmark (*x, y, z*).
  - **Preprocesamiento:**  
    - Normalización de los valores para que estén en el mismo rango (por ejemplo, entre 0 y 1 o con media 0 y desviación estándar 1).
    - Se pueden extraer otras características como distancias o ángulos entre puntos que ayuden a identificar mejor la emoción.

---

## 3. Funciones de Activación Necesarias

- **Capas Ocultas:**  
  - **ReLU:** Es útil porque introduce no linealidad y ayuda a la red a aprender relaciones complejas.
  
- **Capa de Salida:**  
  - **Softmax:** Se usa para clasificar en múltiples categorías, transformando los valores en probabilidades.

---

## 4. Número Máximo de Entradas

- **Definición:**  
  - El número de entradas depende de la cantidad de landmarks.  
  - Por ejemplo, con 468 landmarks en 3D, tendremos **468 × 3 = 1404 entradas**.
  
- **Consideración:**  
  - Podemos reducir el número de entradas usando técnicas de selección de características si algunos landmarks no aportan información relevante.

---

## 5. Valores Esperados en la Salida de la Red

- **Salida de la Red:**  
  - Se espera obtener un vector que indique la probabilidad de cada emoción.  
  - Por ejemplo, para 5 emociones, la salida podría ser algo como:  
    ```
    [0.10, 0.20, 0.50, 0.10, 0.10]
    ```  
    donde cada número representa la probabilidad de que la entrada corresponda a cada emoción.

---

## 6. Valores Máximos que Puede Tener el Bias

- **Descripción:**  
  - El **bias** es un parámetro que se suma a la combinación lineal de entradas en cada neurona.
  
- **Valores Máximos:**  
  - **Teóricamente:** No hay un límite máximo definido para el bias; este se ajusta durante el entrenamiento.
  - **Prácticamente:**  
    - Se suele iniciar con valores pequeños (por ejemplo, entre **-0.1 y 0.1**) para evitar un sesgo muy grande desde el comienzo.
    - Durante el entrenamiento, el bias puede aumentar o disminuir, pero se controla usando técnicas de regularización.

---

## Resumen del Modelo

- **Tipo de Red:** Red Neuronal Feedforward Multicapa (MLP).
- **Entradas:**  
  - Se utilizan los landmarks normalizados de Mediapipe (por ejemplo, 1404 entradas para 468 landmarks en 3D).
- **Capas Ocultas:**  
  - Varias capas completamente conectadas que usan la función **ReLU**.
- **Capa de Salida:**  
  - Tiene tantas neuronas como emociones a clasificar, usando **Softmax** para obtener probabilidades.
- **Patrones:**  
  - Se usan vectores de coordenadas (y posibles características derivadas) que han sido preprocesados y normalizados.
- **Salida:**  
  - Se obtiene un vector de probabilidades, uno por cada emoción.
- **Bias:**  
  - Inicialmente con valores pequeños, sin un límite teórico máximo, pero controlados mediante regularización.

---


