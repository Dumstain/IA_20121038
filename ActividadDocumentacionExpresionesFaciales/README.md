# Documentación: Detección de vida  y Reconocimiento de Emociones Faciales

Esta documentación describe de manera detallada y paso a paso cómo desarrollar un sistema que permita:
- **Verificar que la persona está viva** (detección de *vida*).
- **Reconocer la emoción básica** que la persona está expresando.

El sistema se basa en la utilización de **MediaPipe Face Mesh** para extraer los landmarks faciales, y en técnicas geométricas y de machine learning para analizar los movimientos y expresiones faciales.


---

## 1. Introducción

El propósito principal de este sistema es mejorar la seguridad y la interacción en aplicaciones de reconocimiento facial, garantizando que se está tratando con una persona real y, además, interpretar sus estados emocionales. 
---

## 2. Requisitos y Dependencias

- **Lenguaje:** Python 3.x
- **Librerías:**  
  - [MediaPipe](https://mediapipe.dev/) – para la detección y seguimiento de landmarks faciales.
  - [OpenCV](https://opencv.org/) – para capturar y procesar imágenes en tiempo real.
  - [NumPy](https://numpy.org/) – para el manejo y cálculo de datos numéricos.
  - (Opcional) Librerías de visualización, como Matplotlib, para representar gráficos y resultados.

