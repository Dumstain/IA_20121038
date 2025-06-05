#!/usr/bin/env python3
"""
Script para entrenar todos los modelos de IA del juego
Ejecuta los tres algoritmos: Árbol de Decisión, Red Neuronal y K-Vecinos Cercanos
"""

import subprocess
import sys
import os

def ejecutar_script(nombre_script):
    """Ejecuta un script de Python y maneja errores"""
    print(f"\n{'='*50}")
    print(f"EJECUTANDO: {nombre_script}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, nombre_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Advertencias:", result.stderr)
        print(f"✅ {nombre_script} completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando {nombre_script}:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo: {nombre_script}")
        return False

def main():
    print("🚀 INICIANDO ENTRENAMIENTO DE TODOS LOS MODELOS DE IA")
    print("Este proceso entrenará los tres algoritmos: Árbol, Red Neuronal y KNN")
    
    # Verificar que existe el archivo de datos
    datos_file = 'datos_combinados_ia.csv'
    if not os.path.exists(datos_file):
        print(f"❌ ERROR: No se encontró el archivo '{datos_file}'")
        print("Primero necesitas jugar en modo manual para generar datos de entrenamiento.")
        return
    
    # Lista de scripts de entrenamiento
    scripts = [
        'train_arbol.py',      # Árbol de Decisión
        'train_red_neuronal.py',   # Red Neuronal  
        'train_knn.py'         # K-Vecinos Cercanos
    ]
    
    resultados = {}
    
    for script in scripts:
        if os.path.exists(script):
            resultados[script] = ejecutar_script(script)
        else:
            print(f"⚠️  ADVERTENCIA: No se encontró {script}")
            resultados[script] = False
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE ENTRENAMIENTO")
    print(f"{'='*60}")
    
    exitosos = 0
    fallidos = 0
    
    for script, exito in resultados.items():
        estado = "✅ EXITOSO" if exito else "❌ FALLIDO"
        print(f"{script:25} {estado}")
        if exito:
            exitosos += 1
        else:
            fallidos += 1
    
    print(f"\nModelos entrenados exitosamente: {exitosos}")
    print(f"Modelos con errores: {fallidos}")
    
    if exitosos > 0:
        print(f"\n🎮 Ahora puedes usar el juego en modo automático con {exitosos} algoritmo(s) disponible(s)")
        print("Los archivos de modelo se guardaron en la carpeta 'modelos_entrenados/'")
    
    if fallidos > 0:
        print(f"\n⚠️  Algunos modelos no se pudieron entrenar. Revisa los errores arriba.")

if __name__ == "__main__":
    main()