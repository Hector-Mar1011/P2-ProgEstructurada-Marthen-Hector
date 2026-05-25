"""
Héctor Luis Garcia Marthen
Ux25II175
25/05/2026
Examen Segundo Parcial - Programación Estructurada
"""

# Importación de bibliotecas estándar
import datetime
import math
import random
import statistics
import sys
import time

MAX_EPOCHS = 10
UMBRAL_ERROR_CRITICO = 0.95

"""
Usando sys se valida el entorno de la ejecución mostrando información relevante del sistema,
como la plataforma y la versión de Python.
"""
def obtener_info_sistema():
    print("=== INFORMACIÓN DEL SISTEMA ===")
    print(f"Plataforma: {sys.platform}")
    print(f"Versión de Python: {sys.version}\n")


"""
Usando la biblioteca statistics analiza el rendimiento del entrenamiento
en base a las métricas recolectadas.
"""
def analizar_rendimiento(lista_loss, lista_latencia):
    print("\n=== ANÁLISIS ESTADÍSTICO DE RENDIMIENTO ===")
    if not lista_loss:
        print("No hay datos suficientes para analizar.")
        return

    # 1. Calcular la media (mean) de los valores de pérdida (loss) obtenidos.
    promedio_loss = statistics.mean(lista_loss)
    print(f"• Media de la Pérdida (Loss Mean): {promedio_loss:.4f}")
    
    # 2. Calcular la desviación estándar (stdev) para medir la estabilidad del entrenamiento.
    if len(lista_loss) > 1:
        desviacion_loss = statistics.stdev(lista_loss)
        print(f"• Desviación Estándar de Pérdida (Estabilidad): {desviacion_loss:.4f}")
    else:
        print("• Desviación Estándar de Pérdida: No aplicable (se requiere más de un Epoch).")
        
    # 3. Obtener la mediana (median) de la latencia del proceso.
    if lista_latencia:
        mediana_latencia = statistics.median(lista_latencia)
        print(f"• Mediana de la Latencia del Proceso: {mediana_latencia:.6f} segundos")
    else:
        print("• Mediana de la Latencia: No hay registros de tiempo disponibles.")
        
    print("-" * 75)


"""
Usando datetime gestionamos el tiempo de ejecución del entrenamiento,
mostrando la fecha y hora de inicio, formateando la fecha en un formato legible
y calculando la diferencia de tiempo simulada entre el inicio y el final.
"""
def simular_metricas_entrenamiento(cantidad_epochs):
    # Captura de inicio general
    fecha_inicio = datetime.datetime.now()
    formateo = fecha_inicio.strftime("%Y-%m-%d %H:%M:%S")
    
    print("---- Inicio de simulación ----")
    print(f"Fecha y hora de inicio: {formateo}\n")
    print(f"Iniciando simulación por {cantidad_epochs} epochs...")
    print("-" * 75)

    # Listas para almacenar los datos que procesará 'statistics'
    historial_perdidas = []
    historial_latencias = []
    
    eventos_posibles = [
        "Epoch exitoso", 
        "Gradiente inestable", 
        "Actualización de pesos óptma", 
        "Ajuste de tasa de aprendizaje"
    ]
    
    # --- Simulación del ciclo de entrenamiento ---
    for epoch in range(1, cantidad_epochs + 1):
        # Captura de tiempo para medir la latencia de ESTE epoch específico
        tiempo_inicio_epoch = datetime.datetime.now()
        
        # 1. Generar la fluctuación del error de pérdida (loss) usando decimales aleatorios
        loss = random.uniform(0.1, 1.2)
        historial_perdidas.append(loss)
        
        # 2. Simular la probabilidad de éxito de una iteración
        probabilidad_exito = random.random()
        
        # 3. Seleccionar de manera aleatoria un evento de log de una lista de strings
        evento_seleccionado = random.choice(eventos_posibles)
        
        # Pequeña pausa física real aleatoria para que la latencia no sea cero
        time.sleep(random.uniform(0.01, 0.05))
        
        # Cálculo de la latencia de la iteración actual en segundos
        tiempo_fin_epoch = datetime.datetime.now()
        latencia_epoch = (tiempo_fin_epoch - tiempo_inicio_epoch).total_seconds()
        historial_latencias.append(latencia_epoch)
        
        # Mostrar métricas integradas en la consola
        print(f"Epoch {epoch:02d}/{cantidad_epochs:02d} -> Pérdida: {loss:.4f} | Prob. Éxito: {probabilidad_exito:.2%} | Latencia: {latencia_epoch:.4f}s | Log: [{evento_seleccionado}]")
        
        # Condición para sys.exit() ante un error crítico basado en el umbral
        if loss > UMBRAL_ERROR_CRITICO:
            print(f"\n[CRÍTICO] Pérdida ({loss:.4f}) superó el umbral permitido ({UMBRAL_ERROR_CRITICO}).")
            # LLAMADA DE EMERGENCIA: Mostramos las estadísticas calculadas hasta el momento
            analizar_rendimiento(historial_perdidas, historial_latencias)
            print("Forzando salida limpia del sistema...")
            sys.exit(1) 
            
    print("-" * 75)
    print("¡Entrenamiento completado con éxito!")
    
    # LLAMADA NORMAL: Muestra las estadísticas finales tras completar todos los epochs
    analizar_rendimiento(historial_perdidas, historial_latencias)
    
    # Captura de fin y cálculo de duración total general con datetime
    fecha_fin = datetime.datetime.now()
    duracion = fecha_fin - fecha_inicio
    print(f"Fecha y hora de finalización: {fecha_fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duración total del proceso: {duracion}\n")


def main():
    obtener_info_sistema()
    simular_metricas_entrenamiento(MAX_EPOCHS)


if __name__ == "__main__":
    main()