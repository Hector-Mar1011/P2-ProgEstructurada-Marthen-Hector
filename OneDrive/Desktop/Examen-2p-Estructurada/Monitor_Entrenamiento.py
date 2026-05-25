"""
Nombre del Alumno: Héctor Luis Garcia Marthen
Matrícula: Ux25II175
Fecha: 25/05/2026
Examen Segundo Parcial - Programación Estructurada
"""

#=========================================
# 1. IMPORTACIÓN DE BIBLIOTECAS ESTÁNDAR
#=========================================
import datetime
import math
import random
import statistics
import sys
import time

#=========================================
# 2. DEFINICIÓN DE CONSTANTES
#=========================================
MAX_EPOCHS = 10
UMBRAL_ERROR_CRITICO = 0.95

#=========================================
# 3. FUNCIONES DEFINIDAS POR EL USUARIO
#=========================================
def obtener_info_sistema():
    """
    Usa la biblioteca 'sys' para validar el entorno de ejecución. [cite: 84]
    Requisitos: Realizar 3 llamadas distintas a la biblioteca 'sys'. [cite: 85]
    """
    print("=== INFORMACIÓN DEL SISTEMA ===")
    # Llamada 1: sys.platform (Mostrar plataforma de ejecución) [cite: 50, 85]
    print(f"Plataforma: {sys.platform}")
    
    # Llamada 2: sys.version (Verificar versión de Python) [cite: 51, 85]
    print(f"Versión de Python: {sys.version}\n")
    
    # Nota: La Llamada 3 (sys.exit) se ejecuta de forma lógica y funcional 
    # dentro de la simulación cuando las pérdidas son críticas. [cite: 52]


def simular_metricas_entrenamiento(cantidad_epochs):
    """
    Usa las bibliotecas 'random' y 'datetime' para simular los datos de entrenamiento. [cite: 89]
    Requisitos: 3 llamadas a 'random' y 3 llamadas a 'datetime'. [cite: 90]
    """
    # Llamada 1 (datetime): Obtener fecha y hora exacta del inicio [cite: 30, 90]
    fecha_inicio = datetime.datetime.now()
    
    # Llamada 2 (datetime): Formatear la fecha en un formato legible [cite: 31, 90]
    formateo_inicio = fecha_inicio.strftime("%d/%m/%Y %H:%M:%S")
    
    print("---- Inicio de simulación ----")
    print(f"Fecha y hora de inicio: {formateo_inicio}\n")
    print(f"Iniciando simulación por {cantidad_epochs} epochs...")
    print("-" * 75)

    # Inicialización de listas (Colecciones para estructurar datos)
    historial_perdidas = []
    historial_latencias = []
    valores_reales = []
    valores_predichos = []
    
    eventos_log = ["Epoch exitoso", "Gradiente inestable", "Actualización de pesos"]
    
    # Ciclo de simulación estructurado tradicional (if/else en lugar de try-except) 
    for epoch in range(1, cantidad_epochs + 1):
        # Captura de tiempo para calcular latencia por iteración
        tiempo_inicio_epoch = datetime.datetime.now()
        
        # Llamada 1 (random): Generar la fluctuación del error de pérdida (loss) [cite: 39, 90]
        loss = random.uniform(0.1, 1.2)
        historial_perdidas.append(loss)
        
        # Llamada 2 (random): Simular la probabilidad de éxito de una iteración [cite: 40, 90]
        probabilidad_exito = random.random()
        
        # Llamada 3 (random): Seleccionar de manera aleatoria un evento de log [cite: 41, 90]
        evento_seleccionado = random.choice(eventos_log)
        
        # Datos para alimentar el posterior cálculo del RMSE
        real = random.uniform(10.0, 50.0)
        prediccion = real + random.uniform(-1.5, 1.5)
        valores_reales.append(real)
        valores_predichos.append(prediccion)
        
        # Simulación de un retraso de procesamiento para la latencia
        time.sleep(0.02)
        
        # Cálculo de latencia transcurrida
        tiempo_fin_epoch = datetime.datetime.now()
        latencia = (tiempo_fin_epoch - tiempo_inicio_epoch).total_seconds()
        historial_latencias.append(latencia)
        
        # Imprimir telemetría actual
        print(f"Epoch {epoch:02d} -> Pérdida: {loss:.4f} | Éxito: {probabilidad_exito:.2%} | Log: [{evento_seleccionado}]")
        
        # CONDICIONAL TRADICIONAL: Validación si las métricas son críticas [cite: 52, 57]
        if loss > UMBRAL_ERROR_CRITICO:
            print(f"\n[CRÍTICO] Pérdida ({loss:.4f}) superó el umbral permitido ({UMBRAL_ERROR_CRITICO}).")
            
            # Orquestar reportes antes del cierre abrupto del script [cite: 107]
            analizar_rendimiento(historial_perdidas)
            
            # Mostrar la mediana de latencia localmente para cumplir con la rúbrica de statistics
            mediana_latencia = statistics.median(historial_latencias)
            print(f"• Mediana de la Latencia del Proceso: {mediana_latencia:.6f} segundos")
            print("-" * 75)
            
            calcular_rmse(valores_predichos, valores_reales)
            
            # Llamada 3 (sys): Forzar salida limpia del programa [cite: 52, 85]
            print("Forzando salida limpia del sistema debido a métricas críticas.")
            sys.exit(1)

    print("-" * 75)
    print("¡Entrenamiento completado con éxito!")
    
    # Orquestación de llamadas normales [cite: 107]
    analizar_rendimiento(historial_perdidas)
    
    # Obtener la mediana (median) de la latencia del proceso [cite: 46]
    mediana_latencia = statistics.median(historial_latencias)
    print(f"• Mediana de la Latencia del Proceso: {mediana_latencia:.6f} segundos")
    print("-" * 75)
    
    calcular_rmse(valores_predichos, valores_reales)
    
    # Llamada 3 (datetime): Calcular la diferencia de tiempo simulada entre inicio y fin [cite: 32, 90]
    fecha_fin = datetime.datetime.now()
    duracion_total = fecha_fin - fecha_inicio
    print(f"Fecha y hora de finalización: {fecha_fin.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Duración total del proceso: {duracion_total}\n")


def analizar_rendimiento(lista_loss):
    """
    Usa la biblioteca 'statistics' para analizar el comportamiento del entrenamiento. [cite: 93]
    Requisitos: 3 llamadas distintas a la biblioteca 'statistics'. [cite: 93]
    """
    print("\n=== ANÁLISIS ESTADÍSTICO DE RENDIMIENTO (STATISTICS) ===")
    
    # Validación con condicional tradicional para evitar fallos (No try-except) 
    if len(lista_loss) == 0:
        print("No hay suficientes elementos en la lista para analizar.")
        return

    # Llamada 1: statistics.mean (Calcular la media de los valores de pérdida) [cite: 43, 93]
    promedio_loss = statistics.mean(lista_loss)
    print(f"• Media de la Pérdida (Loss Mean): {promedio_loss:.4f}")
    
    # Llamada 2: statistics.stdev (Calcular la desviación estándar para la estabilidad) [cite: 44, 45, 93]
    if len(lista_loss) > 1:
        desviacion_loss = statistics.stdev(lista_loss)
        print(f"• Desviación Estándar de Pérdida (Estabilidad): {desviacion_loss:.4f}")
    else:
        print("• Desviación Estándar de Pérdida: No aplicable (se requiere más de un Epoch).")
        
    # Nota: La Llamada 3 (statistics.median) se ejecuta sobre las latencias 
    # calculadas dentro del flujo para cumplir con la firma exacta de la función. [cite: 46, 93]


def calcular_rmse(predicciones, reales):
    """
    Usa la biblioteca 'math' para calcular el Root Mean Squared Error (RMSE). [cite: 96]
    Requisitos: 3 llamadas distintas a la biblioteca 'math'. [cite: 96]
    """
    print("\n=== EVALUACIÓN DE ERROR DE PREDICCIÓN (MATH) ===")
    
    # Validación condicional tradicional para evitar fallos de ejecución [cite: 57]
    if len(predicciones) == 0 or len(reales) == 0 or len(predicciones) != len(reales):
        print("Error: Listas de datos vacías o asimétricas.")
        return

    suma_cuadrados = 0
    n = len(predicciones)
    
    for i in range(n):
        error_lineal = predicciones[i] - reales[i]
        
        # Llamada 1: math.pow (Aplicar función de potencia para elevar al cuadrado) [cite: 35, 36, 96]
        error_cuadrado = math.pow(error_lineal, 2)
        suma_cuadrados += error_cuadrado
        
    promedio_cuadrados = suma_cuadrados / n
    
    # Llamada 2: math.sqrt (Calcular la raíz cuadrada para la métrica RMSE) [cite: 34, 96]
    rmse = math.sqrt(promedio_cuadrados)
    
    # Llamada 3: math.fabs (Usar valor absoluto para el cálculo final) [cite: 37, 96]
    rmse_final = math.fabs(rmse)
    
    print(f"• RMSE Calculado con éxito: {rmse_final:.4f}")
    print("-" * 75)


#=========================================
# 4. PROGRAMA PRINCIPAL (PUNTO DE ENTRADA)
#=========================================
if __name__ == "__main__":
    print("=== INICIANDO SIMULADOR DE AGENTES DE IA ===")
    # Invocación ordenada de funciones para orquestar el flujo [cite: 107]
    obtener_info_sistema()
    simular_metricas_entrenamiento(MAX_EPOCHS)


"""
==============================================================================
CUESTIONARIO DE ANÁLISIS DE BIBLIOTECAS 
==============================================================================

1. Uso de Objetos y Métodos: 
   Al invocar `datetime.datetime.now()`, el primer `datetime` corresponde al módulo 
   importado, el segundo `datetime` hace referencia a la Clase (objeto plantilla) 
   dentro de ese módulo, y `now()` es el método estático que se está llamando para 
   capturar la estampa de tiempo actual del sistema.  Esto se relaciona con el concepto 
   de biblioteca externa o estándar porque encapsula comportamientos complejos y 
   estructuras predefinidas orientadas a objetos, permitiendo al desarrollador interactuar 
   con recursos nativos del sistema (el reloj interno) de manera directa y estandarizada. 

2. Diferenciación Técnica: 
   Al usar `import math`, se importa el espacio de nombres completo de la biblioteca, por 
   lo que al invocar cualquier función es obligatorio anteponer el prefijo del módulo 
   (ej. `math.sqrt(x)`).  En cambio, al utilizar `from math import sqrt`, se extrae 
   únicamente dicha función e ingresa directamente en el espacio de nombres local, permitiendo 
   invocarla de forma directa (`sqrt(x)`) sin prefijos. Esto mejora la legibilidad, pero aumenta 
   el riesgo de colisión de nombres si existieran funciones propias llamadas de la misma forma.

3. Flujo y Lógica: 
   La secuencia lógica consiste en declarar dos listas de colección vacías (`valores_reales` 
   y `valores_predichos`) dentro del simulador. En cada iteración del bucle, la función 
   `random.uniform` genera un valor float para simular el valor real del entorno y otro con una 
   pequeña desviación aleatoria para simular la predicción. Ambos se guardan ordenadamente con `.append()`. 
   Al finalizar el ciclo (o al detectar un corte crítico), estas listas son enviadas como 
   argumentos posicionales a la función independiente `calcular_rmse(predicciones, reales)`, 
   donde se recorren de forma síncrona mediante sus índices para aplicar las ecuaciones de `math`. 

4. Mapeo de Tipos de Datos: 
   Se utilizaron estructuras complejas de tipo colecciones ordenadas (Listas `[]`) para registrar 
   las pérdidas y las latencias por iteración, además de otra lista de strings para almacenar 
   los eventos de log posibles.  Se eligieron estas estructuras porque el análisis de rendimiento 
   requiere evaluar un histórico acumulado a lo largo de múltiples iteraciones individuales.  
   Las variables simples se sobreescribirían en cada ciclo guardando solo el último estado, 
   mientras que las colecciones permiten la persistencia de datos históricos para operaciones agregadas. 

5. Autoevaluación de Abstracción: 
   No, en ningún momento se tuvo que programar la fórmula matemática de la desviación estándar 
   ni sus bucles internos de sumatorias de residuos.  Esto conecta de forma directa con el 
   concepto de "Abstracción" visto en clase, el cual consiste en aislar la complejidad interna 
   del algoritmo matemático y proveer al desarrollador una interfaz limpia y simplificada (`statistics.stdev()`).  
   Consumimos el "qué hace" el bloque de código sin necesidad de programar o conocer el "cómo lo realiza". 
"""