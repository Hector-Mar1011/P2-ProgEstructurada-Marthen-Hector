"""
Nombre del Alumno: Héctor Luis Garcia Marthen
Matrícula: Ux25II175
Fecha: 25/05/2026
Examen Segundo Parcial - Programación Estructurada
"""

# ==========================================
# 1. IMPORTACIÓN DE BIBLIOTECAS ESTÁNDAR
# ==========================================
import datetime
import math
import random
import statistics
import sys
import time

# ==========================================
# 2. DEFINICIÓN DE CONSTANTES
# ==========================================
MAX_EPOCHS = 10
UMBRAL_ERROR_CRITICO = 0.95

# ==========================================
# 3. FUNCIONES DEFINIDAS POR EL USUARIO
# ==========================================

def obtener_info_sistema():
    """
    Usa la biblioteca 'sys' para validar el entorno de ejecución.
    Requisitos: Realizar 3 llamadas distintas a la biblioteca 'sys'.
    """
    print("=== INFORMACIÓN DEL SISTEMA ===")
    # Llamada 1: sys.platform (Detectar el sistema operativo)
    print(f"Plataforma de ejecución: {sys.platform}")
    
    # Llamada 2: sys.version (Verificar la versión del intérprete)
    print(f"Versión de Python instalada: {sys.version}\n")
    
    # Nota: La llamada 3 (sys.exit) está implementada estratégicamente 
    # en la función de simulación para actuar ante pérdidas críticas.


def analizar_rendimiento(lista_loss, lista_latencia):
    """
    Usa la biblioteca 'statistics' para analizar el comportamiento del entrenamiento.
    Requisitos: 3 llamadas distintas a la biblioteca 'statistics'.
    """
    print("\n=== ANÁLISIS ESTADÍSTICO DE RENDIMIENTO (STATISTICS) ===")
    
    # Validación con condicional tradicional (Regla de oro: No try-except)
    if len(lista_loss) == 0:
        print("No hay datos de pérdida suficientes para analizar.")
        return

    # Llamada 1: statistics.mean (Media de pérdida)
    promedio_loss = statistics.mean(lista_loss)
    print(f"• Media de la Pérdida (Loss Mean): {promedio_loss:.4f}")
    
    # Llamada 2: statistics.stdev (Desviación estándar para medir estabilidad)
    if len(lista_loss) > 1:
        desviacion_loss = statistics.stdev(lista_loss)
        print(f"• Desviación Estándar de Pérdida (Estabilidad): {desviacion_loss:.4f}")
    else:
        print("• Desviación Estándar de Pérdida: No aplicable (se requiere más de un Epoch).")
        
    # Llamada 3: statistics.median (Mediana de la latencia del proceso)
    if len(lista_latencia) > 0:
        mediana_latencia = statistics.median(lista_latencia)
        print(f"• Mediana de la Latencia del Proceso: {mediana_latencia:.6f} segundos")
        
    print("-" * 75)


def calcular_rmse(predicciones, reales):
    """
    Usa la biblioteca 'math' para calcular el Root Mean Squared Error (RMSE).
    Requisitos: 3 llamadas distintas a la biblioteca 'math'.
    """
    print("\n=== EVALUACIÓN DE ERROR DE PREDICCIÓN (MATH) ===")
    
    # Validación condicional tradicional para evitar fallos de ejecución
    if len(predicciones) == 0 or len(reales) == 0 or len(predicciones) != len(reales):
        print("Error: Listas de datos asimétricas o vacías.")
        return

    suma_cuadrados = 0
    n = len(predicciones)
    
    for i in range(n):
        error_lineal = predicciones[i] - reales[i]
        
        # Llamada 1: math.pow (Elevar las diferencias al cuadrado)
        error_cuadrado = math.pow(error_lineal, 2)
        suma_cuadrados += error_cuadrado
        
    promedio_cuadrados = suma_cuadrados / n
    
    # Llamada 2: math.sqrt (Calcular la raíz cuadrada para el RMSE)
    rmse = math.sqrt(promedio_cuadrados)
    
    # Llamada 3: math.fabs (Garantizar valor absoluto del error final)
    rmse_final = math.fabs(rmse)
    
    print(f"• RMSE Calculado con éxito: {rmse_final:.4f}")
    print("-" * 75)


def simular_metricas_entrenamiento(cantidad_epochs):
    """
    Usa las bibliotecas 'random' y 'datetime' para simular los datos de entrenamiento.
    Requisitos: 3 llamadas a 'random' y 3 llamadas a 'datetime'.
    """
    # Llamada 1 (datetime): datetime.datetime.now() para capturar el inicio exacto
    fecha_inicio = datetime.datetime.now()
    
    # Llamada 2 (datetime): strftime() para dar formato legible en español
    formateo_inicio = fecha_inicio.strftime("%d/%m/%Y %H:%M:%S")
    
    print("---- Inicio de simulación ----")
    print(f"Fecha y hora de inicio: {formateo_inicio}\n")
    print(f"Iniciando simulación por {cantidad_epochs} epochs...")
    print("-" * 75)

    # Estructuras de datos complejas para recolectar métricas
    historial_perdidas = []
    historial_latencias = []
    valores_reales = []
    valores_predichos = []
    
    eventos_log = ["Epoch exitoso", "Gradiente inestable", "Actualización de pesos"]
    
    # Ciclo estructurado tradicional
    for epoch in range(1, cantidad_epochs + 1):
        tiempo_inicio_epoch = datetime.datetime.now()
        
        # Llamada 1 (random): random.uniform para fluctuación de error (loss)
        loss = random.uniform(0.1, 1.2)
        historial_perdidas.append(loss)
        
        # Llamada 2 (random): random.random para probabilidad de éxito de la iteración
        probabilidad_exito = random.random()
        
        # Llamada 3 (random): random.choice para seleccionar evento de la lista de strings
        evento_seleccionado = random.choice(eventos_log)
        
        # Datos complementarios para alimentar el RMSE posterior
        real = random.uniform(10.0, 50.0)
        prediccion = real + random.uniform(-1.5, 1.5)
        valores_reales.append(real)
        valores_predichos.append(prediccion)
        
        # Pequeña pausa real para registrar latencia perceptible
        time.sleep(0.02)
        
        # Cálculo de latencia por iteración
        tiempo_fin_epoch = datetime.datetime.now()
        latencia = (tiempo_fin_epoch - tiempo_inicio_epoch).total_seconds()
        historial_latencias.append(latencia)
        
        # Mostrar telemetría en consola
        print(f"Epoch {epoch:02d} -> Pérdida: {loss:.4f} | Éxito: {probabilidad_exito:.2%} | Latencia: {latencia:.4f}s | Log: [{evento_seleccionado}]")
        
        # Validación de condición crítica mediante condicional tradicional
        if loss > UMBRAL_ERROR_CRITICO:
            print(f"\n[CRÍTICO] Pérdida descontrolada ({loss:.4f}) superó el umbral ({UMBRAL_ERROR_CRITICO}).")
            
            # Orquestación de reportes de emergencia antes de forzar el cierre
            analizar_rendimiento(historial_perdidas, historial_latencias)
            calcular_rmse(valores_predichos, valores_reales)
            
            # Llamada 3 (sys): Forzar salida limpia del programa por error crítico
            print("Forzando salida limpia del sistema debido a métricas críticas.")
            sys.exit(1)

    print("-" * 75)
    print("¡Entrenamiento completado de manera exitosa!")
    
    # Orquestación de reportes normales al finalizar el ciclo completo
    analizar_rendimiento(historial_perdidas, historial_latencias)
    calcular_rmse(valores_predichos, valores_reales)
    
    # Llamada 3 (datetime): Calcular diferencia de tiempo entre inicio y fin
    fecha_fin = datetime.datetime.now()
    duracion_total = fecha_fin - fecha_inicio
    print(f"Fecha y hora de finalización: {fecha_fin.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Duración total de la simulación: {duracion_total}\n")


# ==========================================
# 4. PROGRAMA PRINCIPAL (PUNTO DE ENTRADA)
# ==========================================
if __name__ == "__main__":
    print("=== INICIANDO SIMULADOR DE AGENTES DE IA ===")
    obtener_info_sistema()
    simular_metricas_entrenamiento(MAX_EPOCHS)


"""
==============================================================================
PARTE 3: CUESTIONARIO DE ANÁLISIS DE BIBLIOTECAS
==============================================================================

1. Uso de Objetos y Métodos:
   Al invocar `datetime.datetime.now()`, el primer `datetime` corresponde al módulo 
   importado, el segundo `datetime` hace referencia a la Clase (u objeto plantilla) 
   dentro de ese módulo, y `now()` es el método estático que se está ejecutando para 
   extraer la estampa de tiempo actual del sistema. Esto se relaciona con el concepto 
   de biblioteca externa o estándar porque encapsula comportamientos complejos y 
   estructuras predefinidas orientadas a objetos, permitiendo al desarrollador interactuar 
   con recursos nativos del sistema sin programar la captura del reloj desde cero.

2. Diferenciación Técnica:
   Al usar `import math`, se importa el espacio de nombres completo de la biblioteca. 
   Por ende, para invocar cualquier función, es obligatorio anteponer el prefijo del módulo 
   (ej. `math.sqrt(x)`). En contraste, al utilizar `from math import sqrt`, se extrae 
   únicamente dicha función e ingresa directamente en el espacio de nombres local del script, 
   permitiendo invocarla de manera directa (`sqrt(x)`) sin prefijos, optimizando la legibilidad 
   pero aumentando el riesgo de colisión de nombres si existieran funciones homónimas.

3. Flujo y Lógica:
   La secuencia lógica consiste en declarar dos listas estructuradas vacías (`valores_reales` 
   y `valores_predichos`) dentro de la función de simulación. En cada iteración del bucle 
   `for` de los epochs, la función `random.uniform` genera un valor float para simular el valor 
   de referencia real y otro con una pequeña desviación para simular la predicción del agente. 
   Ambos valores se almacenan concurrentemente en sus respectivas listas mediante `.append()`. 
   Al finalizar el ciclo (o detectar un corte crítico), estas colecciones son enviadas como 
   argumentos posicionales a la función independiente `calcular_rmse(predicciones, reales)`, 
   la cual indexa y empareja secuencialmente los datos para aplicar las operaciones aritméticas de `math`.

4. Mapeo de Tipos de Datos:
   Se utilizaron dos tipos de datos complejos de tipo colección ordenable: las Listas (`[]`) 
   para el almacenamiento secuencial de las fluctuaciones flotantes de pérdida y latencia, y 
   las Listas de Cadenas de Texto para los eventos posibles del Log. Se eligieron estas 
   estructuras en lugar de variables simples porque el análisis de rendimiento del entrenamiento 
   requiere evaluar un comportamiento histórico continuo a lo largo de múltiples iteraciones. 
   Las variables simples se sobreescribirían en cada ciclo guardando solo el último estado, 
   mientras que las colecciones preservan la trazabilidad completa, permitiendo realizar cálculos 
   agregados estadísticos y probabilísticos finales.

5. Autoevaluación de Abstracción:
   No, en ningún momento se tuvo que programar manualmente la ecuación matemática de la 
   desviación estándar poblacional o muestral, ni los bucles internos de sumatorias de residuos. 
   Esto conecta directamente con el concepto de "Abstracción", el cual consiste en ocultar 
   la complejidad interna del backend operativo subyacente y proveer una interfaz de uso pública 
   y simplificada (`statistics.stdev()`). Como programadores, consumimos el "qué hace" la función 
   sin necesidad de preocuparnos ni reconstruir el "cómo lo hace" internamente.
"""