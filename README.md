# Simulación Serial y Paralela del Campo Magnético de Curvas Paramétricas 3D (Ley de Biot-Savart)

Proyecto final — Proyecto final: simulación serial y paralela de un fenómeno físico


## Descripción
Implementación numérica y análisis de desempeño (speedup, eficiencia y ajuste de la Ley de Amdahl) para el cálculo del campo magnético $\vec{B}(x, y, z)$ producido por conductores filiformes descritos por curvas paramétricas tridimensionales (solenoide finito, espiral de Arquímedes, nudo de trébol y curvas arbitrarias) mediante la Ley de Biot-Savart discreta.

## Estructura del Repositorio
* `serial.py`: Implementación secuencial de referencia compilada en un hilo con `@njit` e interfaz gráfica interactiva con deslizador en el eje $z$.
* `paralelo.py`: Implementación paralela en CPU mediante `numba.njit(parallel=True)` y `numba.prange`, distribuyendo las filas de la malla espacial entre los hilos disponibles.
* `medir.py`: Script automatizado de verificación numérica (`np.allclose`), benchmark con `time.perf_counter()`, exportación de métricas a CSV y regeneración de todas las figuras.
* `requirements.txt`: Dependencias de Python necesarias para ejecutar el proyecto.

## Entorno de Ejecución y Hardware
* **Procesador:** AMD Ryzen 5 5600X (6 núcleos físicos / 12 hilos lógicos SMT)
* **Lenguaje:** Python 3.11+
* **Librerías principales:** NumPy, Numba, SciPy, Matplotlib

## Instalación
1. Crear y activar un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt  o en su defecto pip install numpy numba matplotlib scipy #librerias 