# Perceptrón Multicapa (MLP) con Interfaz Gráfica

Este proyecto implementa un Perceptrón Multicapa (MLP) en Python, acompañado de una interfaz gráfica de usuario (GUI) construida con Tkinter. Permite a los usuarios crear, entrenar, evaluar, guardar y cargar modelos de MLP para tareas de clasificación.

## Descripción General

El objetivo principal es proporcionar una herramienta educativa y experimental para entender el funcionamiento de las redes neuronales de tipo MLP. La implementación cubre los componentes esenciales:

*   **Capas (Layers)**: Definición de capas neuronales con pesos, biases y funciones de activación.
*   **MLP**: Orquestación de múltiples capas, incluyendo propagación hacia adelante (feedforward) y retropropagación del error (backpropagation) para el entrenamiento.
*   **GUI**: Una interfaz intuitiva para interactuar con el MLP, visualizar el progreso del entrenamiento y realizar predicciones.

## Características Principales

*   **Creación Dinámica de Redes**:
    *   Especificar el número de neuronas de entrada y salida.
    *   Definir múltiples capas ocultas, cada una con un número específico de neuronas.
    *   Seleccionar funciones de activación (Sigmoide, ReLU, Tanh) para las capas ocultas y la capa de salida (incluyendo 'linear' para la salida).
*   **Entrenamiento del Modelo**:
    *   Cargar conjuntos de datos de entrenamiento y prueba desde archivos de texto (`.txt` o `.csv`).
    *   Configurar hiperparámetros: número de épocas, tasa de aprendizaje y tamaño del lote (batch size).
    *   Opción para "Continuar Entrenamiento" sobre un modelo ya entrenado.
    *   El entrenamiento se ejecuta en un hilo separado para mantener la GUI responsiva.
*   **Evaluación y Visualización**:
    *   Cálculo de la precisión en los conjuntos de entrenamiento y prueba después de cada época.
    *   Visualización gráfica de la precisión del entrenamiento y prueba a lo largo de las épocas.
    *   Logs detallados de las operaciones y el progreso del entrenamiento.
*   **Predicción (Feedforward)**:
    *   Ingresar datos manualmente para obtener predicciones.
    *   Cargar un archivo de datos para realizar predicciones en lote.
    *   Opción para guardar los resultados de las predicciones en un archivo.
*   **Persistencia del Modelo**:
    *   Guardar la arquitectura y los pesos del modelo entrenado en formato JSON.
    *   Cargar modelos previamente guardados para continuar el entrenamiento o realizar predicciones.
*   **Manejo de Salida**: Redirección de `stdout` y `stderr` al área de logs de la GUI para una mejor depuración y seguimiento.

## Requisitos

*   Python 3.x
*   Tkinter (generalmente incluido con las instalaciones estándar de Python)
*   NumPy
*   Matplotlib

Puedes instalar las bibliotecas necesarias usando pip:
```bash
pip install numpy matplotlib
```

## Cómo Ejecutar la Aplicación

1.  Clona o descarga este repositorio.
2.  Asegúrate de tener todos los archivos Python (`main.py`, `perceptron_gui.py`, `multilayer_perceptron.py`, `layer.py`) en el mismo directorio.
3.  Ejecuta el script principal desde tu terminal:
    ```bash
    python main.py
    ```
    Esto abrirá la interfaz gráfica del Perceptrón Multicapa.

## Usando la GUI

La interfaz está dividida en secciones:

1.  **Inicialización de Red**:
    *   **Crear Nueva Red**: Abre un diálogo para definir la arquitectura de un nuevo MLP.
    *   **Cargar Red desde Archivo**: Permite cargar un modelo MLP previamente guardado (archivo `.json`).
2.  **Información de la Red**: Muestra el estado actual y la arquitectura de la red cargada/creada.
3.  **Acciones**: Botones para interactuar con la red (se habilitan una vez que hay una red activa).
    *   **Entrenar Red**: Abre el diálogo para configurar y comenzar un nuevo entrenamiento.
    *   **Continuar Entrenamiento**: Abre el diálogo para continuar entrenando la red actual.
    *   **Ejecutar Red (Feedforward)**: Permite ingresar datos manualmente o desde un archivo para obtener predicciones.
    *   **Guardar Red**: Guarda la red actual (arquitectura y pesos) en un archivo `.json`.
    *   **Mostrar Gráfico de Precisión**: Muestra un gráfico del historial de precisión del entrenamiento.
4.  **Salida/Logs**: Muestra mensajes de estado, progreso del entrenamiento y errores.

## Ejemplos de Problemas y Cómo Probarlos

A continuación, se explica cómo probar el MLP con algunos conjuntos de datos clásicos. Asegúrate de tener los archivos de datos en las rutas especificadas o ajusta las rutas en la GUI al cargar los archivos.

### Preparación General de Datos

*   Los archivos de datos de entrada deben ser archivos de texto (`.txt` o `.csv`) donde cada fila representa una muestra y los valores (características) están separados por comas.
*   Los archivos de datos de salida deben seguir el mismo formato, con cada fila correspondiendo a la salida esperada para la muestra de entrada respectiva.

### 1. Problema XOR (OR Exclusivo)

Este es un problema clásico no linealmente separable.

*   **Archivos de Datos**:
    *   Entrenamiento (Entradas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\xor_data\xor_train_inputs.txt`
    *   Entrenamiento (Salidas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\xor_data\xor_train_outputs.txt`
    *   Prueba (Entradas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\xor_data\xor_test_inputs.txt`
    *   Prueba (Salidas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\xor_data\xor_test_outputs.txt`

    Contenido de `xor_train_inputs.txt` (y `xor_test_inputs.txt`):
    ```
    0,0
    0,1
    1,0
    1,1
    ```
    Contenido de `xor_train_outputs.txt` (y `xor_test_outputs.txt`):
    ```
    0
    1
    1
    0
    ```

*   **Configuración de Red en la GUI (al "Crear Nueva Red")**:
    *   Número de neuronas de entrada: `2`
    *   Número de neuronas de salida: `1`
    *   Capas ocultas (neuronas por capa, ej: 5,3): `4` (esto crea una capa oculta con 4 neuronas)
    *   Activación capas ocultas: `sigmoid`
    *   Activación capa de salida: `sigmoid`

*   **Parámetros de Entrenamiento en la GUI**:
    *   Número de épocas: `5000` (o más, ej. `10000`)
    *   Tasa de aprendizaje (η): `0.1` (puedes probar hasta `0.5`)
    *   Tamaño del lote (Batch Size): `1` (o `4`)

*   **Resultado Esperado**: La red debería alcanzar una precisión del 100% en los datos de entrenamiento y prueba.

### 2. Problema XNOR (OR Exclusivo Negado)

La operación lógica inversa a XOR.

*   **Archivos de Datos**:
    *   Entrenamiento (Entradas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\nxor_data\nxor_train_inputs.txt`
    *   Entrenamiento (Salidas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\nxor_data\nxor_train_outputs.txt`
    *   Prueba (Entradas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\nxor_data\nxor_test_inputs.txt`
    *   Prueba (Salidas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\nxor_data\nxor_test_outputs.txt`

    Contenido de `nxor_train_inputs.txt` (y `nxor_test_inputs.txt`):
    ```
    0,0
    0,1
    1,0
    1,1
    ```
    Contenido de `nxor_train_outputs.txt` (y `nxor_test_outputs.txt`):
    ```
    1
    0
    0
    1
    ```

*   **Configuración de Red en la GUI**: Similar a XOR.
    *   Número de neuronas de entrada: `2`
    *   Número de neuronas de salida: `1`
    *   Capas ocultas: `4`
    *   Activación capas ocultas: `sigmoid`
    *   Activación capa de salida: `sigmoid`

*   **Parámetros de Entrenamiento en la GUI**:
    *   Número de épocas: `5000` - `10000`
    *   Tasa de aprendizaje (η): `0.1` - `0.5`
    *   Tamaño del lote (Batch Size): `1` (o `4`)

*   **Resultado Esperado**: Precisión del 100%.

### 3. Problema del Círculo (Clasificación Dentro/Fuera)

Clasificar puntos 2D según si están dentro o fuera de un círculo imaginario.

*   **Archivos de Datos**:
    *   Entrenamiento (Entradas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\circle_train\circle_train_inputs.txt`
    *   Entrenamiento (Salidas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\circle_train\circle_train_outputs.txt`
    *   Prueba (Entradas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\circle_train\circle_test_inputs.txt`
    *   Prueba (Salidas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\circle_train\circle_test_outputs.txt`

    Contenido de `circle_train_inputs.txt` (y `circle_test_inputs.txt`):
    ```
    0.1,0.1
    -0.2,0.3
    0.4,-0.1
    0.3,0.2
    0.7,0.7
    -0.8,0.6
    0.5,-0.9
    -0.9,-0.8
    0.0,0.4
    0.4,0.0
    0.8,0.0
    0.0,0.8
    ```
    Contenido de `circle_train_outputs.txt` (y `circle_test_outputs.txt`):
    (Clase 0 para dentro, Clase 1 para fuera, asumiendo un radio ~0.5)
    ```
    0
    0
    0
    0
    1
    1
    1
    1
    0
    0
    1
    1
    ```

*   **Configuración de Red en la GUI**:
    *   Número de neuronas de entrada: `2`
    *   Número de neuronas de salida: `1`
    *   Capas ocultas: `8,4` (dos capas ocultas) o una sola capa como `10`.
    *   Activación capas ocultas: `sigmoid` o `relu`
    *   Activación capa de salida: `sigmoid`

*   **Parámetros de Entrenamiento en la GUI**:
    *   Número de épocas: `5000` - `20000`
    *   Tasa de aprendizaje (η): `0.01` - `0.1`
    *   Tamaño del lote (Batch Size): `4` (o el número total de muestras si es pequeño como este ejemplo)

*   **Resultado Esperado**: La red debería aprender una frontera de decisión aproximadamente circular. La precisión dependerá de la cantidad de datos y la complejidad de la red. Con los datos de ejemplo, se espera una buena precisión, aunque no necesariamente 100% perfecto debido a la simplicidad de los datos.

## Estructura de Archivos del Proyecto

*   `main.py`: Punto de entrada de la aplicación. Inicializa la GUI y maneja el bucle principal y el cierre de la aplicación.
*   `perceptron_gui.py`: Contiene la clase `MLPApp` que define toda la interfaz gráfica de usuario y la lógica de interacción.
*   `multilayer_perceptron.py`: Contiene la clase `MultilayerPerceptron` que implementa la lógica de la red neuronal (capas, entrenamiento, predicción, guardado/carga).
*   `layer.py`: Contiene la clase `Layer` que define una capa individual de neuronas, incluyendo sus pesos, biases y funciones de activación.

## Posibles Mejoras Futuras

*   Soporte para diferentes funciones de coste (ej. Error Cuadrático Medio).
*   Más funciones de activación y sus derivadas.
*   Optimización del rendimiento para conjuntos de datos más grandes.
*   Validación cruzada.
*   Regularización (L1, L2, Dropout).
*   Diferentes algoritmos de optimización (ej. Adam, RMSprop).
*   Visualización de los pesos de la red o de las fronteras de decisión (para problemas 2D).
*   Soporte para problemas de regresión de forma más explícita.
```// filepath: README.md
# Perceptrón Multicapa (MLP) con Interfaz Gráfica

Este proyecto implementa un Perceptrón Multicapa (MLP) desde cero en Python, acompañado de una interfaz gráfica de usuario (GUI) construida con Tkinter. Permite a los usuarios crear, entrenar, evaluar, guardar y cargar modelos de MLP para tareas de clasificación.

## Descripción General

El objetivo principal es proporcionar una herramienta educativa y experimental para entender el funcionamiento de las redes neuronales de tipo MLP. La implementación cubre los componentes esenciales:

*   **Capas (Layers)**: Definición de capas neuronales con pesos, biases y funciones de activación.
*   **MLP**: Orquestación de múltiples capas, incluyendo propagación hacia adelante (feedforward) y retropropagación del error (backpropagation) para el entrenamiento.
*   **GUI**: Una interfaz intuitiva para interactuar con el MLP, visualizar el progreso del entrenamiento y realizar predicciones.

## Características Principales

*   **Creación Dinámica de Redes**:
    *   Especificar el número de neuronas de entrada y salida.
    *   Definir múltiples capas ocultas, cada una con un número específico de neuronas.
    *   Seleccionar funciones de activación (Sigmoide, ReLU, Tanh) para las capas ocultas y la capa de salida (incluyendo 'linear' para la salida).
*   **Entrenamiento del Modelo**:
    *   Cargar conjuntos de datos de entrenamiento y prueba desde archivos de texto (`.txt` o `.csv`).
    *   Configurar hiperparámetros: número de épocas, tasa de aprendizaje y tamaño del lote (batch size).
    *   Opción para "Continuar Entrenamiento" sobre un modelo ya entrenado.
    *   El entrenamiento se ejecuta en un hilo separado para mantener la GUI responsiva.
*   **Evaluación y Visualización**:
    *   Cálculo de la precisión en los conjuntos de entrenamiento y prueba después de cada época.
    *   Visualización gráfica de la precisión del entrenamiento y prueba a lo largo de las épocas.
    *   Logs detallados de las operaciones y el progreso del entrenamiento.
*   **Predicción (Feedforward)**:
    *   Ingresar datos manualmente para obtener predicciones.
    *   Cargar un archivo de datos para realizar predicciones en lote.
    *   Opción para guardar los resultados de las predicciones en un archivo.
*   **Persistencia del Modelo**:
    *   Guardar la arquitectura y los pesos del modelo entrenado en formato JSON.
    *   Cargar modelos previamente guardados para continuar el entrenamiento o realizar predicciones.
*   **Manejo de Salida**: Redirección de `stdout` y `stderr` al área de logs de la GUI para una mejor depuración y seguimiento.

## Requisitos

*   Python 3.x
*   Tkinter (generalmente incluido con las instalaciones estándar de Python)
*   NumPy
*   Matplotlib

Puedes instalar las bibliotecas necesarias usando pip:
```bash
pip install numpy matplotlib
```

## Cómo Ejecutar la Aplicación

1.  Clona o descarga este repositorio.
2.  Asegúrate de tener todos los archivos Python (`main.py`, `perceptron_gui.py`, `multilayer_perceptron.py`, `layer.py`) en el mismo directorio.
3.  Ejecuta el script principal desde tu terminal:
    ```bash
    python main.py
    ```
    Esto abrirá la interfaz gráfica del Perceptrón Multicapa.

## Usando la GUI

La interfaz está dividida en secciones:

1.  **Inicialización de Red**:
    *   **Crear Nueva Red**: Abre un diálogo para definir la arquitectura de un nuevo MLP.
    *   **Cargar Red desde Archivo**: Permite cargar un modelo MLP previamente guardado (archivo `.json`).
2.  **Información de la Red**: Muestra el estado actual y la arquitectura de la red cargada/creada.
3.  **Acciones**: Botones para interactuar con la red (se habilitan una vez que hay una red activa).
    *   **Entrenar Red**: Abre el diálogo para configurar y comenzar un nuevo entrenamiento.
    *   **Continuar Entrenamiento**: Abre el diálogo para continuar entrenando la red actual.
    *   **Ejecutar Red (Feedforward)**: Permite ingresar datos manualmente o desde un archivo para obtener predicciones.
    *   **Guardar Red**: Guarda la red actual (arquitectura y pesos) en un archivo `.json`.
    *   **Mostrar Gráfico de Precisión**: Muestra un gráfico del historial de precisión del entrenamiento.
4.  **Salida/Logs**: Muestra mensajes de estado, progreso del entrenamiento y errores.

## Ejemplos de Problemas y Cómo Probarlos

A continuación, se explica cómo probar el MLP con algunos conjuntos de datos clásicos. Asegúrate de tener los archivos de datos en las rutas especificadas o ajusta las rutas en la GUI al cargar los archivos.

### Preparación General de Datos

*   Los archivos de datos de entrada deben ser archivos de texto (`.txt` o `.csv`) donde cada fila representa una muestra y los valores (características) están separados por comas.
*   Los archivos de datos de salida deben seguir el mismo formato, con cada fila correspondiendo a la salida esperada para la muestra de entrada respectiva.

### 1. Problema XOR (OR Exclusivo)

Este es un problema clásico no linealmente separable.

*   **Archivos de Datos**:
    *   Entrenamiento (Entradas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\xor_data\xor_train_inputs.txt`
    *   Entrenamiento (Salidas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\xor_data\xor_train_outputs.txt`
    *   Prueba (Entradas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\xor_data\xor_test_inputs.txt`
    *   Prueba (Salidas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\xor_data\xor_test_outputs.txt`

    Contenido de `xor_train_inputs.txt` (y `xor_test_inputs.txt`):
    ```
    0,0
    0,1
    1,0
    1,1
    ```
    Contenido de `xor_train_outputs.txt` (y `xor_test_outputs.txt`):
    ```
    0
    1
    1
    0
    ```

*   **Configuración de Red en la GUI (al "Crear Nueva Red")**:
    *   Número de neuronas de entrada: `2`
    *   Número de neuronas de salida: `1`
    *   Capas ocultas (neuronas por capa, ej: 5,3): `4` (esto crea una capa oculta con 4 neuronas)
    *   Activación capas ocultas: `sigmoid`
    *   Activación capa de salida: `sigmoid`

*   **Parámetros de Entrenamiento en la GUI**:
    *   Número de épocas: `5000` (o más, ej. `10000`)
    *   Tasa de aprendizaje (η): `0.1` (puedes probar hasta `0.5`)
    *   Tamaño del lote (Batch Size): `1` (o `4`)

*   **Resultado Esperado**: La red debería alcanzar una precisión del 100% en los datos de entrenamiento y prueba.

### 2. Problema XNOR (OR Exclusivo Negado)

La operación lógica inversa a XOR.

*   **Archivos de Datos**:
    *   Entrenamiento (Entradas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\nxor_data\nxor_train_inputs.txt`
    *   Entrenamiento (Salidas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\nxor_data\nxor_train_outputs.txt`
    *   Prueba (Entradas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\nxor_data\nxor_test_inputs.txt`
    *   Prueba (Salidas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\nxor_data\nxor_test_outputs.txt`

    Contenido de `nxor_train_inputs.txt` (y `nxor_test_inputs.txt`):
    ```
    0,0
    0,1
    1,0
    1,1
    ```
    Contenido de `nxor_train_outputs.txt` (y `nxor_test_outputs.txt`):
    ```
    1
    0
    0
    1
    ```

*   **Configuración de Red en la GUI**: Similar a XOR.
    *   Número de neuronas de entrada: `2`
    *   Número de neuronas de salida: `1`
    *   Capas ocultas: `4`
    *   Activación capas ocultas: `sigmoid`
    *   Activación capa de salida: `sigmoid`

*   **Parámetros de Entrenamiento en la GUI**:
    *   Número de épocas: `5000` - `10000`
    *   Tasa de aprendizaje (η): `0.1` - `0.5`
    *   Tamaño del lote (Batch Size): `1` (o `4`)

*   **Resultado Esperado**: Precisión del 100%.

### 3. Problema del Círculo (Clasificación Dentro/Fuera)

Clasificar puntos 2D según si están dentro o fuera de un círculo imaginario.

*   **Archivos de Datos**:
    *   Entrenamiento (Entradas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\circle_train\circle_train_inputs.txt`
    *   Entrenamiento (Salidas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\circle_train\circle_train_outputs.txt`
    *   Prueba (Entradas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\circle_train\circle_test_inputs.txt`
    *   Prueba (Salidas): `c:\Users\chris\Documents\Universidad\Computación Emergente\Tarea1\circle_train\circle_test_outputs.txt`

    Contenido de `circle_train_inputs.txt` (y `circle_test_inputs.txt`):
    ```
    0.1,0.1
    -0.2,0.3
    0.4,-0.1
    0.3,0.2
    0.7,0.7
    -0.8,0.6
    0.5,-0.9
    -0.9,-0.8
    0.0,0.4
    0.4,0.0
    0.8,0.0
    0.0,0.8
    ```
    Contenido de `circle_train_outputs.txt` (y `circle_test_outputs.txt`):
    (Clase 0 para dentro, Clase 1 para fuera, asumiendo un radio ~0.5)
    ```
    0
    0
    0
    0
    1
    1
    1
    1
    0
    0
    1
    1
    ```

*   **Configuración de Red en la GUI**:
    *   Número de neuronas de entrada: `2`
    *   Número de neuronas de salida: `1`
    *   Capas ocultas: `8,4` (dos capas ocultas) o una sola capa como `10`.
    *   Activación capas ocultas: `sigmoid` o `relu`
    *   Activación capa de salida: `sigmoid`

*   **Parámetros de Entrenamiento en la GUI**:
    *   Número de épocas: `5000` - `20000`
    *   Tasa de aprendizaje (η): `0.01` - `0.1`
    *   Tamaño del lote (Batch Size): `4` (o el número total de muestras si es pequeño como este ejemplo)

*   **Resultado Esperado**: La red debería aprender una frontera de decisión aproximadamente circular. La precisión dependerá de la cantidad de datos y la complejidad de la red. Con los datos de ejemplo, se espera una buena precisión, aunque no necesariamente 100% perfecto debido a la simplicidad de los datos.

## Estructura de Archivos del Proyecto

*   `main.py`: Punto de entrada de la aplicación. Inicializa la GUI y maneja el bucle principal y el cierre de la aplicación.
*   `perceptron_gui.py`: Contiene la clase `MLPApp` que define toda la interfaz gráfica de usuario y la lógica de interacción.
*   `multilayer_perceptron.py`: Contiene la clase `MultilayerPerceptron` que implementa la lógica de la red neuronal (capas, entrenamiento, predicción, guardado/carga).
*   `layer.py`: Contiene la clase `Layer` que define una capa individual de neuronas, incluyendo sus pesos, biases y funciones de activación.

## Posibles Mejoras Futuras

*   Soporte para diferentes funciones de coste (ej. Error Cuadrático Medio).
*   Más funciones de activación y sus derivadas.
*   Optimización del rendimiento para conjuntos de datos más grandes.
*   Traducir comentarios al inglés.