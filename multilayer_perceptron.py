import json # Para guardar/cargar la estructura y pesos
import matplotlib.pyplot as plt
import numpy as np

from layer import Layer

class MultilayerPerceptron:
    def __init__(self):
        self.layers = []
        self.history = {'train_accuracy': [], 'test_accuracy': []} # Para graficar

    def add_layer(self, layer):
        """Añade una capa a la red."""
        if not isinstance(layer, Layer):
            raise TypeError("Solo se pueden añadir objetos de la clase Layer.")
        self.layers.append(layer)

    def _forward_propagation(self, X_batch):
        """
        Realiza la propagación hacia adelante a través de todas las capas.

        Args:
            X_batch (np.array): Un lote de datos de entrada.
                               Forma: (n_muestras, n_caracteristicas_entrada)

        Returns:
            np.array: La salida de la última capa.
        """
        current_output = X_batch
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def predict(self, X):
        """
        Realiza predicciones para un conjunto de datos X.
        Convierte las salidas de la red a la clase predicha (0 o 1 basado en 0.5 para sigmoide).
        """
        # Asegurarse de que X sea al menos 2D para consistencia (batch de 1 o más)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        output_proba = self._forward_propagation(X)
        # Para clasificación binaria con salida sigmoide, umbral en 0.5
        # Para múltiples salidas (clasificación multiclase con softmax), se usaría argmax.
        # Asumimos una salida sigmoide por ahora para la precisión.
        # Si tienes múltiples neuronas de salida, necesitas definir cómo se interpreta la "clase".
        if output_proba.shape[1] == 1: # Salida única (clasificación binaria)
            return (output_proba >= 0.5).astype(int)
        else: # Múltiples salidas (ej. one-hot encoding) - tomar el índice del máximo
            return np.argmax(output_proba, axis=1).reshape(-1, 1)


    def _backward_propagation(self, y_batch, learning_rate):
        """
        Realiza la retropropagación del error y actualiza los pesos y biases.

        Args:
            y_batch (np.array): Las salidas esperadas para el lote de entrada.
                               Forma: (n_muestras, n_caracteristicas_salida)
            learning_rate (float): Tasa de aprendizaje.
        """
        # Asegurar que y_batch sea 2D
        if y_batch.ndim == 1:
            y_batch = y_batch.reshape(-1, 1)

        # Empezar desde la última capa (capa de salida)
        output_layer = self.layers[-1]

        # Error en la capa de salida
        # Para la función de coste de entropía cruzada con sigmoide: error = output - y_true
        # Para error cuadrático medio: error = y_true - output_layer.output
        # Vamos a usar error = output_layer.output - y_batch (consistente con derivada de entropía cruzada)
        error_output_layer = output_layer.output - y_batch

        # Delta para la capa de salida
        # delta = error * derivada_activacion(salida_capa)
        output_layer.delta = error_output_layer * output_layer.activation_derivative(output_layer.output)

        # Propagar el error hacia atrás
        # Empezamos desde la penúltima capa y vamos hacia la primera capa oculta
        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i+1]

            # Error de la capa oculta propagado desde la capa siguiente
            error_hidden_layer = np.dot(next_layer.delta, next_layer.weights.T)
            current_layer.delta = error_hidden_layer * current_layer.activation_derivative(current_layer.output)

        # Actualizar pesos y biases para todas las capas
        # Empezamos desde la primera capa hasta la última
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # La entrada a la capa i es la salida de la capa i-1, o X_batch si es la primera capa.
            # Ya almacenamos layer.inputs en la pasada forward.

            # El gradiente de los pesos es la transpuesta de las entradas de la capa multiplicada por el delta de la capa.
            # Se promedia sobre el batch (dividiendo por el número de muestras)
            grad_weights = np.dot(layer.inputs.T, layer.delta) / layer.inputs.shape[0]
            grad_biases = np.sum(layer.delta, axis=0, keepdims=True) / layer.inputs.shape[0]

            layer.weights -= learning_rate * grad_weights
            layer.biases -= learning_rate * grad_biases

    def calculate_accuracy(self, X, y):
        """Calcula la precisión de las predicciones."""
        if X.shape[0] == 0:
            return 0.0
        predictions = self.predict(X)
        if y.ndim == 1:
            y = y.reshape(-1,1)

        # Para manejar el caso de múltiples salidas (one-hot), se necesitaría comparar np.argmax(y, axis=1)
        # Asumimos y como etiquetas de clase directas (0 o 1) si la salida es una neurona,
        # o etiquetas de clase (0, 1, ...) si la salida es multiclase (y las predicciones son índices).
        if predictions.shape[1] == 1 and y.shape[1] == 1: # Clasificación binaria o regresión con una salida
            return np.mean(predictions == y) * 100
        elif predictions.shape[1] > 1 and y.shape[1] > 1: # Multiclase one-hot vs one-hot
            return np.mean(np.all(predictions == y, axis=1)) * 100
        elif y.shape[1] > 1 and predictions.ndim == 1: # y es one-hot, predictions son índices de clase
             y_labels = np.argmax(y, axis=1)
             return np.mean(predictions.flatten() == y_labels) * 100
        elif predictions.shape[1] == 1 and y.shape[1] > 1 : # predictions es clase, y es one-hot (compara predictions con argmax de y)
             y_labels = np.argmax(y, axis=1)
             return np.mean(predictions.flatten() == y_labels) * 100
        else: # Caso por defecto, o si las formas no coinciden bien
             print(f"Advertencia: Formas de predicción ({predictions.shape}) y etiquetas ({y.shape}) no manejadas directamente para precisión.")
             # Intenta una comparación simple, podría no ser correcta para todos los casos.
             return np.mean(predictions == y) * 100


    def train(self, X_train, y_train, X_test, y_test, epochs, learning_rate=0.01, batch_size=32):
        """
        Entrena la red neuronal.

        Args:
            X_train (np.array): Datos de entrenamiento.
            y_train (np.array): Salidas esperadas para el entrenamiento.
            X_test (np.array): Datos de prueba.
            y_test (np.array): Salidas esperadas para los datos de prueba.
            epochs (int): Número de épocas de entrenamiento.
            learning_rate (float): Tasa de aprendizaje.
            batch_size (int): Tamaño del lote para el entrenamiento.
        """
        n_samples = X_train.shape[0]
        self.history = {'train_accuracy': [], 'test_accuracy': []} # Reiniciar historial

        print(f"Iniciando entrenamiento por {epochs} épocas con learning rate {learning_rate} y batch size {batch_size}...")

        for epoch in range(epochs):
            # Mezclar datos de entrenamiento
            permutation = np.random.permutation(n_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                # Asegurarse de que y_batch tenga la forma correcta para el cálculo de error
                if y_batch.ndim == 1:
                    y_batch = y_batch.reshape(-1, self.layers[-1].weights.shape[1]) # Asume que la última capa tiene n_outputs neuronas

                self._forward_propagation(X_batch)
                self._backward_propagation(y_batch, learning_rate)

            # Calcular precisión al final de cada época
            train_accuracy = self.calculate_accuracy(X_train, y_train)
            test_accuracy = self.calculate_accuracy(X_test, y_test)

            self.history['train_accuracy'].append(train_accuracy)
            self.history['test_accuracy'].append(test_accuracy)

            print(f"Época {epoch+1}/{epochs} - Precisión Entrenamiento: {train_accuracy:.2f}% - Precisión Prueba: {test_accuracy:.2f}%")

        print("Entrenamiento completado.")
        #self.plot_accuracy()


    def plot_accuracy(self):
        """Genera un gráfico de la precisión por época."""
        if not self.history['train_accuracy'] or not self.history['test_accuracy']:
            print("No hay historial de precisión para graficar.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_accuracy'], label='Precisión Entrenamiento', marker='o')
        plt.plot(self.history['test_accuracy'], label='Precisión Prueba', marker='x')
        plt.title('Precisión del Perceptrón Multicapa por Época')
        plt.xlabel('Época')
        plt.ylabel('Precisión (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, file_path):
        """Guarda la arquitectura y los pesos del modelo en un archivo JSON."""
        model_data = {
            'architecture': [],
            'weights': [],
            'biases': [],
            'history': self.history
        }
        for layer in self.layers:
            model_data['architecture'].append({
                'n_inputs': layer.weights.shape[0],
                'n_neurons': layer.weights.shape[1],
                'activation': layer.activation_function_name
            })
            model_data['weights'].append(layer.weights.tolist())
            model_data['biases'].append(layer.biases.tolist())

        try:
            with open(file_path, 'w') as f:
                json.dump(model_data, f, indent=4)
            print(f"Modelo guardado en {file_path}")
        except IOError as e:
            print(f"Error al guardar el modelo: {e}")

    @classmethod
    def load_model(cls, file_path):
        """Carga la arquitectura y los pesos del modelo desde un archivo JSON."""
        try:
            with open(file_path, 'r') as f:
                model_data = json.load(f)

            mlp = cls()
            for i, layer_arch in enumerate(model_data['architecture']):
                layer = Layer(layer_arch['n_inputs'], layer_arch['n_neurons'], layer_arch['activation'])
                layer.weights = np.array(model_data['weights'][i])
                layer.biases = np.array(model_data['biases'][i])
                mlp.add_layer(layer)

            if 'history' in model_data:
                mlp.history = model_data['history']
            else:
                 mlp.history = {'train_accuracy': [], 'test_accuracy': []}


            print(f"Modelo cargado desde {file_path}")
            return mlp
        except FileNotFoundError:
            print(f"Error: Archivo de modelo '{file_path}' no encontrado.")
            return None
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return None