import numpy as np

# --- Funciones de Activación (Mantener o expandir las existentes) ---
def sigmoid_activation(z):
    """Función de activación sigmoide."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(output):
    """Derivada de la función sigmoide."""
    return output * (1 - output)

def relu_activation(z):
    """Función de activación ReLU."""
    return np.maximum(0, z)

def relu_derivative(output):
    """Derivada de la función ReLU."""
    # La derivada es 1 si output > 0, y 0 si output <= 0
    # output ya es el resultado de relu_activation(z)
    return np.where(output > 0, 1, 0)

# (Puedes añadir tanh y su derivada también si lo deseas)

class Layer:
    def __init__(self, n_inputs, n_neurons, activation_function_name='sigmoid'):
        """
        Inicializa una capa de la red neuronal.

        Args:
            n_inputs (int): Número de entradas a esta capa (neuronas en la capa anterior).
            n_neurons (int): Número de neuronas en esta capa.
            activation_function_name (str): Nombre de la función de activación a usar ('sigmoid', 'relu').
        """
        # Inicialización de pesos aleatorios con valores pequeños.
        # El factor 0.01 ayuda a evitar que las neuronas se saturen al inicio.
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        # Un bias por neurona en esta capa.
        self.biases = np.zeros((1, n_neurons))
        self.activation_function_name = activation_function_name
        self.activation_function = None
        self.activation_derivative = None

        if activation_function_name == 'sigmoid':
            self.activation_function = sigmoid_activation
            self.activation_derivative = sigmoid_derivative
        elif activation_function_name == 'relu':
            self.activation_function = relu_activation
            self.activation_derivative = relu_derivative
        else:
            # Por defecto o si el nombre no es reconocido, usa sigmoide
            print(f"Advertencia: Función de activación '{activation_function_name}' no reconocida. Usando sigmoide.")
            self.activation_function = sigmoid_activation
            self.activation_derivative = sigmoid_derivative

        # Variables para almacenar valores durante la retropropagación
        self.inputs = None  # Entradas a esta capa
        self.z = None       # Suma ponderada (antes de la activación)
        self.output = None  # Salida de esta capa (después de la activación)
        self.delta = None   # Error delta para esta capa

    def forward(self, inputs_np):
        """
        Calcula la salida de la capa (propagación hacia adelante).

        Args:
            inputs_np (np.array): Array de NumPy con las entradas a la capa.
                                  Debe tener la forma (n_muestras, n_inputs_de_la_capa).

        Returns:
            np.array: La salida de la capa después de la activación.
        """
        if not isinstance(inputs_np, np.ndarray):
            raise TypeError("Las entradas deben ser un array de NumPy.")
        if inputs_np.shape[1] != self.weights.shape[0]:
            raise ValueError(f"El número de características de entrada ({inputs_np.shape[1]}) "
                             f"no coincide con el esperado por los pesos de la capa ({self.weights.shape[0]}).")

        self.inputs = inputs_np
        self.z = np.dot(inputs_np, self.weights) + self.biases
        if self.activation_function:
            self.output = self.activation_function(self.z)
        else: # Para la capa de salida lineal si se decidiera usar (aunque usualmente se usa sigmoide/softmax para clasificación)
            self.output = self.z
        return self.output

    def __repr__(self):
        return f"Layer(inputs={self.weights.shape[0]}, neurons={self.weights.shape[1]}, activation='{self.activation_function_name}')"



