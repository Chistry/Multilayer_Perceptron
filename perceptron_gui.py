import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib import pyplot as plt
import numpy as np
import os
import threading

from layer import Layer, sigmoid_activation, relu_activation
from multilayer_perceptron import MultilayerPerceptron

class MLPApp: 
    def __init__(self, root):
        self.root = root
        root.title("Perceptrón Multicapa")
        self.mlp = None 
        self.history_plot_window = None 

        # --- Marco Principal ---
        self.mainframe = ttk.Frame(root, padding="10")
        self.mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Sección de Creación/Carga de Red ---
        init_frame = ttk.LabelFrame(self.mainframe, text="Inicialización de Red", padding="10")
        init_frame.grid(column=0, row=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(init_frame, text="Crear Nueva Red", command=self.show_creation_dialog).grid(column=0, row=0, padx=5)
        ttk.Button(init_frame, text="Cargar Red desde Archivo", command=self.load_network_from_file).grid(column=1, row=0, padx=5)

        # --- Sección de Información de Red (se actualiza después de crear/cargar) ---
        self.network_info_frame = ttk.LabelFrame(self.mainframe, text="Información de la Red", padding="10")
        self.network_info_frame.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.network_status_label = ttk.Label(self.network_info_frame, text="Ninguna red cargada/creada.")
        self.network_status_label.pack(pady=5)

        # --- Sección de Acciones (se habilita después de tener una red) ---
        self.actions_frame = ttk.LabelFrame(self.mainframe, text="Acciones", padding="10")
        self.actions_frame.grid(column=0, row=2, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.train_button = ttk.Button(self.actions_frame, text="Entrenar Red", command=self.show_train_dialog, state=tk.DISABLED)
        self.train_button.grid(column=0, row=0, padx=5, pady=5)

        self.continue_train_button = ttk.Button(self.actions_frame, text="Continuar Entrenamiento", command=self.show_continue_train_dialog, state=tk.DISABLED)
        self.continue_train_button.grid(column=1, row=0, padx=5, pady=5)

        self.run_button = ttk.Button(self.actions_frame, text="Ejecutar Red (Feedforward)", command=self.show_run_dialog, state=tk.DISABLED)
        self.run_button.grid(column=0, row=1, padx=5, pady=5)

        self.save_button = ttk.Button(self.actions_frame, text="Guardar Red", command=self.save_network_to_file, state=tk.DISABLED)
        self.save_button.grid(column=1, row=1, padx=5, pady=5)

        self.plot_button = ttk.Button(self.actions_frame, text="Mostrar Gráfico de Precisión", command=self.show_accuracy_plot, state=tk.DISABLED)
        self.plot_button.grid(column=0, row=2, columnspan=2, padx=5, pady=5)


        # --- Sección de Resultados ---
        results_frame = ttk.LabelFrame(self.mainframe, text="Salida/Logs", padding="10")
        results_frame.grid(column=0, row=3, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        # ... (Text widget para logs como lo tienes)
        self.results_text = tk.Text(results_frame, height=15, width=80, state=tk.DISABLED)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text['yscrollcommand'] = scrollbar.set
        self.is_training = False


    def _enable_actions(self):
        """Habilita los botones de acción una vez que la red está lista."""
        if self.is_training: # No habilitar si está entrenando
            return
        self.train_button.config(state=tk.NORMAL)
        self.continue_train_button.config(state=tk.NORMAL)
        self.run_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        if self.mlp and self.mlp.history and (self.mlp.history['train_accuracy'] or self.mlp.history['test_accuracy']):
            self.plot_button.config(state=tk.NORMAL)
        else:
            self.plot_button.config(state=tk.DISABLED)

    def _disable_actions_during_training(self):
        """Deshabilita botones específicos durante el entrenamiento."""
        self.train_button.config(state=tk.DISABLED)
        self.continue_train_button.config(state=tk.DISABLED)
        # self.init_frame.winfo_children()[0].config(state=tk.DISABLED) # Crear
        # self.init_frame.winfo_children()[1].config(state=tk.DISABLED) # Cargar
        self.save_button.config(state=tk.DISABLED)
        self.plot_button.config(state=tk.DISABLED)




    def _append_log_on_main_thread(self, text_to_append):
        """Método interno para actualizar el bloqu e de texto en el hilo principal."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, text_to_append + "\n")
        self.results_text.config(state=tk.DISABLED)
        self.results_text.see(tk.END)

    def append_log(self, text):
        # Programa la actualización del bloque de texto para que se ejecute en el hilo principal de Tkinter
        if hasattr(self, 'root') and self.root:
            self.root.after(0, self._append_log_on_main_thread, text)
        else: 
            print(text)

    def _redirect_stdout(self):
        """Redirige la salida estándar (print) al widget de texto de logs."""

        class StdoutRedirector:
            def __init__(self, text_widget_logger):
                self.text_widget_logger = text_widget_logger
                self.buffer = ""

            def write(self, message):
                self.buffer += message
                if "\n" in self.buffer:
                    lines = self.buffer.split("\n")
                    for line in lines[:-1]:
                        if line.strip(): # Solo añade si no está vacía después de quitar espacios
                            self.text_widget_logger(line)
                    self.buffer = lines[-1]

            def flush(self):
                # Si queda algo en el buffer al hacer flush (ej. al final del programa)
                if self.buffer.strip():
                     self.text_widget_logger(self.buffer)
                self.buffer = ""


        sys.stdout = StdoutRedirector(self.append_log)
        sys.stderr = StdoutRedirector(self.append_log) # También redirigir errores


    def show_creation_dialog(self):
        if self.is_training:
            messagebox.showwarning("Entrenamiento en Curso", "No se puede crear una nueva red mientras el entrenamiento está en progreso.", parent=self.root)
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Crear Nueva Red MLP")
        dialog.geometry("450x450") 
        dialog.grab_set() 

        # Variables para los campos de entrada
        n_inputs_var = tk.StringVar(value="2") # Ejemplo
        n_outputs_var = tk.StringVar(value="1") # Ejemplo
        hidden_layers_str_var = tk.StringVar(value="4,3") # Neuronas por capa oculta, ej: "4,3"
        activation_hidden_var = tk.StringVar(value="sigmoid")
        activation_output_var = tk.StringVar(value="sigmoid")

        # --- Layout del diálogo ---
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(expand=True, fill=tk.BOTH)

        ttk.Label(frame, text="Número de neuronas de entrada:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=n_inputs_var, width=25).grid(row=0, column=1, pady=2)

        ttk.Label(frame, text="Número de neuronas de salida:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=n_outputs_var, width=25).grid(row=1, column=1, pady=2)

        ttk.Label(frame, text="Capas ocultas (neuronas por capa, ej: 5,3):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=hidden_layers_str_var, width=25).grid(row=2, column=1, pady=2)

        ttk.Label(frame, text="Activación capas ocultas:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(frame, textvariable=activation_hidden_var, values=['sigmoid', 'relu', 'tanh'], width=22).grid(row=3, column=1, pady=2)

        ttk.Label(frame, text="Activación capa de salida:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(frame, textvariable=activation_output_var, values=['sigmoid', 'relu', 'tanh', 'linear'], width=22).grid(row=4, column=1, pady=2) # 'linear' podría ser no tener activación

        def on_confirm_creation():
            try:
                n_inputs = int(n_inputs_var.get())
                n_outputs = int(n_outputs_var.get())

                hidden_layers_config_str = hidden_layers_str_var.get()
                if hidden_layers_config_str.strip(): # Si no está vacío
                    hidden_neurons_per_layer = [int(n.strip()) for n in hidden_layers_config_str.split(',') if n.strip()]
                else: # No hay capas ocultas
                    hidden_neurons_per_layer = []

                act_hidden = activation_hidden_var.get()
                act_output = activation_output_var.get()

                if n_inputs <= 0 or n_outputs <= 0 or any(n <= 0 for n in hidden_neurons_per_layer):
                    messagebox.showerror("Error de Validación", "El número de neuronas debe ser positivo.", parent=dialog)
                    return

                self.mlp = MultilayerPerceptron()

                # Capa de entrada (implícita) a la primera capa oculta (o capa de salida si no hay ocultas)
                current_n_inputs = n_inputs

                # Añadir capas ocultas
                for n_neurons_hidden in hidden_neurons_per_layer:
                    self.mlp.add_layer(Layer(current_n_inputs, n_neurons_hidden, activation_function_name=act_hidden))
                    current_n_inputs = n_neurons_hidden # La salida de esta capa es la entrada de la siguiente

                # Añadir capa de salida
                self.mlp.add_layer(Layer(current_n_inputs, n_outputs, activation_function_name=act_output))

                self.network_status_label.config(text=f"Nueva red creada. Arquitectura: {self.get_mlp_architecture_string()}")
                self.append_log(f"Red MLP creada con éxito: {self.get_mlp_architecture_string()}")
                self._enable_actions()
                dialog.destroy()

            except ValueError:
                messagebox.showerror("Error de Entrada", "Por favor, ingrese números válidos para las neuronas.", parent=dialog)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo crear la red: {e}", parent=dialog)
                self.mlp = None
                self._disable_actions()
                self.network_status_label.config(text="Error al crear la red.")

        confirm_button = ttk.Button(frame, text="Crear Red", command=on_confirm_creation)
        confirm_button.grid(row=6, column=0, columnspan=2, pady=20)

        cancel_button = ttk.Button(frame, text="Cancelar", command=dialog.destroy)
        cancel_button.grid(row=7, column=0, columnspan=2, pady=5)

    def get_mlp_architecture_string(self):
        if not self.mlp or not self.mlp.layers:
            return "N/A"
        arch_parts = []
        arch_parts.append(f"Entrada({self.mlp.layers[0].weights.shape[0]})")
        for i, layer in enumerate(self.mlp.layers):
            type_str = "Oculta"
            if i == len(self.mlp.layers) -1:
                type_str = "Salida"
            arch_parts.append(f"Capa {type_str}({layer.weights.shape[1]} neuronas, act: {layer.activation_function_name})")
        return " -> ".join(arch_parts)

    def load_network_from_file(self):
        if self.is_training:
            messagebox.showwarning("Entrenamiento en Curso", "No se puede cargar una red mientras el entrenamiento está en progreso.", parent=self.root)
            return
        file_path = filedialog.askopenfilename(
            title="Cargar Modelo MLP",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            defaultextension=".json"
        )
        if not file_path:
            return

        loaded_mlp = MultilayerPerceptron.load_model(file_path)
        if loaded_mlp:
            self.mlp = loaded_mlp
            self.network_status_label.config(text=f"Red cargada desde '{os.path.basename(file_path)}'. Arquitectura: {self.get_mlp_architecture_string()}")
            self.append_log(f"Red MLP cargada con éxito desde '{file_path}'.")
            self._enable_actions()
            # Si el modelo cargado tiene historial, podríamos ofrecer mostrar el gráfico
            if self.mlp.history and (self.mlp.history['train_accuracy'] or self.mlp.history['test_accuracy']):
                self.plot_button.config(state=tk.NORMAL)
        else:
            messagebox.showerror("Error de Carga", f"No se pudo cargar el modelo desde '{file_path}'. Ver logs para detalles.")
            self.network_status_label.config(text="Error al cargar la red.")
            self._disable_actions()

    def _training_thread_target(self, X_train, y_train, X_test, y_test, epochs, lr, batch_s, continue_training_flag):
        """Función que se ejecuta en un hilo separado para el entrenamiento."""
        try:
            self.mlp.train(X_train, y_train, X_test, y_test, epochs, lr, batch_s)
            # Una vez completado, programa la actualización de la GUI en el hilo principal
            self.root.after(0, self._on_training_complete, True, None, continue_training_flag)
        except Exception as e:
            # Si hay un error, también actualiza la GUI desde el hilo principal
            self.root.after(0, self._on_training_complete, False, e, continue_training_flag)

    def _on_training_complete(self, success, error, continue_training_flag):
        """Se llama desde el hilo principal después de que el hilo de entrenamiento termina."""
        self.is_training = False # Restablecer el flag
        if success:
            self.append_log("Entrenamiento finalizado con éxito.")
            if self.mlp.history and self.mlp.history['test_accuracy'] and self.mlp.history['test_accuracy'][-1] is not None:
                 last_accuracy = self.mlp.history['test_accuracy'][-1]
                 self.network_status_label.config(text=f"Red entrenada. Última precisión prueba: {last_accuracy:.2f}%")
            else:
                 self.network_status_label.config(text="Red entrenada. No hay historial de precisión disponible.")
        else:
            self.append_log(f"Error durante el entrenamiento: {error}")
            messagebox.showerror("Error de Entrenamiento", f"Ocurrió un error: {error}", parent=self.root)
            self.network_status_label.config(text="Error durante el entrenamiento.")

        self._enable_actions() 

    # --- Funciones para leer datos de archivos ---
    def _load_data_from_file(self, file_path, is_output=False, num_outputs_expected=1):
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("File Error", f"The file '{file_path or 'specified'}' does not exist.")
            return None
        try:
            data = []
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = [float(x.strip()) for x in line.split(',')]
                        data.append(row)
                    except ValueError:
                        messagebox.showerror("Format Error", f"Format error at line {line_num} in file '{os.path.basename(file_path)}'. Expected numbers separated by commas.")
                        return None
            data_np = np.array(data)
            if is_output:
                data_np = np.round(data_np).astype(int)
                if data_np.ndim == 1:
                    data_np = data_np.reshape(-1, 1)
                if data_np.shape[1] != num_outputs_expected:
                    messagebox.showerror("Output Format Error", f"Output file '{os.path.basename(file_path)}' has {data_np.shape[1]} columns, but {num_outputs_expected} were expected.")
                    return None
            print(f"Loaded data from {file_path}: shape={data_np.shape}")
            return data_np
        except Exception as e:
            messagebox.showerror("Read Error", f"Could not read file '{os.path.basename(file_path)}': {e}")
            return None

    def show_train_dialog(self, continue_training=False):
        """
        Use the same MLP instance and confirm correct epochs/data.
        """
        if not self.mlp:
            messagebox.showerror("Sin Red", "Primero crea o carga una red.", parent=self.root)
            return
        if self.is_training:
            messagebox.showwarning("Entrenamiento en Curso", "El entrenamiento ya está en progreso.", parent=self.root)
            return

   
        dialog = tk.Toplevel(self.root)
        dialog.title("Entrenar Red MLP" if not continue_training else "Continuar Entrenamiento")
        dialog.geometry("550x450")
        dialog.grab_set()

        train_file_var = tk.StringVar(self.root, value=getattr(self, '_last_train_file', ''))
        train_out_file_var = tk.StringVar(self.root, value=getattr(self, '_last_train_out_file', ''))
        test_file_var = tk.StringVar(self.root, value=getattr(self, '_last_test_file', ''))
        test_out_file_var = tk.StringVar(self.root, value=getattr(self, '_last_test_out_file', ''))
        epochs_var = tk.StringVar(self.root, value=getattr(self, '_last_epochs', "100"))
        lr_var = tk.StringVar(self.root, value=getattr(self, '_last_lr', "0.01"))
        batch_size_var = tk.StringVar(self.root, value=getattr(self, '_last_batch_size', "32"))

        frame = ttk.Frame(dialog, padding="10")
        frame.pack(expand=True, fill=tk.BOTH)

        row_idx = 0

        # --- Archivos de Datos ---
        ttk.Label(frame, text="Archivo de entrenamiento (.txt, CSV):").grid(row=row_idx, column=0, sticky=tk.W, pady=3)
        ttk.Entry(frame, textvariable=train_file_var, width=40).grid(row=row_idx, column=1, pady=3)
        ttk.Button(frame, text="Buscar...", command=lambda: train_file_var.set(filedialog.askopenfilename(title="Archivo de Entrenamiento", filetypes=(("Text files", "*.txt"),("CSV files", "*.csv"),("All files", "*.*")) or train_file_var.get()))).grid(row=row_idx, column=2, padx=5, pady=3)
        row_idx += 1

        ttk.Label(frame, text="Archivo de salidas de entrenamiento (.txt, CSV):").grid(row=row_idx, column=0, sticky=tk.W, pady=3)
        ttk.Entry(frame, textvariable=train_out_file_var, width=40).grid(row=row_idx, column=1, pady=3)
        ttk.Button(frame, text="Buscar...", command=lambda: train_out_file_var.set(filedialog.askopenfilename(title="Salidas de Entrenamiento", filetypes=(("Text files", "*.txt"),("CSV files", "*.csv"),("All files", "*.*")) or train_out_file_var.get()))).grid(row=row_idx, column=2, padx=5, pady=3)
        row_idx += 1

        ttk.Label(frame, text="Archivo de datos de prueba (.txt, CSV):").grid(row=row_idx, column=0, sticky=tk.W, pady=3)
        ttk.Entry(frame, textvariable=test_file_var, width=40).grid(row=row_idx, column=1, pady=3)
        ttk.Button(frame, text="Buscar...", command=lambda: test_file_var.set(filedialog.askopenfilename(title="Archivo de Prueba", filetypes=(("Text files", "*.txt"),("CSV files", "*.csv"),("All files", "*.*")) or test_file_var.get()))).grid(row=row_idx, column=2, padx=5, pady=3)
        row_idx += 1

        ttk.Label(frame, text="Archivo de salidas de prueba (.txt, CSV):").grid(row=row_idx, column=0, sticky=tk.W, pady=3)
        ttk.Entry(frame, textvariable=test_out_file_var, width=40).grid(row=row_idx, column=1, pady=3)
        ttk.Button(frame, text="Buscar...", command=lambda: test_out_file_var.set(filedialog.askopenfilename(title="Salidas de Prueba", filetypes=(("Text files", "*.txt"),("CSV files", "*.csv"),("All files", "*.*")) or test_out_file_var.get()))).grid(row=row_idx, column=2, padx=5, pady=3)
        row_idx += 1

        ttk.Separator(frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=3, sticky='ew', pady=10)
        row_idx += 1

        # --- Parámetros de Entrenamiento ---
        ttk.Label(frame, text="Número de épocas:").grid(row=row_idx, column=0, sticky=tk.W, pady=3)
        ttk.Entry(frame, textvariable=epochs_var, width=10).grid(row=row_idx, column=1, sticky=tk.W, pady=3)
        row_idx += 1

        ttk.Label(frame, text="Tasa de aprendizaje (η):").grid(row=row_idx, column=0, sticky=tk.W, pady=3)
        ttk.Entry(frame, textvariable=lr_var, width=10).grid(row=row_idx, column=1, sticky=tk.W, pady=3)
        row_idx += 1

        ttk.Label(frame, text="Tamaño del lote (Batch Size):").grid(row=row_idx, column=0, sticky=tk.W, pady=3)
        ttk.Entry(frame, textvariable=batch_size_var, width=10).grid(row=row_idx, column=1, sticky=tk.W, pady=3)
        row_idx += 1


        def on_start_training():
            try:
                train_file = train_file_var.get()
                train_out_file = train_out_file_var.get()
                test_file = test_file_var.get()
                test_out_file = test_out_file_var.get()
                epochs = int(epochs_var.get())
                lr = float(lr_var.get())
                batch_s = int(batch_size_var.get())

                if not all([train_file, train_out_file, test_file, test_out_file]):
                    messagebox.showerror("Error de Archivos", "Todos los archivos de datos son requeridos.", parent=dialog)
                    return
                if epochs <= 0 or lr <= 0 or batch_s <=0:
                    messagebox.showerror("Error de Parámetros", "Épocas, tasa de aprendizaje y tamaño de lote deben ser positivos.", parent=dialog)
                    return

                # Guardar estos paths para la próxima vez
                self._last_train_file = train_file
                self._last_train_out_file = train_out_file
                self._last_test_file = test_file
                self._last_test_out_file = test_out_file
                self._last_epochs = epochs_var.get()
                self._last_lr = lr_var.get()
                self._last_batch_size = batch_size_var.get()

                num_outputs = self.mlp.layers[-1].weights.shape[1]

                X_train = self._load_data_from_file(train_file)
                y_train = self._load_data_from_file(train_out_file, is_output=True, num_outputs_expected=num_outputs)
                X_test = self._load_data_from_file(test_file)
                y_test = self._load_data_from_file(test_out_file, is_output=True, num_outputs_expected=num_outputs)

                if X_train is None or y_train is None or X_test is None or y_test is None:
                    return

                # Validar dimensiones
                if X_train.shape[0] != y_train.shape[0]:
                    messagebox.showerror("Error de Dimensiones", f"El número de muestras en '{os.path.basename(train_file)}' ({X_train.shape[0]}) no coincide con '{os.path.basename(train_out_file)}' ({y_train.shape[0]}).", parent=dialog)
                    return
                if X_test.shape[0] != y_test.shape[0]:
                    messagebox.showerror("Error de Dimensiones", f"El número de muestras en '{os.path.basename(test_file)}' ({X_test.shape[0]}) no coincide con '{os.path.basename(test_out_file)}' ({y_test.shape[0]}).", parent=dialog)
                    return

                num_inputs_expected = self.mlp.layers[0].weights.shape[0]
                if X_train.shape[1] != num_inputs_expected:
                    messagebox.showerror("Error de Dimensiones", f"Los datos de entrenamiento tienen {X_train.shape[1]} características, pero la red espera {num_inputs_expected}.", parent=dialog)
                    return
                if X_test.shape[1] != num_inputs_expected:
                    messagebox.showerror("Error de Dimensiones", f"Los datos de prueba tienen {X_test.shape[1]} características, pero la red espera {num_inputs_expected}.", parent=dialog)
                    return

                if y_train.shape[1] != num_outputs:
                    messagebox.showerror("Error de Dimensiones", f"Las salidas de entrenamiento tienen {y_train.shape[1]} columnas, pero la red tiene {num_outputs} neuronas de salida.", parent=dialog)
                    return
                if y_test.shape[1] != num_outputs:
                     messagebox.showerror("Error de Dimensiones", f"Las salidas de prueba tienen {y_test.shape[1]} columnas, pero la red tiene {num_outputs} neuronas de salida.", parent=dialog)
                     return

                self.is_training = True # Establecer el flag
                self._disable_actions_during_training() 

                self.append_log(f"{'Continuando' if continue_training else 'Iniciando'} entrenamiento...")
                dialog.destroy() 
                
                # Crear y iniciar el hilo de entrenamiento
                training_args = (X_train, y_train, X_test, y_test, epochs, lr, batch_s, continue_training)
                thread = threading.Thread(target=self._training_thread_target, args=training_args)
                thread.daemon = True  # Permite que el programa principal salga aunque el hilo esté corriendo
                thread.start()
                

            except ValueError as ve:
                messagebox.showerror("Error de Entrada", f"Valor inválido: {ve}", parent=dialog)
                self.is_training = False # Restablecer en caso de error antes de iniciar el hilo
                self._enable_actions()
            except Exception as e:
                messagebox.showerror("Error de Preparación", f"Ocurrió un error al preparar el entrenamiento: {e}", parent=dialog)
                self.append_log(f"Error durante la preparación del entrenamiento: {e}")
                self.is_training = False # Restablecer en caso de error
                self._enable_actions()


        ttk.Button(frame, text="Iniciar Entrenamiento", command=on_start_training).grid(row=row_idx, column=0, columnspan=3, pady=15)
        row_idx += 1
        ttk.Button(frame, text="Cancelar", command=dialog.destroy).grid(row=row_idx, column=0, columnspan=3, pady=5)


    def show_continue_train_dialog(self):
        self.show_train_dialog(continue_training=True)

    def show_run_dialog(self):
        if self.is_training:
            messagebox.showwarning("Entrenamiento en Curso", "Espere a que termine el entrenamiento actual para ejecutar la red.", parent=self.root)
            return
        if not self.mlp:
            messagebox.showwarning("Advertencia", "Primero debe crear o cargar una red.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Ejecutar Red MLP (Feedforward)")
        dialog.geometry("450x300")
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding="10")
        frame.pack(expand=True, fill=tk.BOTH)

        # --- Entrada por Teclado ---
        ttk.Label(frame, text=f"Introducir vector de entrada ({self.mlp.layers[0].weights.shape[0]} valores, separados por comas):").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        input_vector_var = tk.StringVar()
        input_entry = ttk.Entry(frame, textvariable=input_vector_var, width=50)
        input_entry.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=5)

        def on_run_keyboard():
            try:
                input_str = input_vector_var.get().strip()
                if not input_str:
                    messagebox.showinfo("Resultado", "Entrada vacía.", parent=dialog)
                    return

                input_values = [float(x.strip()) for x in input_str.split(',')]
                input_np = np.array(input_values).reshape(1, -1) 

                if input_np.shape[1] != self.mlp.layers[0].weights.shape[0]:
                     messagebox.showerror("Error de Entrada", f"Se esperaban {self.mlp.layers[0].weights.shape[0]} valores de entrada, pero se proporcionaron {input_np.shape[1]}.", parent=dialog)
                     return

                prediction_raw = self.mlp._forward_propagation(input_np) 
                prediction_class = self.mlp.predict(input_np) 

                self.append_log(f"Entrada (teclado): {input_values}")
                self.append_log(f"  -> Salida cruda de la red: {prediction_raw.flatten().tolist()}")
                self.append_log(f"  -> Clase/Valor Predicho: {prediction_class.flatten().tolist()}")
                messagebox.showinfo("Resultado de Predicción", f"Entrada: {input_values}\nSalida Cruda: {prediction_raw.flatten()}\nClase/Valor Predicho: {prediction_class.flatten()}", parent=dialog)

            except ValueError:
                messagebox.showerror("Error de Entrada", "Por favor, ingrese números válidos separados por comas.", parent=dialog)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo procesar la entrada: {e}", parent=dialog)

        ttk.Button(frame, text="Ejecutar con entrada de teclado", command=on_run_keyboard).grid(row=2, column=0, columnspan=2, pady=10)

        # --- Entrada por Archivo ---
        ttk.Separator(frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky='ew', pady=10)
        ttk.Label(frame, text="Cargar archivo de prueba (.txt, CSV):").grid(row=4, column=0, sticky=tk.W, pady=5)

        file_path_var = tk.StringVar()
        def browse_run_file():
            path = filedialog.askopenfilename(title="Seleccionar Archivo de Prueba para Ejecutar", filetypes=(("Text files", "*.txt"),("CSV files", "*.csv"),("All files", "*.*")))
            if path:
                file_path_var.set(path)
                file_label.config(text=os.path.basename(path))

        file_label = ttk.Label(frame, text="Ningún archivo seleccionado.")
        file_label.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Button(frame, text="Buscar Archivo...", command=browse_run_file).grid(row=4, column=1, sticky=tk.E, pady=5)

        def on_run_file():
            file_path = file_path_var.get()
            if not file_path:
                messagebox.showerror("Error de Archivo", "Por favor, seleccione un archivo de prueba.", parent=dialog)
                return

            X_test = self._load_data_from_file(file_path)
            if X_test is None: return

            if X_test.shape[1] != self.mlp.layers[0].weights.shape[0]:
                messagebox.showerror("Error de Dimensiones", f"Los datos del archivo tienen {X_test.shape[1]} características, pero la red espera {self.mlp.layers[0].weights.shape[0]}.", parent=dialog)
                return

            self.append_log(f"Ejecutando red con archivo: {os.path.basename(file_path)}")
            predictions_raw = self.mlp._forward_propagation(X_test)
            predictions_class = self.mlp.predict(X_test)

            results_str = f"Resultados para '{os.path.basename(file_path)}':\n"
            results_str += "Entrada -> Salida Cruda -> Clase/Valor Predicho\n"
            results_str += "--------------------------------------------\n"
            for i in range(X_test.shape[0]):
                input_row_str = ", ".join(map(lambda x: f"{x:.2f}", X_test[i]))
                raw_out_str = ", ".join(map(lambda x: f"{x:.4f}", predictions_raw[i]))
                class_out_str = ", ".join(map(str, predictions_class[i])) 

                log_line = f"{input_row_str} -> RAW: [{raw_out_str}] -> PRED: [{class_out_str}]"
                results_str += log_line + "\n"
                self.append_log(log_line)

            # Mostrar en un message box o en el log principal
            self.append_log("--- Fin de ejecución con archivo ---")

            # Podrías querer guardar los resultados en un archivo también
            save_results_path = filedialog.asksaveasfilename(
                title="Guardar Resultados de Predicción",
                defaultextension=".txt",
                filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
            )
            if save_results_path:
                try:
                    with open(save_results_path, 'w') as f_out:
                        f_out.write(results_str)
                    self.append_log(f"Resultados guardados en: {save_results_path}")
                    messagebox.showinfo("Resultados Guardados", f"Las predicciones han sido guardadas en '{os.path.basename(save_results_path)}'.", parent=dialog)
                except Exception as e_save:
                    self.append_log(f"Error al guardar resultados: {e_save}")
                    messagebox.showerror("Error al Guardar", f"No se pudieron guardar los resultados: {e_save}", parent=dialog)
            else:
                messagebox.showinfo("Predicciones", "Predicciones generadas y mostradas en el log. No se seleccionó archivo para guardar.", parent=dialog)


        ttk.Button(frame, text="Ejecutar con archivo", command=on_run_file).grid(row=6, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Cerrar", command=dialog.destroy).grid(row=7, column=0, columnspan=2, pady=5)


    def save_network_to_file(self):
        if self.is_training:
            messagebox.showwarning("Entrenamiento en Curso", "Espere a que termine el entrenamiento actual para guardar la red.", parent=self.root)
            return
        if not self.mlp:
            messagebox.showwarning("Advertencia", "No hay red para guardar.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Guardar Modelo MLP",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if file_path:
            self.mlp.save_model(file_path)
            self.append_log(f"Red guardada en '{file_path}'.")
            messagebox.showinfo("Guardado", f"Red MLP guardada en '{os.path.basename(file_path)}'.")


    def show_accuracy_plot(self):
        if self.is_training:
            messagebox.showwarning("Entrenamiento en Curso", "Espere a que termine el entrenamiento para ver el gráfico.", parent=self.root)
            return
        if not self.mlp or not self.mlp.history or \
           (not self.mlp.history['train_accuracy'] and not self.mlp.history['test_accuracy']):
            messagebox.showinfo("Gráfico de Precisión", "No hay datos de historial de precisión para mostrar.", parent=self.root)
            return

        # Cerrar ventana anterior si existe para evitar múltiples gráficos
        if self.history_plot_window and self.history_plot_window.winfo_exists():
            try:
                self.history_plot_window.destroy()
            except tk.TclError: 
                pass


        # Crear una nueva Toplevel para el gráfico de Matplotlib
        self.history_plot_window = tk.Toplevel(self.root)
        self.history_plot_window.title("Gráfico de Precisión por Época")
        self.history_plot_window.geometry("700x500")

        fig, ax = plt.subplots(figsize=(6.5, 4.5)) # Ajusta el tamaño para que quepa bien

        if self.mlp.history['train_accuracy']:
            ax.plot(self.mlp.history['train_accuracy'], label='Precisión Entrenamiento', marker='o')
        if self.mlp.history['test_accuracy']:
            ax.plot(self.mlp.history['test_accuracy'], label='Precisión Prueba', marker='x')

        ax.set_title('Precisión del MLP por Época')
        ax.set_xlabel('Época')
        ax.set_ylabel('Precisión (%)')
        if self.mlp.history['train_accuracy'] or self.mlp.history['test_accuracy']:
            ax.legend()
        ax.grid(True)

        # Integrar Matplotlib con Tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=self.history_plot_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

        # Añadir una barra de herramientas si se desea (opcional)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, self.history_plot_window)
        toolbar.update()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

