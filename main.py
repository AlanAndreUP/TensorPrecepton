import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tkinter as tk
from tkinter import filedialog, ttk
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Clase para almacenar el historial de entrenamiento
class TrainingHistory:
    def __init__(self):
        self.weights = []
        self.biases = []
        self.errors = []
        self.epochs = []
        
    def add_record(self, epoch, weights, bias, error):
        self.epochs.append(epoch)
        self.weights.append(weights.flatten())
        self.biases.append(bias)
        self.errors.append(error)
        
    def get_dataframe(self):
        records = []
        for e, w, b, err in zip(self.epochs, self.weights, self.biases, self.errors):
            record = {'Época': e + 1}
            for i, wi in enumerate(w):
                record[f'w{i}'] = wi
            record['bias'] = b[0]
            record['Error'] = err
            records.append(record)
        return pd.DataFrame(records)

# Lectura de dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path, sep=";") 
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    return df, X, y, scaler_X, scaler_y

# Arquitectura de la Red Neuronal
class LinearRegression(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(1, kernel_initializer=tf.initializers.RandomNormal(stddev=0.01))

    def call(self, inputs):
        return self.dense(inputs)

# Entrenamiento
def train_model(X, y, lr=0.01, epochs=50, batch_size=32):
    model = LinearRegression()
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.SGD(learning_rate=lr)
    history = TrainingHistory()

    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
    
    # Para almacenar predicciones iniciales
    initial_predictions = model(X).numpy().flatten()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataset:
            with tf.GradientTape() as tape:
                predictions = model(batch_X)
                loss = loss_fn(batch_y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss = loss.numpy()
            
        # Guardar historial
        weights, bias = model.dense.get_weights()
        history.add_record(epoch, weights, bias, epoch_loss)
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss}")
    
    # Predicciones finales
    final_predictions = model(X).numpy().flatten()
    
    return model, history, initial_predictions, final_predictions, y

class NeuralNetworkGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Red Neuronal Artificial - Regresión Lineal")
        self.setup_gui()
        
    def setup_gui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Frame izquierdo para controles
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame derecho para visualizaciones
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Controles
        self.select_btn = ttk.Button(left_frame, text="Seleccionar CSV", command=self.select_file)
        self.select_btn.pack(pady=5)
        
        # Tabla de datos
        self.tree = ttk.Treeview(left_frame, show="headings", height=5)
        self.tree.pack(pady=5, fill=tk.X)
        
        # Parámetros de entrenamiento
        params_frame = ttk.LabelFrame(left_frame, text="Parámetros de Entrenamiento")
        params_frame.pack(pady=5, fill=tk.X)
        
        ttk.Label(params_frame, text="Épocas:").grid(row=0, column=0, padx=5, pady=5)
        self.epochs_var = tk.StringVar(value="50")
        self.epochs_entry = ttk.Entry(params_frame, textvariable=self.epochs_var, width=10)
        self.epochs_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=5)
        self.lr_var = tk.StringVar(value="0.01")
        self.lr_entry = ttk.Entry(params_frame, textvariable=self.lr_var, width=10)
        self.lr_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Botón de entrenamiento
        self.train_btn = ttk.Button(left_frame, text="Entrenar Modelo", command=self.train_and_show_results, state=tk.DISABLED)
        self.train_btn.pack(pady=5)
        
        # Tabla de pesos
        self.weights_tree = ttk.Treeview(right_frame, show="headings", height=10)
        self.weights_tree.pack(pady=5, fill=tk.X)
        
        # Frame para la gráfica
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(pady=5, fill=tk.BOTH, expand=True)
        
    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.df, self.X, self.y, self.scaler_X, self.scaler_y = load_data(file_path)
            
            # Configurar tabla de datos
            self.tree["columns"] = list(self.df.columns)
            for col in self.df.columns:
                self.tree.heading(col, text=col)
            
            for i in self.tree.get_children():
                self.tree.delete(i)
            for i, row in self.df.head().iterrows():
                self.tree.insert("", "end", values=list(row))
            
            self.train_btn.config(state=tk.NORMAL)
            
    def train_and_show_results(self):
        epochs = int(self.epochs_var.get())
        lr = float(self.lr_var.get())
        
        # Entrenar modelo
        self.model, history, initial_pred, final_pred, actual = train_model(
            self.X, self.y, lr=lr, epochs=epochs)
        
        # Mostrar tabla de pesos
        weights_df = history.get_dataframe()
        
        self.weights_tree["columns"] = list(weights_df.columns)
        for col in weights_df.columns:
            self.weights_tree.heading(col, text=col)
            
        for i in self.weights_tree.get_children():
            self.weights_tree.delete(i)
        
        for i, row in weights_df.iterrows():
            self.weights_tree.insert("", "end", values=list(row))
            
        # Mostrar gráfica de error
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(actual, 'b-', label='y deseada')
        ax.plot(final_pred, 'r--', label='y calculada')
        ax.set_title('Comparación y deseada vs y calculada')
        ax.set_xlabel('Muestra')
        ax.set_ylabel('Valor')
        ax.legend()
        self.canvas.draw()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = NeuralNetworkGUI()
    app.run()
