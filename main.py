import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tkinter as tk
from tkinter import filedialog, ttk
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

# Entrenamiento modificado para guardar historial
def train_model(X, y, lr=0.01, epochs=50, batch_size=32):
    model = LinearRegression()
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.SGD(learning_rate=lr)

    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
    training_history = []
    
    for epoch in range(epochs):
        w0 = model.get_weights()  # Pesos iniciales de la época
        epoch_loss = None
        
        for batch_X, batch_y in dataset:
            with tf.GradientTape() as tape:
                predictions = model(batch_X)
                loss = loss_fn(batch_y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss = loss.numpy()
        
        wf = model.get_weights()  # Pesos finales de la época
        training_history.append({
            'epoch': epoch + 1,
            'w0': w0,
            'wf': wf,
            'error': epoch_loss
        })
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss}")
    
    return model, training_history

# Predicción
def predict(model, scaler_X, scaler_y, new_data):
    new_data = np.array(new_data).astype(np.float32).reshape(1, -1)
    new_data = scaler_X.transform(new_data)
    prediction = model(new_data).numpy()
    return scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()

# Interfaz gráfica con Tkinter
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        global dataset_X, dataset_y, scaler_X, scaler_y
        df, dataset_X, dataset_y, scaler_X, scaler_y = load_data(file_path)
        for i in tree.get_children():
            tree.delete(i)
        for row in df.itertuples(index=False):
            tree.insert("", "end", values=row)
        train_btn.config(state=tk.NORMAL)

def train_and_show_weights():
    global trained_model, training_history
    trained_model, training_history = train_model(dataset_X, dataset_y)
    
    # Actualizar tabla de pesos
    for i in weights_tree.get_children():
        weights_tree.delete(i)
        
    for entry in training_history:
        w0_weights = ', '.join([f"{w:.4f}" for w in entry['w0'][0].flatten()] + [f"{entry['w0'][1][0]:.4f}"])
        wf_weights = ', '.join([f"{w:.4f}" for w in entry['wf'][0].flatten()] + [f"{entry['wf'][1][0]:.4f}"])
        weights_tree.insert("", "end", values=(
            entry['epoch'],
            w0_weights,
            wf_weights,
            f"{entry['error']:.4f}"
        ))
    
    # Actualizar gráfico yd vs yc
    update_plot()
    predict_btn.config(state=tk.NORMAL)

def update_plot():
    # Limpiar frame anterior
    for widget in plot_frame.winfo_children():
        widget.destroy()
    
    # Generar predicciones
    yd = scaler_y.inverse_transform(dataset_y.reshape(-1, 1)).flatten()
    yc_scaled = trained_model(dataset_X).numpy()
    yc = scaler_y.inverse_transform(yc_scaled).flatten()
    
    # Crear figura
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(yd, yc, alpha=0.5)
    plt.plot([min(yd), max(yd)], [min(yd), max(yd)], 'k--')
    plt.xlabel("Valor Real (yd)")
    plt.ylabel("Valor Predicho (yc)")
    plt.title("Comparación yd vs yc")
    
    # Integrar gráfico en Tkinter
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def get_weights(model):
    w, b = model.dense.get_weights()
    return f"Pesos: {w.flatten()}, Bias: {b}"

def make_prediction():
    input_values = [float(entry.get()) for entry in input_entries]
    prediction = predict(trained_model, scaler_X, scaler_y, input_values)
    prediction_label.config(text=f"Predicción: {prediction[0]:.4f}")

# Configuración de la ventana
root = tk.Tk()
root.title("Red Neuronal Artificial - Regresión Lineal")

# Frame principal
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Sección de datos
data_frame = tk.Frame(main_frame)
data_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

tree = ttk.Treeview(data_frame, columns=("Datos"), show="headings")
tree.pack(fill=tk.BOTH, expand=True)

# Sección de entrenamiento
train_frame = tk.Frame(main_frame)
train_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Botones
button_frame = tk.Frame(train_frame)
button_frame.pack()

select_btn = tk.Button(button_frame, text="Seleccionar CSV", command=select_file)
select_btn.pack(side=tk.LEFT, padx=5)

train_btn = tk.Button(button_frame, text="Entrenar Modelo", command=train_and_show_weights, state=tk.DISABLED)
train_btn.pack(side=tk.LEFT, padx=5)

# Tabla de pesos
weights_tree = ttk.Treeview(train_frame, columns=("Época", "Pesos Iniciales (w0)", "Pesos Finales (wf)", "Error"), show="headings")
weights_tree.heading("Época", text="Época")
weights_tree.heading("Pesos Iniciales (w0)", text="Pesos Iniciales (w0)")
weights_tree.heading("Pesos Finales (wf)", text="Pesos Finales (wf)")
weights_tree.heading("Error", text="Error")
weights_tree.pack(fill=tk.BOTH, expand=True)

# Sección de predicción
predict_frame = tk.Frame(train_frame)
predict_frame.pack()

input_entries = [tk.Entry(predict_frame, width=10) for _ in range(3)]
for entry in input_entries:
    entry.pack(side=tk.LEFT, padx=5)

predict_btn = tk.Button(train_frame, text="Predecir", command=make_prediction, state=tk.DISABLED)
predict_btn.pack(pady=10)

prediction_label = tk.Label(train_frame, text="Predicción: ")
prediction_label.pack()

# Marco para la gráfica
plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True)

root.mainloop()
