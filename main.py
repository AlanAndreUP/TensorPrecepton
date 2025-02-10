import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tkinter as tk
from tkinter import filedialog, ttk
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_path):
    df = pd.read_csv(csv_path, sep=";") 
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    return df, X, y, scaler_y

# Definir el modelo de regresión lineal
class LinearRegression(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(1, kernel_initializer=tf.initializers.RandomNormal(stddev=0.01))

    def call(self, inputs):
        return self.dense(inputs)

# Función de pérdida y optimizador
def train_model(X, y, lr=0.01, epochs=50, batch_size=32):
    model = LinearRegression()
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.SGD(learning_rate=lr)

    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
    for epoch in range(epochs):
        for batch_X, batch_y in dataset:
            with tf.GradientTape() as tape:
                predictions = model(batch_X)
                loss = loss_fn(batch_y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Epoch {epoch + 1}: Loss = {loss.numpy()}")
    return model

# Evaluar los pesos aprendidos
def get_weights(model):
    w, b = model.dense.get_weights()
    return f"Pesos: {w.flatten()}, Bias: {b}"

# Interfaz gráfica con Tkinter
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        global dataset_X, dataset_y, scaler_y
        df, dataset_X, dataset_y, scaler_y = load_data(file_path)
        
        for i in tree.get_children():
            tree.delete(i)
        for row in df.itertuples(index=False):
            tree.insert("", "end", values=row)
        train_btn.config(state=tk.NORMAL)

def train_and_show_weights():
    model = train_model(dataset_X, dataset_y)
    weights_label.config(text=get_weights(model))

# Crear ventana Tkinter
root = tk.Tk()
root.title("Regresión Lineal con TensorFlow")

frame = tk.Frame(root)
frame.pack(pady=20)

select_btn = tk.Button(frame, text="Seleccionar CSV", command=select_file)
select_btn.pack()

tree = ttk.Treeview(root, columns=("data"), show="headings")
tree.pack()

train_btn = tk.Button(root, text="Entrenar Modelo", command=train_and_show_weights, state=tk.DISABLED)
train_btn.pack(pady=10)

weights_label = tk.Label(root, text="")
weights_label.pack()

root.mainloop()
