'''
Practica 8.
Implementar una CNN para clasificar el dataset MNIST (KAGGLE).
RICARDO AARON HERNANDEZ MACIAS
218588048

'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import struct
import matplotlib.pyplot as plt

# Funciones para cargar los archivos IDX
# Hace cuenta que el 'dataset' con las imagenes esta en archivos raros (.idx) por lo que se usan estas funciones para poder leerlos
def load_images(filename):
    with open(filename, 'rb') as f:
        _, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols, 1)
    return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        _, num_labels = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# Cargar los datos de entrenamiento y prueba
train_images = load_images('./MNIST/train-images.idx3-ubyte')
train_labels = load_labels('./MNIST/train-labels.idx1-ubyte')
test_images = load_images('./MNIST/t10k-images.idx3-ubyte')
test_labels = load_labels('./MNIST/t10k-labels.idx1-ubyte')

# Normalizamos las imágenes
train_images = train_images / 255.0
test_images = test_images / 255.0

# Definimos la arquitectura de la CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 salidas para cada dígito (0-9)

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Graficar la precisión del entrenamiento y validación
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Probar el modelo con algunas imágenes del conjunto de prueba
num_test_images = 5     #cambia esta variable por el numero de pruebas que quieres hacer desp del entenamiento y la grafica anterior
for i in range(num_test_images):
    test_img = test_images[i].reshape(1, 28, 28, 1)
    prediction = model.predict(test_img)
    predicted_label = np.argmax(prediction)

    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicción: {predicted_label}, Etiqueta real: {test_labels[i]}')
    plt.show()
