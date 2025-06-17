import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Class names for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

MODEL_PATH = 'cnn_model.h5'

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_model():
    print("Training model...")
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = build_model()
    model.fit(train_images, train_labels, epochs=10, validation_split=0.1)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {test_acc:.2f}")
    model.save(MODEL_PATH)
    print("Model saved to cnn_model.h5")

def predict_image(model, image_path):
    img = Image.open(image_path).resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 32, 32, 3))
    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)]

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        label.config(text="Predicting...")
        result = predict_image(model, file_path)
        img = Image.open(file_path).resize((150, 150))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        label.config(text=f"Prediction: {result}")

# Train model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    train_and_save_model()

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# UI Setup
app = tk.Tk()
app.title("Image Classifier")
app.geometry("300x400")
btn = tk.Button(app, text="Select Image", command=open_file)
btn.pack(pady=20)
image_label = tk.Label(app)
image_label.pack()
label = tk.Label(app, text="Prediction: ", font=("Arial", 14))
label.pack(pady=10)
app.mainloop()
