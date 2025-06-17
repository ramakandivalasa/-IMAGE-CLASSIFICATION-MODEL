# IMAGE-CLASSIFICATION-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: RAMA KANDIVALASA 

*INTERN ID*: CT04DN904

*DOMAIN*: MACHINE LEARNING 

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

## 📌 Project Title: BUILD A CONVOLUTIONAL NEURAL NETWORK (CNN) FOR IMAGE CLASSIFICATION USING TENSORFLOW OR PYTORCH.

### OBJECTIVE:
To build a Convolutional Neural Network (CNN) for image classification using TensorFlow.

---

## DATASET DETAILS:

- Dataset: CIFAR-10 (downloaded automatically using Keras)
- Number of images: 60,000 total
  - 50,000 for training
  - 10,000 for testing
- Image size: 32x32 pixels, RGB (3 channels)
- Number of classes: 10
- Classes:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck

---

## REQUIREMENTS:

Create a file called `requirements.txt` with the following content:

tensorflow
numpy
Pillow

Install the dependencies with:

pip install -r requirements.txt

---

## MODEL ARCHITECTURE:

The CNN model used in this project has the following architecture:

1. Conv2D layer: 32 filters, kernel size 3x3, activation='relu', input_shape=(32, 32, 3)  
2. MaxPooling2D: pool size 2x2  
3. Conv2D layer: 64 filters, kernel size 3x3, activation='relu'  
4. MaxPooling2D: pool size 2x2  
5. Conv2D layer: 64 filters, kernel size 3x3, activation='relu'  
6. Flatten  
7. Dense layer: 64 units, activation='relu'  
8. Dense output layer: 10 units (number of classes), activation='softmax'

---

## COMPILATION DETAILS:

- Loss function: Sparse Categorical Crossentropy  
- Optimizer: Adam  
- Metrics: Accuracy  
- Epochs: 10  
- Batch size: 64  

---

## HOW TO RUN:

1. Open terminal or command prompt  
2. Navigate to your project folder  
3. Run the following commands:

pip install -r requirements.txt
python image_classifier_gui.py

To exit the terminal:

exit

---

## OUTPUT (TERMINAL LOGS):

Training model...
Epoch 1/10
782/782 [==============================] - loss: 1.5271 - accuracy: 0.4487
Epoch 2/10
782/782 [==============================] - loss: 1.1835 - accuracy: 0.5832
Epoch 3/10
782/782 [==============================] - loss: 1.0372 - accuracy: 0.6354
Epoch 4/10
782/782 [==============================] - loss: 0.9484 - accuracy: 0.6682
Epoch 5/10
782/782 [==============================] - loss: 0.8804 - accuracy: 0.6907
Epoch 6/10
782/782 [==============================] - loss: 0.8245 - accuracy: 0.7118
Epoch 7/10
782/782 [==============================] - loss: 0.7776 - accuracy: 0.7283
Epoch 8/10
782/782 [==============================] - loss: 0.7359 - accuracy: 0.7426
Epoch 9/10
782/782 [==============================] - loss: 0.6964 - accuracy: 0.7557
Epoch 10/10
782/782 [==============================] - loss: 0.6614 - accuracy: 0.7682

Evaluating on test data...
Test loss: 0.8503
Test accuracy: 0.7124

Model saved to cnn_model.h5

---

## GUI OUTPUT EXAMPLE:

When you run the script, a GUI window will open. It allows you to select an image and get the classification result.

Example output in GUI:

Predicted Class: bird


---

## PROJECT FILES:

├── image_classifier_gui.py # Main GUI and model code
├── cnn_model.h5 # Trained model (saved automatically)
├── requirements.txt # Python dependencies
└── README.md # Documentation file (this file)


---

## SUMMARY:

This project successfully demonstrates a basic CNN architecture applied to the CIFAR-10 dataset with GUI support. The model achieves around 71% accuracy on test data and can classify custom images using a simple Tkinter-based GUI.

---
![Image](https://github.com/user-attachments/assets/d34b27aa-e567-4a19-a71e-e439085c4f8c)

![Image](https://github.com/user-attachments/assets/db8d8900-c5a9-499f-9bad-d7684b5c97a6)
