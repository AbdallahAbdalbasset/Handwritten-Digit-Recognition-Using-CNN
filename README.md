# Handwritten Digit Recognition Using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits using the **MNIST dataset**. The model is built in **Python** using **TensorFlow/Keras** and includes an interactive interface for predicting digits from user-uploaded images.

---

## 📌 Features

- Preprocesses the MNIST dataset (reshaping, grayscale normalization, visualization).  
- Builds a CNN with two convolutional layers, max-pooling, flattening, and dense layers.  
- Uses **ReLU** activation for hidden layers and **Softmax** for output.  
- Trains the network with the **Adam optimizer** and **sparse categorical cross-entropy loss**.  
- Includes an early stopping callback when training accuracy reaches 98%.  
- Evaluates model performance on the test set (>90% accuracy).  
- Interactive prediction: upload an image and get the predicted digit with visualization.

---

## 🛠 Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- OpenCV  
- PIL (Pillow)  
- Google Colab

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**:  
   [Open in Colab]([https://colab.research.google.com/drive/<YOUR_NOTEBOOK_ID>](https://colab.research.google.com/drive/19w_XJtBc30OFR0Y7y37gIHsntEvorNOA?usp=sharing))  

2. Run each cell sequentially to:
   - Load and preprocess MNIST data.
   - Build and train the CNN model.
   - Evaluate the model on test images.
   - Upload custom images for prediction.

3. For predictions, upload a handwritten digit image (28×28 grayscale preferred). The notebook will preprocess and display the predicted digit.

---

## 📈 Model Architecture

- Conv2D → 32 filters, 3×3 kernel, ReLU  
- MaxPooling2D → 2×2  
- Conv2D → 64 filters, 3×3 kernel, ReLU  
- MaxPooling2D → 2×2  
- Flatten  
- Dense → 128 units, ReLU  
- Dense → 10 units, Softmax
