# Fashion-MNIST Classification with a Multi-Layer Perceptron

This repository contains a Jupyter Notebook that implements a fully connected neural network (MLP) for image classification on the **Fashion-MNIST** dataset using Keras.

The project covers the complete workflow from data loading and preprocessing to model training, prediction inspection, and class-level evaluation.

---

## Project Overview

The notebook performs the following steps:

```text
Fashion-MNIST
      |
      v
Load Training and Test Data
      |
      v
Visualize Sample Images
      |
      v
Flatten 28×28 Images to 784-D Vectors
      |
      v
Convert to float32
      |
      v
Normalize Pixel Values to [0, 1]
      |
      v
One-Hot Encode Labels
      |
      v
Build MLP Classifier
      |
      v
Train for 30 Epochs
      |
      v
Evaluate Accuracy
      |
      v
Inspect Correct and Incorrect Predictions
      |
      v
Classification Report + Confusion Matrix
```

---

## Dataset

The project uses the **Fashion-MNIST** dataset available directly through Keras.

Fashion-MNIST contains grayscale images with a resolution of:

```text
28 × 28 pixels
```

The dataset contains:

```text
60,000 training images
10,000 test images
10 classes
```

The classes are:

| Label | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Data Preprocessing

Each `28 × 28` image is flattened into a vector of length:

```text
784
```

The resulting shapes are:

```text
Training matrix: (60000, 784)
Testing matrix:  (10000, 784)
```

The input arrays are converted to `float32` and normalized from the original pixel range:

```text
[0, 255]
```

to:

```text
[0, 1]
```

The integer class labels are converted into 10-dimensional one-hot vectors using Keras `to_categorical`.

---

## Model Architecture

The classifier is implemented using Keras `Sequential`.

Architecture:

```text
Input: 784 features
      |
      v
Dense(512, ReLU)
      |
      v
Dropout(0.2)
      |
      v
Dense(256, ReLU)
      |
      v
Dropout(0.2)
      |
      v
Dense(128)
      |
      v
ReLU
      |
      v
Dropout(0.2)
      |
      v
Dense(64, ReLU)
      |
      v
Dropout(0.2)
      |
      v
Dense(10, Softmax)
```

Dropout is used throughout the hidden layers to reduce overfitting.

---

## Training Configuration

The model is compiled with:

```text
Loss function: Categorical Cross-Entropy
Optimizer:     Adam
Learning rate: 0.001
Metric:        Accuracy
Batch size:    64
Epochs:        30
```

The notebook currently trains using:

```python
model.fit(
    X_train,
    Y_train,
    batch_size=64,
    epochs=30,
    validation_data=(X_test, Y_test)
)
```

> **Evaluation note:** The official Fashion-MNIST test set is also used as `validation_data` during training in the current notebook. Therefore, values stored by Keras as `val_accuracy` and `val_loss` are measured on the test set rather than on a separate validation split

---

## Results

In the saved notebook run, the highest monitored accuracy was:

```text
89.44%
```

at:

```text
Epoch 30
```

The final classification report shows an overall accuracy of approximately:

```text
89%
```

Class-level performance varies across categories.

Examples from the saved run include:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| T-shirt/top | 0.86 | 0.84 | 0.85 |
| Trouser | 0.99 | 0.98 | 0.99 |
| Pullover | 0.76 | 0.85 | 0.81 |
| Dress | 0.91 | 0.89 | 0.90 |
| Coat | 0.78 | 0.82 | 0.80 |
| Sandal | 0.98 | 0.98 | 0.98 |
| Shirt | 0.76 | 0.67 | 0.72 |
| Sneaker | 0.95 | 0.97 | 0.96 |
| Bag | 0.98 | 0.97 | 0.98 |
| Ankle boot | 0.97 | 0.96 | 0.97 |

The weakest class in the recorded run is `Shirt`, which is frequently confused with visually similar upper-body clothing classes.

---

## Evaluation

The notebook evaluates the trained classifier using several methods.

### Accuracy Curve

Training and monitored test accuracy are plotted across all 30 epochs.

### Correct Predictions

Random correctly classified test images are displayed together with:

- Predicted label
- Actual label

### Incorrect Predictions

Random misclassified test images are displayed to inspect common failure cases.

### Classification Report

Scikit-learn's `classification_report` is used to calculate:

- Precision
- Recall
- F1-score
- Support

for all 10 Fashion-MNIST classes.

### Confusion Matrix

A confusion matrix is generated to analyze class-to-class prediction errors.

---

## Requirements

The notebook imports the following main packages:

```text
numpy
pandas
matplotlib
seaborn
tensorflow
scikit-learn
```

Install them with:

```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

---

## Running the Notebook

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Install the dependencies:

```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Fashionmnist_mlp.ipynb
```

Then run the cells sequentially.

Fashion-MNIST is downloaded automatically by Keras if it is not already cached locally.

---

## Conclusion

This project demonstrates a complete Fashion-MNIST image-classification workflow using a Multi-Layer Perceptron implemented with Keras.

The model transforms grayscale images into flattened feature vectors, applies several fully connected layers with dropout regularization, and predicts one of ten clothing categories using a softmax output layer.

The saved experiment reaches approximately **89% classification accuracy**, while the detailed class-level evaluation highlights the greater difficulty of distinguishing visually similar categories such as shirts, pullovers, coats, and T-shirts.
