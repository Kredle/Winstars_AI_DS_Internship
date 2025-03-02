# MNIST Classification (Task_1)

## Project Overview
This project demonstrates how to classify handwritten digits from the **MNIST dataset** using three different models:
- **Random Forest Classifier** (`sklearn`)
- **Feed-Forward Neural Network (MLPClassifier)** (`sklearn`)
- **Convolutional Neural Network (CNN)** (`TensorFlow/Keras`)

The dataset consists of **70,000 grayscale images (28x28 pixels)** representing digits from **0 to 9**.

## Features
- **Data Loading**: Fetching the MNIST dataset from `keras.datasets`.
- **Preprocessing**: Normalization and reshaping of image data.
- **Model Training**: Using `RandomForestClassifier`, `MLPClassifier`, and a custom `CNN` model.
- **Evaluation**: Measuring accuracy using `accuracy_score`.
- **Encapsulation**: Using an interface (`MnistClassifierInterface`) to unify different models.

## Project Structure
```
Task_1/
│── winstars_ai_task_1.ipynb  # Main script for training and testing the models
│── requirements.txt     # List of required dependencies
│── README.md            # Project documentation
```

This script will:
- Load the MNIST dataset.
- Train and evaluate three models (`RandomForest`, `MLPClassifier`, `CNN`).
- Display accuracy for each model.

## 📝 Code Explanation
### `winstars_ai_task_1.ipynb`

#### 📌 Importing Required Libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from tensorflow.keras import layers, models, Input
```

#### 📌 Loading MNIST Dataset
```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

#### 📌 Defining a MnistClassifierInterface
```python
from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass
```

#### Implementing Different Models

##### **Random Forest Model**
```python
class RandomForestModel(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, x_train, y_train):
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train_flat, y_train)

    def predict(self, x_test):
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        return self.model.predict(x_test_flat)
```

##### **Feed-Forward Neural Network Model**
```python
class FeedForwardNNModel(MnistClassifierInterface):
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=10,
            verbose=True
        )

    def train(self, x_train, y_train):
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train_flat, y_train)

    def predict(self, x_test):
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        return self.model.predict(x_test_flat)
```

##### **Convolutional Neural Network Model**
```python
class CNNModel(MnistClassifierInterface):
    def __init__(self):
        self.model = models.Sequential([
            Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x_train, y_train):
        x_train_exp = x_train[..., np.newaxis]
        self.model.fit(x_train_exp, y_train, epochs=5, batch_size=32, verbose=2)

    def predict(self, x_test):
        x_test_exp = x_test[..., np.newaxis]
        predictions = self.model.predict(x_test_exp)
        return np.argmax(predictions, axis=1)
```

#### Model Selection and Execution
```python
class MnistClassifier:
    def __init__(self, algorithm: str):
        if algorithm == 'rf':
            self.model = RandomForestModel()
        elif algorithm == 'nn':
            self.model = FeedForwardNNModel()
        elif algorithm == 'cnn':
            self.model = CNNModel()
        else:
            raise ValueError("Invalid algorithm. Use 'rf', 'nn', or 'cnn'.")

    def train(self, x_train, y_train):
        self.model.train(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
```

#### Running the Models and Evaluating Accuracy
```python
x_train, x_test = x_train / 255.0, x_test / 255.0

for algorithm in ['rf', 'nn', 'cnn']:
    classifier = MnistClassifier(algorithm=algorithm)
    classifier.train(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy ({algorithm}): {accuracy:.4f}')
```

## Results
- **Random Forest Accuracy**: ~94%
- **Feed-Forward NN Accuracy**: ~97%
- **CNN Accuracy**: ~99%

---

