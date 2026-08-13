# Pulsar Classification Using an Artificial Neural Network

## Project Overview

This project builds a small **Artificial Neural Network (ANN)** using **TensorFlow/Keras** to classify pulsar candidates as either pulsars or non-pulsars.

The model uses the exact architecture required for the assignment:

```text
Dense(16, activation="relu")
        ↓
Dense(8, activation="relu")
        ↓
Dense(1, activation="sigmoid")
```

The network is trained using the **Adam optimizer** and **binary cross-entropy loss** for binary classification.

## Dataset

https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate

The dataset contains:

* 8 numerical input features
* 1 target column: `target_class`

The target is binary:

* `0` = Non-pulsar
* `1` = Pulsar

Some input columns contain missing values, so median imputation is applied before model training.

## Data Preprocessing

The preprocessing pipeline includes:

1. Loading the dataset using Pandas.
2. Removing unnecessary spaces from column names.
3. Separating the input features and target variable.
4. Splitting the labeled dataset into:

   * 80% training data
   * 20% testing data
5. Using `stratify=y` to preserve the target-class distribution.
6. Replacing missing values using median imputation.
7. Standardizing the numerical features using `StandardScaler`.

## ANN Architecture

The neural network contains three Dense layers:

```python
model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])
```

The first two layers use **ReLU** activation, while the output layer uses **Sigmoid** activation because this is a binary classification problem.

## Model Compilation

The model is compiled using:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

* **Optimizer:** Adam
* **Loss Function:** Binary Cross-Entropy
* **Evaluation Metric:** Accuracy

## Model Training

The ANN is trained for **30 epochs** using a batch size of 32.

```python
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.20
)
```

A portion of the training data is used as validation data so that the training and validation loss can be monitored across epochs.

## Model Evaluation

After training, the model is evaluated on the held-out test dataset.

The notebook achieved approximately:

```text
Test Accuracy: 98.16%
```

The exact accuracy can vary slightly between runs because neural-network training includes some randomness.

## Training vs Validation Loss

The notebook plots both training loss and validation loss across all 30 epochs.

This plot helps evaluate whether the neural network is learning effectively and whether there are signs of overfitting.

```python
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()
```

## Forward Propagation and Backpropagation

**Forward propagation** passes the input features through each layer of the neural network to generate a prediction, and the loss function measures how different the prediction is from the actual target.

**Backpropagation** sends the error information backward through the network and calculates how much each weight contributed to the error. The Adam optimizer then updates the weights so that the model can make better predictions during the next epoch.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow
* Keras
* Google Colab / Jupyter Notebook

## Files

```text
ANN.ipynb
pulsar_data_train.csv
README.md
```

## How to Run

1. Open `ANN.ipynb` in Google Colab or Jupyter Notebook.
2. Upload `pulsar_data_train.csv`.
3. Run all notebook cells in order.
4. Train the ANN for 30 epochs.
5. View the final test accuracy.
6. View the training vs validation loss graph.

## Conclusion

The ANN successfully learns to classify pulsar candidates using the provided numerical features. With the required three-layer neural-network architecture and appropriate preprocessing, the model achieves approximately **98% test accuracy**, showing strong classification performance on the held-out test data.
