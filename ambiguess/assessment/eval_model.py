"""Prediction model utilities for the quantitative AmbiGuess Assessment."""
import os.path

import numpy as np
import tensorflow as tf
import uncertainty_wizard as uwiz

from supervisor_benchmark.model_architectures import FullyConnectedNet, SimpleCnn, Resnet, Densenet


def train_model(run_id: int, x_train: np.ndarray, y_train: np.ndarray):
    architecture = ambi_eval_architecture(run_id)
    return architecture.train_model(x_train, y_train)


def ambi_eval_architecture(run_id):
    if run_id == 0:
        architecture = SimpleCnn()
    elif run_id == 1:
        architecture = FullyConnectedNet()
    elif run_id == 2:
        architecture = Resnet()
    elif run_id == 3:
        architecture = Densenet()
    else:
        raise ValueError("unexpected run id")
    return architecture


def train_or_load_model(run_id: int,
                        dataset: str,
                        x_train: np.ndarray,
                        y_train: np.ndarray,
                        clean: bool) -> uwiz.models.Stochastic():
    """Loads a model from disk if it exists, otherwise trains a new one."""
    if clean:
        model_name = "model_clean"
    else:
        model_name = "model_mixed_ambiguous"

    model_path = os.path.join("/artifacts", dataset, "quant_eval", f"run_{run_id}", model_name)
    if os.path.exists(model_path):
        model = uwiz.models.load_model(model_path)
    else:
        model = train_model(run_id, x_train, y_train)
        model.save(model_path)
    return model


def train_simple_cnn(x_train: np.ndarray, y_train: np.ndarray) -> uwiz.models.Stochastic():
    """Simple, off-the-shelve convolutional neural network.

    Taken from https://keras.io/examples/vision/mnist_convnet/"""
    x_train = x_train.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    model = uwiz.models.StochasticSequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    return model
