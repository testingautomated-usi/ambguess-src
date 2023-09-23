import abc

import os.path
from typing import Union, List, Tuple

import numpy as np
import tensorflow as tf
import uncertainty_wizard as uwiz
import uncertainty_wizard.models

from supervisor_benchmark import test_set, adversarial_attack
from supervisor_benchmark.model_utils.densenet import densenet_model
from supervisor_benchmark.test_set import TestSets

Activations = Union[List[np.ndarray], np.ndarray]
"""Type Alias for activation traces. Shape: (samples x neurons). 

Typically a two dimensional float numpy array.
For convenience, we allow the user to pass in a higher dimensional array,
or a list of activations. In both cases, we flatten the input to the required
two dimensional format."""


def _run_folder(run_id: int) -> str:
    return os.path.join("/artifacts", "supervisor_benchmark", "runs", str(run_id))


def preprocess_x(x: np.ndarray) -> np.ndarray:
    """Preprocess the input data."""
    x = x.astype("float32") / 255
    x = np.expand_dims(x, -1)
    return x


def architecture_choice(run_id: int):
    if run_id < 5:
        return SimpleCnn()
    elif run_id < 10:
        return FullyConnectedNet()
    elif run_id < 15:
        return Resnet()
    elif run_id < 20:
        return Densenet()
    else:
        raise ValueError("unexpected run id")


class ModelArchitecture(abc.ABC):
    """Abstract superclass for all architectures"""

    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of the architecture"""

    @abc.abstractmethod
    def get_sa_layers(self):
        """Returns the layer indexes to be used for SA."""

    @abc.abstractmethod
    def get_dissector_layers(self):
        """Returns the layer indexes to be used for dissector"""

    def train_or_load_model(self,
                            dataset: TestSets,
                            run_id: int) -> uwiz.models.StochasticSequential:
        """Get the model from the file system, or train it if it does not exist."""
        model_path = os.path.join(_run_folder(run_id), f"standard_model_{dataset.dataset}")
        if os.path.exists(model_path):
            model = uwiz.models.load_model(model_path)
        else:
            x, y = dataset.mixed_ambiguous_training_data()
            model = self.train_model(x, y)
            model.save(model_path)
            # Create adversarial data
            print("Done creating model. Now creating adversarial data.")
            # with tf.device("/cpu:0"):
            #     model = uwiz.models.load_model(model_path)
            adversarial_attack.attack(dataset=dataset,
                                      run_id=run_id,
                                      model=model.inner)
        return model

    def train_or_load_ensemble(self,
                               dataset_name: str,
                               run_id: int,
                               num_processes: int) -> uwiz.models.LazyEnsemble:
        """Get the ensemble from the file system, or train it if it does not exist."""
        model_path = os.path.join(_run_folder(run_id), f"ensemble_model_{dataset_name}")
        if os.path.exists(model_path):
            return uwiz.models.load_model(model_path)
        else:
            os.makedirs(model_path)
            model = uwiz.models.LazyEnsemble(
                model_save_path=model_path,
                num_models=20,
                default_num_processes=num_processes
            )

        if dataset_name == "mnist":
            model.create(self._training_ensemble_wrapper_mnist())
        elif dataset_name == "fmnist":
            model.create(self._training_ensemble_wrapper_fmnist())
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        return model

    @abc.abstractmethod
    def _training_ensemble_wrapper_mnist(self):
        """A picklable subclass model creation Callable for mnist"""

    @abc.abstractmethod
    def _training_ensemble_wrapper_fmnist(self):
        """A picklable subclass model creation Callable for fmnist"""

    @abc.abstractmethod
    def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> uwiz.models.StochasticSequential:
        """Trains a new model instance."""

    @staticmethod
    def get_activations_and_pred(model: uwiz.models.StochasticSequential,
                                 layers: List[int],
                                 x: np.ndarray) -> Tuple[Activations, np.ndarray]:
        """Get activations for a model.

        The activations are returned as a list of numpy arrays, one for each layer.
        """
        if x.dtype == np.uint8:
            x = preprocess_x(x)
        keras_model = model.inner
        inp = keras_model.input
        outputs = [layer.output for i, layer in enumerate(keras_model.layers)
                   if i in layers]
        outputs.append(keras_model.output)
        transparent_model = tf.keras.Model(inputs=inp, outputs=outputs)

        activations = transparent_model.predict(x, verbose=1)
        inner_activations = activations[:-1]
        pred = np.argmax(activations[-1], axis=1)
        return inner_activations, pred


def _simple_cnn_mnist_trainer(model_id: int):
    x, y = test_set.MnistTestSets("/artifacts").mixed_ambiguous_training_data()
    return SimpleCnn().train_model(x, y), None


def _simple_cnn_fmnist_trainer(model_id: int):
    x, y = test_set.FashionMnistTestSets("/artifacts").mixed_ambiguous_training_data()
    return SimpleCnn().train_model(x, y), None


class SimpleCnn(ModelArchitecture):
    """Simple, off-the-shelve convolutional neural network.

    Taken from https://keras.io/examples/vision/mnist_convnet/"""

    def name(self) -> str:
        return "Conv. NN"

    def _training_ensemble_wrapper_mnist(self):
        return _simple_cnn_mnist_trainer

    def _training_ensemble_wrapper_fmnist(self):
        return _simple_cnn_fmnist_trainer

    def get_sa_layers(self):
        return [4]

    def get_dissector_layers(self):
        return list(range(4))

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> uwiz.models.StochasticSequential:
        """Simple, off-the-shelve convolutional neural network.
        Taken from https://keras.io/examples/vision/mnist_convnet/"""

        x_train = preprocess_x(x_train)

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


def _simple_fc_mnist_trainer(model_id: int):
    x, y = test_set.MnistTestSets("/artifacts").mixed_ambiguous_training_data()
    return FullyConnectedNet().train_model(x, y), None


def _simple_fc_fmnist_trainer(model_id: int):
    x, y = test_set.FashionMnistTestSets("/artifacts").mixed_ambiguous_training_data()
    return FullyConnectedNet().train_model(x, y), None


class FullyConnectedNet(ModelArchitecture):

    def name(self) -> str:
        return "Fully Connected NN."

    def get_sa_layers(self):
        return [3]

    def get_dissector_layers(self):
        return [1, 3]

    def _training_ensemble_wrapper_mnist(self):
        return _simple_fc_mnist_trainer

    def _training_ensemble_wrapper_fmnist(self):
        return _simple_cnn_mnist_trainer

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> uwiz.models.StochasticSequential:
        x_train = preprocess_x(x_train)

        # convert class vectors to binary class matrices
        y_train = tf.keras.utils.to_categorical(y_train, 10)

        model = uwiz.models.StochasticSequential([
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])

        batch_size = 128
        epochs = 30

        # Shuffle to get both ambiguous and nominal data in validation set
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        return model


class Resnet_Reshape(tf.keras.layers.Layer):
    def __init__(self):
        super(Resnet_Reshape, self).__init__(trainable=False)

    def build(self, input_shape):  # Create the state of the layer (weights)
        pass

    def call(self, inputs):  # Defines the computation from inputs to outputs
        x = tf.reshape(inputs, (-1, 28, 28))
        return tf.stack((x, x, x), axis=-1)


def _resnet_mnist_trainer(model_id: int):
    x, y = test_set.MnistTestSets("/artifacts").mixed_ambiguous_training_data()
    return Resnet().train_model(x, y), None


def _resnet_fmnist_trainer(model_id: int):
    x, y = test_set.FashionMnistTestSets("/artifacts").mixed_ambiguous_training_data()
    return Resnet().train_model(x, y), None


class Resnet(ModelArchitecture):
    """Model built on top of Resnet50.

    Inspired by https://www.kaggle.com/code/donatastamosauskas/using-resnet-for-mnist/notebook
    """

    def name(self) -> str:
        return "Resnet50"

    def get_sa_layers(self):
        return [4]

    def get_dissector_layers(self):
        return [2, 4]

    def _training_ensemble_wrapper_mnist(self):
        return _resnet_mnist_trainer

    def _training_ensemble_wrapper_fmnist(self):
        return _resnet_fmnist_trainer

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> uwiz.models.StochasticSequential:
        x_train = preprocess_x(x_train)
        # convert class vectors to binary class matrices
        y_train = tf.keras.utils.to_categorical(y_train, 10)

        model = uwiz.models.StochasticSequential([
            tf.keras.Input(shape=(28, 28, 1)),
            Resnet_Reshape(),
            tf.keras.applications.resnet50.ResNet50(include_top=False, pooling='avg'),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(125, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])

        model.inner.layers[0].trainable = False

        batch_size = 128
        epochs = 30

        # Shuffle to get both ambiguous and nominal data in validation set
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        return model


def _densenet_mnist_trainer(model_id: int):
    x, y = test_set.MnistTestSets("/artifacts").mixed_ambiguous_training_data()
    return Densenet().train_model(x, y), None


def _densenet_fmnist_trainer(model_id: int):
    x, y = test_set.FashionMnistTestSets("/artifacts").mixed_ambiguous_training_data()
    return Densenet().train_model(x, y), None


class Densenet(ModelArchitecture):
    """Model built on top of Densenet (similar to the Resnet above)"""

    def name(self) -> str:
        return "Densenet"

    def get_sa_layers(self):
        return [218]

    def get_dissector_layers(self):
        # All of the following filter, except the last one (i.e, all dense but last one)
        # [i for i,l in enumerate(model.inner.layers) if isinstance(l, tf.keras.layers.Dense)]
        return [68, 70, 81, 83, 209, 211]

    def _training_ensemble_wrapper_mnist(self):
        return _densenet_mnist_trainer

    def _training_ensemble_wrapper_fmnist(self):
        return _densenet_fmnist_trainer

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> uwiz.models.StochasticSequential:
        x_train = preprocess_x(x_train)
        # convert class vectors to binary class matrices
        y_train = tf.keras.utils.to_categorical(y_train, 10)

        keras_model = densenet_model(classes=10, shape=(28, 28, 1), nb_layers=[6, 12], nb_filter=16, dropout_rate=0.1)
        model = uncertainty_wizard.models.stochastic_from_keras(keras_model)

        batch_size = 256
        epochs = 15

        # Shuffle to get both ambiguous and nominal data in validation set
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        return model


if __name__ == '__main__':
    Resnet().train_model(np.zeros((1000, 28, 28)), np.ones((1000, 1)))
