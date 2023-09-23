"""Utility classes to retrieve the 5 different test sets."""
import abc
import os.path
from typing import Tuple

import numpy
import numpy as np
import tensorflow as tf

from packing.numpy_arrays import load_numpy_arrays

STATIC_DATASETS_FOLDER = "../resources/"


def get(dataset_name: str, artifact_path: str) -> 'TestSets':
    """Factory method to get a `TestSets` object for the given dataset name."""
    if dataset_name == "mnist":
        return MnistTestSets(artifact_path)
    elif dataset_name == "fmnist":
        return FashionMnistTestSets(artifact_path)


class TestSets(abc.ABC):
    """Utility class to collect the 5 different types of test sets."""

    def __init__(self,
                 dataset: str,
                 artifact_path: str):
        self.dataset = dataset
        self.artifact_path = artifact_path

    def mixed_ambiguous_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """A training set consisting of all nominal and ambiguous training data."""
        nom_x, nom_y, = self._nominal_training_data()
        amb_x, amb_y, _ = load_numpy_arrays(
            datasets_folder=os.path.join(self.artifact_path, self.dataset, 'training'),
            dataset=self.dataset, split='training',
        )
        return np.concatenate((nom_x, amb_x)), np.concatenate((nom_y, amb_y))

    @abc.abstractmethod
    def _nominal_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """A training set consisting of all nominal training data."""
        pass

    def ambiguous_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """A test set consisting of all ambiguous test data."""
        amb_x, amb_y, _ = load_numpy_arrays(
            datasets_folder=os.path.join(self.artifact_path, self.dataset, 'test'),
            dataset=self.dataset, split='test',
        )
        return amb_x, amb_y

    @abc.abstractmethod
    def invalid_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """A test set consisting of all invalid test data."""
        pass

    @abc.abstractmethod
    def corrupted_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """A test set consisting of all corrupted test data."""
        pass

    @abc.abstractmethod
    def nominal_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """A test set consisting of all nominal test data."""
        pass

    def adversarial_test_data(self, run_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """A test set consisting of all adversarial test data for a given run."""
        folder = os.path.join(self.artifact_path, "supervisor_benchmark", "runs", str(run_id), "adv_data", self.dataset)
        return (
            numpy.load(os.path.join(folder, "adversarial_x.npy")),
            numpy.load(os.path.join(folder, "adversarial_y.npy"))
        )

    def non_nominal_as_dict(self, run_id: int) -> dict:
        """A dictionary of all non-nominal test data for a given run."""
        res = dict()
        res['ambiguous'] = self.ambiguous_test_data()
        res['invalid'] = self.invalid_test_data()
        res['corrupted'] = self.corrupted_test_data()
        res['adversarial'] = self.adversarial_test_data(run_id=run_id)
        return res


class MnistTestSets(TestSets):
    """Utility class to collect the 5 different types of test sets for the MNIST dataset."""

    def __init__(self, artifact_path: str):
        super().__init__(dataset='mnist', artifact_path=artifact_path)

    # docstr-coverage:inherited
    def _nominal_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return tf.keras.datasets.mnist.load_data()[0]

    # docstr-coverage:inherited
    def invalid_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x, ignored_labels = tf.keras.datasets.fashion_mnist.load_data()[1]
        return x, np.full_like(ignored_labels, -1)

    # docstr-coverage:inherited
    def corrupted_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.load(os.path.join(STATIC_DATASETS_FOLDER, 'mnist_c_images.npy'))
        x = x.reshape((10000, 28, 28))
        y = np.load(os.path.join(STATIC_DATASETS_FOLDER, 'mnist_c_labels.npy'))
        return x, y

    # docstr-coverage:inherited
    def nominal_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return tf.keras.datasets.mnist.load_data()[1]


class FashionMnistTestSets(TestSets):
    """Utility class to collect the 5 different types of test sets for the Fashion MNIST dataset."""

    def __init__(self, artifact_path: str):
        super().__init__(dataset='fmnist', artifact_path=artifact_path)

    # docstr-coverage:inherited
    def _nominal_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return tf.keras.datasets.fashion_mnist.load_data()[0]

    # docstr-coverage:inherited
    def invalid_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x, ignored_labels = tf.keras.datasets.mnist.load_data()[1]
        return x, np.full_like(ignored_labels, -1)

    # docstr-coverage:inherited
    def corrupted_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.load(os.path.join(STATIC_DATASETS_FOLDER, 'fmnist-c-test.npy')),
            np.load(os.path.join(STATIC_DATASETS_FOLDER, 'fmnist-c-test-labels.npy')),
        )

    # docstr-coverage:inherited
    def nominal_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return tf.keras.datasets.fashion_mnist.load_data()[1]
