"""Utilities to store the dataset as numpy arrays."""
import os.path
from typing import List, Tuple

import numpy as np

from sampling.sample import Sample


def to_numpy_arrays(samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    """Creates images / probabilistic lables numpy arrays"""
    images = np.array([sample.image for sample in samples])
    # Convert float-images to uint8-images
    uint8_images = np.floor(images * 256).astype(np.uint8)

    def prob_labels(sample: Sample) -> np.ndarray:
        """Converts a sample's probabilistic labels to a numpy array"""
        res = np.zeros(sample.num_classes, dtype=float)
        res[sample.class_1] = sample.label_1
        res[sample.class_2] = sample.label_2
        return res

    labels = np.array([prob_labels(sample) for sample in samples])
    return uint8_images, labels


def save_numpy_arrays(images: np.ndarray,
                      prob_labels: np.ndarray,
                      det_labels: np.ndarray,
                      datasets_folder: str,
                      dataset: str,
                      split: str):
    """Saves numpy arrays to disk."""
    folder = os.path.join(datasets_folder, "npy-arrays")
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(os.path.join(folder, f"{dataset}-{split}-images.npy"), images)
    np.save(os.path.join(folder, f"{dataset}-{split}-prob-labels.npy"), prob_labels)
    np.save(os.path.join(folder, f"{dataset}-{split}-det-labels.npy"), det_labels)


def load_numpy_arrays(datasets_folder: str, dataset: str, split: str):
    """Loads numpy arrays from disk."""
    folder = os.path.join(datasets_folder, "npy-arrays")
    images = np.load(os.path.join(folder, f"{dataset}-{split}-images.npy"))
    prob_labels = np.load(os.path.join(folder, f"{dataset}-{split}-prob-labels.npy"))
    det_labels = np.load(os.path.join(folder, f"{dataset}-{split}-det-labels.npy"))
    return images, det_labels, prob_labels
