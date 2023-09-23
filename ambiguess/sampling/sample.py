"""Data-Classes Related to Sampling."""

import dataclasses

import numpy as np


@dataclasses.dataclass
class Sample:
    """A sample drawn from the latent space"""
    coordinates: np.ndarray
    image: np.ndarray
    class_1: int
    class_2: int
    label_1: float
    label_2: float
    num_classes: int
    autoencoder_id: int


