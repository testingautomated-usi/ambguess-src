"""Everything related to drawing the deterministic labels."""
import random
from typing import List

import numpy as np

from sampling.sample import Sample

PLACEHOLDER = 254


def draw_det_labels(samples: List[Sample], seed: int = 0) -> np.ndarray:
    """Draws deterministic labels for the samples."""
    rng = random.Random(seed)
    labels = np.full(len(samples), dtype=np.uint8, fill_value=PLACEHOLDER)
    for i, s in enumerate(samples):
        uniform_sample = rng.uniform(0, 1)
        if uniform_sample < s.label_1:
            labels[i] = s.class_1
        else:
            labels[i] = s.class_2

    assert np.sum(labels == PLACEHOLDER) == 0, "Some labels were not drawn"
    return labels
