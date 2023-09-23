"""Utilities to store the dataset as csv."""
import itertools
import os
from typing import List

import numpy as np
import pandas
import tqdm

from sampling.sample import Sample


def save_as_csv(samples: List[Sample],
                imgs: np.ndarray, # Using separate numpy array for images as np.uint8
                det_labels: np.ndarray,
                datasets_folder: str,
                dataset: str,
                split: str):
    """Perist the samples, including all auxiliary information, to a csv file."""
    coordinates = list(itertools.product(list(range(28)), list(range(28))))
    entries = []
    for i, sample in tqdm.tqdm(enumerate(samples), desc='Creating CSV rows'):
        new_entry = {
            'dataset': dataset,
            'split': split,
            'class_1': sample.class_1,
            'class_2': sample.class_2,
            'p(class_1)': sample.label_1,
            'p(class_2)': sample.label_2,
            'det_label': det_labels[i],
            'autoencoder_id': sample.autoencoder_id,
        }
        for coord in coordinates:
            new_entry['x_{}'.format(coord)] = imgs[i][coord]
        entries.append(new_entry)

    df = pandas.DataFrame(entries)
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)
    df.to_csv(os.path.join(datasets_folder, f"{dataset}-{split}.csv"), index=True)
