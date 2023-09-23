"""Utiliy to store and load the raw (unpackaged) samples on the file disc."""
import os.path
import pickle
import shutil
import warnings
from typing import List

from aae.autoencoders import R_AAE
from aae.model_folder import AE_DEF, get_r_aaes_in_model_folder
from sampling.sample import Sample


def get_ae_sample_folders(split_folder: str) -> List[AE_DEF]:
    """The names of all 'per-ae' folder in the given split folder."""
    # For simplicity, i use the same naming convention as for models
    return get_r_aaes_in_model_folder(split_folder)


def unsampelled_aes(model_folder: str, split_folder: str) -> List[AE_DEF]:
    """A list of all ae_defs for which no samples have been generated yet."""
    res = get_r_aaes_in_model_folder(model_folder)
    raw_folder = os.path.join(split_folder, "raw")
    for existing_ae_def in get_ae_sample_folders(raw_folder):
        try:
            res.remove(existing_ae_def)
        except ValueError:
            warnings.warn(f"Found samples for unknown autoencoder.")
    return res


def store_samples_for_ae(r_aae: R_AAE, split_folder: str, samples: List[Sample]):
    """Store the given samples in the given split folder."""
    folder = os.path.join(split_folder, "raw", f"{r_aae.class_1}-{r_aae.class_2}-{r_aae.random_id}")

    # Create folder, delete existing content
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    # Store samples
    pickle.dump(samples, open(os.path.join(folder, "samples.pkl"), "wb"))


def load_samples_for_ae(aae_def: AE_DEF, split_folder: str) -> List[Sample]:
    """Load the samples for the given autoencoder from the given split folder."""
    folder = os.path.join(split_folder, "raw", f"{aae_def[0]}-{aae_def[1]}-{aae_def[2]}")
    return pickle.load(open(os.path.join(folder, "samples.pkl"), "rb"))


def load_all_samples_for_split(split_folder: str) -> List[Sample]:
    """Load all samples for the given split."""
    raw_folder = os.path.join(split_folder, "raw")
    res = []
    for ae_def in get_ae_sample_folders(raw_folder):
        res.extend(load_samples_for_ae(ae_def, split_folder))
    return res