"""Utilities to load all successfully trained rAAEs from disk."""
import os
from typing import List, Tuple

AE_DEF = Tuple[int, int, int, str]


def get_r_aaes_in_model_folder(model_folder) -> List[AE_DEF]:
    """
    Get the list of all the r_aaes in a model folder.
    Returns:
        list: a list of tuples (class1, class2, autoencoder_id, filename)
    """
    aes_defs = []
    for filename in os.listdir(model_folder):
        if not os.path.isdir(f"{model_folder}/{filename}"):
            continue

        try:
            first_class, second_class, ae_id = filename.split("-")
            first_class, second_class, ae_id = int(first_class), int(second_class), int(ae_id)
        except:
            raise ValueError(f"Unexpected folder: {filename}")

        aes_defs.append((first_class, second_class, ae_id, filename))
    return aes_defs
