"""Measures Softmax and MC-Dropout performance."""
import gc
from typing import Tuple

import numpy as np
import tensorflow as tf

from supervisor_benchmark import model_architectures
from supervisor_benchmark import result_files
from supervisor_benchmark.binary_classification_utils import Evaluation
from supervisor_benchmark.test_set import TestSets

APPROACHES = [
    "pred_entropy",
    "MI",
    "MS",
]


def run(data: TestSets,
        run_id: int):
    """Perform the evaluation and persist the results."""
    architecture = model_architectures.architecture_choice(run_id=run_id)
    model = architecture.train_or_load_ensemble(data.dataset, run_id=run_id, num_processes=0)

    non_nominal_as_dict = data.non_nominal_as_dict(run_id=run_id)
    # Remove adversarial data from ensemble
    try:
        non_nominal_as_dict.pop("adversarial")
    except KeyError:
        print("No adversarial data to remove.")

    if len(non_nominal_as_dict) == 0:
        print("No non-nominal data to evaluate.")
        return

    # Run for nominal data (which is also used later)
    nom_outputs = _quantify(data.nominal_test_data(), model)
    evals = dict()
    for i, approach in enumerate(APPROACHES):
        nom_pred, nom_unc = nom_outputs[i]
        evals[approach] = Evaluation(nom_unc)

    for test_set_name, test_set_x_y in non_nominal_as_dict.items():
        outputs = _quantify(test_set_x_y, model)
        for i, approach in enumerate(APPROACHES):
            auc_roc = evals[approach].auc_roc(outputs[i][1])
            result_files.register_results(
                dataset_and_split=(data.dataset, test_set_name),
                supervisor=_uwiz_to_result_name(approach),
                metric="auc_roc",
                value=auc_roc,
                run_id=run_id,
                artifacts_folder=data.artifact_path)

    tf.keras.backend.clear_session()
    gc.collect()


def _uwiz_to_result_name(supervisor_uwiz_name: str) -> str:
    """Converts a supervisor name from uwiz alias to a name used in the result files."""
    if supervisor_uwiz_name == "pred_entropy":
        return "Deep Ensemble (PE)"
    elif supervisor_uwiz_name == "MI":
        return "Deep Ensemble (MI)"
    elif supervisor_uwiz_name == "MS":
        return "Deep Ensemble (MS)"
    else:
        raise ValueError(f"Unknown supervisor name: {supervisor_uwiz_name}")


def _quantify(data: Tuple[np.ndarray, np.ndarray], model):
    x = model_architectures.preprocess_x(data[0])
    return model.predict_quantified(x=x,
                                    quantifier=APPROACHES,
                                    as_confidence=False,
                                    num_processes=0)
