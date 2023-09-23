"""Measures Softmax and MC-Dropout performance."""
import gc
from typing import Tuple

import numpy as np
import tensorflow as tf
import uncertainty_wizard as uwiz

from supervisor_benchmark import result_files, model_architectures
from supervisor_benchmark.binary_classification_utils import Evaluation
from supervisor_benchmark.deepgini import DeepGini
from supervisor_benchmark.test_set import TestSets

APPROACHES = [
    "SM",
    "PCS",
    "SE",
    "custom::deep_gini",
    "VR",
    "pred_entropy",
    "MI",
    "MS",
]

uwiz.quantifiers.QuantifierRegistry().register(DeepGini())


def run(data: TestSets,
        run_id: int):
    """Perform the evaluation and persist the results."""
    architecture = model_architectures.architecture_choice(run_id=run_id)
    model = architecture.train_or_load_model(dataset=data, run_id=run_id)

    # Run for nominal data (which is also used later)
    nom_outputs = _quantify(data.nominal_test_data(), model)
    evals = dict()
    for i, approach in enumerate(APPROACHES):
        _, nom_unc = nom_outputs[i]
        evals[approach] = Evaluation(nom_unc)

    for test_set_name, test_set_x_y in data.non_nominal_as_dict(run_id=run_id).items():
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
    del model
    tf.keras.backend.clear_session()
    gc.collect()


def _uwiz_to_result_name(supervisor_uwiz_name: str) -> str:
    """Converts a supervisor name from uwiz alias to a name used in the result files."""
    if supervisor_uwiz_name == "SM":
        return "Max. Softmax"
    elif supervisor_uwiz_name == "PCS":
        return "PCS"
    elif supervisor_uwiz_name == "SE":
        return "Softmax Entropy"
    elif supervisor_uwiz_name == "VR":
        return "MC-Dropout (VR)"
    elif supervisor_uwiz_name == "pred_entropy":
        return "MC-Dropout (PE)"
    elif supervisor_uwiz_name == "MI":
        return "MC-Dropout (MI)"
    elif supervisor_uwiz_name == "MS":
        return "MC-Dropout (MS)"
    elif supervisor_uwiz_name == "custom::deep_gini":
        return "DeepGini"
    else:
        raise ValueError(f"Unknown supervisor name: {supervisor_uwiz_name}")


def _quantify(data: Tuple[np.ndarray, np.ndarray], model):
    x = data[0]
    if x.dtype == np.uint8 or np.max(x) > 1.01:
        x = model_architectures.preprocess_x(data[0])
    return model.predict_quantified(x=x,
                                    quantifier=APPROACHES,
                                    sample_size=100,
                                    batch_size=32,
                                    as_confidence=False,
                                    verbose=1)
