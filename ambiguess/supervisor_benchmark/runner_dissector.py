"""Evaluates Dissector"""
import os
from typing import Tuple, List

import numpy as np
import tensorflow as tf
import tqdm
import uncertainty_wizard as uwiz

from supervisor_benchmark import model_architectures
from supervisor_benchmark import result_files
from supervisor_benchmark.binary_classification_utils import Evaluation
from supervisor_benchmark.model_architectures import _run_folder
from supervisor_benchmark.test_set import TestSets


def _run_evaluation(dataset, run_id, base_model):
    submodel_dir = os.path.join(_run_folder(run_id), "dissector", dataset.dataset)
    architecture = model_architectures.architecture_choice(run_id=run_id)
    pv_nominal = _get_pv_scores(architecture=architecture,
                                base_model=base_model,
                                x=dataset.nominal_test_data()[0],
                                submodel_dir=submodel_dir)
    evaluator = Evaluation(pv_nominal)
    for test_set_name, ts in dataset.non_nominal_as_dict(run_id=run_id).items():
        ts_pv = _get_pv_scores(architecture=architecture,
                               base_model=base_model, x=ts[0],
                               submodel_dir=submodel_dir)
        auc_roc = evaluator.auc_roc(ts_pv)
        result_files.register_results(
            dataset_and_split=(dataset.dataset, test_set_name),
            supervisor="Dissector",
            metric="auc_roc",
            value=auc_roc,
            run_id=run_id,
            artifacts_folder=dataset.artifact_path)


def _get_pv_scores(
        architecture: model_architectures.ModelArchitecture,
        base_model: uwiz.models.StochasticSequential,
        x: np.ndarray, submodel_dir) -> np.ndarray:
    with tf.device("/cpu:0"):
        dissector_layers = architecture.get_dissector_layers()
        ats, pred = architecture.get_activations_and_pred(model=base_model,
                                                         layers=dissector_layers,
                                                         x=x)
        sv_scores = []
        for i, layer in enumerate(sorted(dissector_layers)):
            submodel = tf.keras.models.load_model(_submodel_path(submodel_dir, layer))
            submode_prob_pred = submodel(ats[i])
            sv_scores.append(_sv_scores(submodel_prob_pred=submode_prob_pred, model_pred=pred))

        # we use Dissector-linear (y = x), i.e., weights equal to the submodel index,
        #    as suggested in the dissector paper for mnist
        weighted_sv = [sv * i for i, sv in enumerate(sv_scores)]
        pv_scores = np.sum(np.asarray(weighted_sv), axis=0) / np.sum(range(len(dissector_layers)))

        # we take negative pv scores, as our auc-roc expects uncertainty, not confidence
        return -pv_scores


def _pred_and_runner_up(arr: np.ndarray) -> Tuple[int, int]:
    pred = np.argmax(arr)
    pred_copy = np.copy(arr)
    pred_copy[pred] = 0
    runner_up = np.argmax(pred_copy)
    return pred, runner_up


def _sv_scores(submodel_prob_pred: np.ndarray, model_pred: np.ndarray) -> np.ndarray:
    svs = []
    # This could be much faster if vectorized
    for i in tqdm.tqdm(range(model_pred.shape[0]), desc="Calculating SV scores"):
        pred, runner_up = _pred_and_runner_up(submodel_prob_pred[i])
        l_x = submodel_prob_pred[i][model_pred[i]]
        l_sh = submodel_prob_pred[i][runner_up]
        l_h = submodel_prob_pred[i][pred]
        if pred == model_pred[i]:
            svs.append(l_x / (l_x + l_sh))
        else:
            svs.append(1 - (l_h / (l_h + l_x)))
    return np.array(svs)


def run(data: TestSets,
        run_id: int):
    """Perform the evaluation and persist the results."""
    architecture = model_architectures.architecture_choice(run_id=run_id)
    with tf.device("/cpu:0"):
        uwiz_model = architecture.train_or_load_model(dataset=data, run_id=run_id)
    _train_submodels_if_needed(dataset=data, run_id=run_id, base_model=uwiz_model)
    with tf.device("/cpu:0"):
        _run_evaluation(dataset=data, run_id=run_id, base_model=uwiz_model)


def _submodel_path(submodel_dir: str, layer: int) -> str:
    return os.path.join(submodel_dir, f"layer_{layer}")


def _train_submodels_if_needed(dataset: TestSets,
                               run_id: int,
                               base_model: uwiz.models.StochasticSequential) -> None:
    architecture = model_architectures.architecture_choice(run_id=run_id)
    submodel_dir = os.path.join(_run_folder(run_id), "dissector", dataset.dataset)
    dissector_layers = architecture.get_dissector_layers()
    submodel_dirs_missing = [not os.path.exists(_submodel_path(submodel_dir, layer))
                             for layer in dissector_layers]
    if any(submodel_dirs_missing):
        if not os.path.exists(submodel_dir):
            os.makedirs(submodel_dir)
        _train_submodels(dataset=dataset,
                         base_model=base_model,
                         submodel_dir=submodel_dir,
                         dissector_layers=dissector_layers,
                         architecture=architecture)


def _train_submodels(dataset: TestSets,
                     base_model: tf.keras.Sequential,
                     submodel_dir: str,
                     dissector_layers: List[int],
                     architecture: model_architectures.ModelArchitecture) -> None:
    train_x, train_y = dataset.mixed_ambiguous_training_data()
    train_y = tf.keras.utils.to_categorical(train_y)
    with tf.device("/cpu:0"):
        train_ats, _ = architecture.get_activations_and_pred(model=base_model,
                                                            layers=dissector_layers,
                                                            x=train_x)
        for i, ats in enumerate(train_ats):
            input_shape = list(ats.shape)
            input_shape = input_shape[1:]
            submodel = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=input_shape),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(10, activation="softmax"),
                ]
            )
            submodel.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                             optimizer="adam",
                             metrics=["accuracy"])
            submodel.fit(ats, train_y, batch_size=32, epochs=30, validation_split=0.1,
                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)])

            submodel.save(_submodel_path(submodel_dir, dissector_layers[i]))
