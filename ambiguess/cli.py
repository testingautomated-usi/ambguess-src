#!/usr/bin/python

"""This is the main entrypoint of the reproduction package.

It allows access to all functionality required to reproduce our results,
as a single `typer`-based command-line application."""
import os
import random
import shutil
import warnings
from enum import Enum
from random import Random
from time import sleep
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import tqdm
import typer
import uncertainty_wizard as uwiz
from matplotlib import pyplot as plt

import supervisor_benchmark
from aae.model_folder import get_r_aaes_in_model_folder
from assessment import eval_runner, eval_model
from packing.csv_exports import save_as_csv
from packing.draw_det_labels import draw_det_labels
from packing.numpy_arrays import to_numpy_arrays, save_numpy_arrays, load_numpy_arrays
from sampling import sample_storage
from sampling.density_based_sampler import DensityMap
from sampling.sample import Sample
from supervisor_benchmark import test_set, runner_softmax_and_dropout, result_files, runner_deep_ensemble, \
    runner_surprise, adversarial_attack, runner_vae, runner_dissector, model_architectures
from supervisor_benchmark.model_architectures import architecture_choice
from supervisor_benchmark.result_files import plot_aggregate

uwiz.models.ensemble_utils.DynamicGpuGrowthContextManager.enable_dynamic_gpu_growth()

app = typer.Typer()


@app.command(hidden=True)
def train(dataset: str = typer.Argument(..., help="(mnist/fmnist)"),
          first_class: int = typer.Argument(..., help="Class-Index of the first class (0-9)"),
          second_class: int = typer.Argument(..., help="Class-Index of the first class (0-9)"),
          num_tries: int = typer.Argument(..., help="Number of tries to find a good model per class-pair")):
    """Trains a new 2-class regularized adversarial autoencoder.

    **Attention:** Training is a random process, hence the results will be different
    from the ones we observed. We don't recommend you run this command for reproduction.
    All models are available in our artifacts folder"""
    from aae.autoencoders import R_AAE

    dataset = _check_valid_dataset(dataset)
    assert first_class < second_class, "First class must be smaller than second class"
    assert 0 <= first_class < 10, "First class must be between 0 and 9"
    assert 0 <= second_class < 10, "Second class must be between 0 and 9"

    for _ in range(num_tries):
        raae = R_AAE(dim=(128, 2, 784),
                     img_shape=(28, 28, 1),
                     num_labels=10,
                     learning_rate=0.001,
                     batch_size=1000,
                     num_epochs=80,
                     labels=(first_class, second_class))

        print(f"Training model for class-pair {first_class}-{second_class} [id = {raae.random_id}]")
        if dataset == "mnist":
            (x_train, y_train), (_, _) = _load_mnist()
        elif dataset == "fmnist":
            (x_train, y_train), (_, _) = _load_fmnist()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        dc_fake_real_acc = raae.train(x_train, y_train)
        if not 0.4 < dc_fake_real_acc < 0.6:
            print(f"Discriminator fake/real accuracy ( {dc_fake_real_acc} )  is insufficient. "
                  f"Discarded Autoencoder {first_class}-{second_class} [id = {raae.random_id}].")
            continue
        dc_class_accuracy = raae.eval_class_accuracy(x_train, y_train)
        if dc_class_accuracy < 0.9:
            print(f"2-Class accuracy: {dc_class_accuracy} is insufficient. "
                  f"Discarded Autoencoder {first_class}-{second_class} [id = {raae.random_id}].")
            continue
        raae.save_raae_with_random_id(f"/artifacts/{dataset}/models/")


def _check_valid_dataset(dataset):
    dataset = dataset.lower().strip()
    assert dataset in ["mnist", "fmnist"], "Only MNIST and FMNIST are supported"
    return dataset


def _check_valid_split(split: str):
    split = split.lower().strip()
    assert split in ["training", "test"], "Only MNIST and FMNIST are supported"
    return split


def _ensemble_id_to_class_labels(model_id: int):
    count = 0
    for first_class in range(9):
        for second_class in range(first_class + 1, 10):
            if count == model_id:
                return first_class, second_class
            count += 1


def _mnist_train_ensemble(model_id):
    first_class, second_class = _ensemble_id_to_class_labels(model_id)
    train(dataset="mnist", first_class=first_class, second_class=second_class, num_tries=20)


def _load_mnist(*args, **kwargs):
    """Loads the MNIST dataset, and caches it on disk."""
    import tensorflow as tf
    return tf.keras.datasets.mnist.load_data()


def _fmnist_train_ensemble(model_id):
    first_class, second_class = _ensemble_id_to_class_labels(model_id)
    train(dataset="fmnist", first_class=first_class, second_class=second_class, num_tries=20)


def _load_fmnist(*args, **kwargs):
    """Loads the MNIST dataset, and caches it on disk."""
    import tensorflow as tf
    return tf.keras.datasets.fashion_mnist.load_data()


@app.command()
def train_all(dataset: str = typer.Argument(..., help="(mnist/fmnist)"),
              processes: int = typer.Argument(..., help="Number of processes to use")):
    """Trains all 2-class regularized adversarial autoencoders.

    Attention: This will take a while, especially if you're not running on a gpu!"""
    if processes > 1:
        typer.echo(f"Note: Running multiple processes in parallel"
                   f" will make the stdout logs quite messy."
                   f"\n"
                   f"Training will start in 5 seconds.")
        sleep(5)

    from uncertainty_wizard.models import LazyEnsemble
    import math

    dataset = _check_valid_dataset(dataset)
    # We leverage uncertainty wizards ensemble utilities to handle tf gpu config
    #   and parallelization
    runner = LazyEnsemble(model_save_path="/tmp/train_all_ensemble/",
                          num_models=math.comb(10, 2),
                          default_num_processes=processes)

    # Ensemble task. Derives classes from ensemble id and trains the corresponding model
    if dataset == "mnist":
        # Pre-Fetch the MNIST dataset before parallelizing tasks
        runner.run_model_free(task=_load_mnist, num_times=1)
        # Run Training Sessions in parallel
        runner.run_model_free(task=_mnist_train_ensemble)
    elif dataset == "fmnist":
        # Pre-Fetch the MNIST dataset before parallelizing tasks
        runner.run_model_free(task=_load_fmnist, num_times=1)
        # Run Training Sessions in parallel
        runner.run_model_free(task=_fmnist_train_ensemble)
    else:
        raise ValueError("Only MNIST is supported")


TYPER_MAX_LABEL_DIFF_ARGUMENT = typer.Argument(...,
                                               help="Maximum label difference in considered segment anchors",
                                               min=0.1, max=1.0)

TYPER_SEGMENTS_PER_DIM_ARG = typer.Argument(50,
                                            help="Number of segments per dimension. "
                                                 "The total #segments is `ndim^num_segment_per_dim`"
                                            , min=2)

TYPER_SAMPLES_PER_AE = typer.Argument(..., help="Number of samples to generate per autoencoder", min=1)

TYPER_SAMPLING_REPLACE = typer.Argument(False, help="Replace Previous Sampling."
                                                    "If true, previously generated samples for this split"
                                                    "will be deleted."
                                                    "If false, aae from which samples are already stored will "
                                                    "be skipped.")


@app.command()
def sample_points(dataset: str = typer.Argument(..., help="(mnist/fmnist)"),
                  split: str = typer.Argument(..., help="Name of the split to sample (e.g. 'test')"),
                  max_label_diff: float = TYPER_MAX_LABEL_DIFF_ARGUMENT,
                  samples_per_ae: int = TYPER_SAMPLES_PER_AE,
                  seed: int = typer.Argument(..., help="Random seed"),
                  segments_per_dim: int = TYPER_SEGMENTS_PER_DIM_ARG,
                  replace: Optional[bool] = TYPER_SAMPLING_REPLACE):
    """Creates and stores raw samples a 2-class regularized adversarial autoencoder.

    Attention: This command does not create a new dataset, only new raw samples.
    To create a new dataset, use the `build-dataset` command after running `sample-points`."""
    from aae.autoencoders import R_AAE
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    split_folder = f"/artifacts/{dataset}/{split}"
    model_folder = f"/artifacts/{dataset}/models"
    if replace:
        shutil.rmtree(split_folder, ignore_errors=True)
    os.makedirs(os.path.join(split_folder, "raw"), exist_ok=True)

    aes_defs = get_r_aaes_in_model_folder(model_folder=model_folder)
    remaining_aae_defs = sample_storage.unsampelled_aes(model_folder=model_folder, split_folder=split_folder)

    typer.echo(f"Samples will be generated for {len(remaining_aae_defs)} models "
               f"({samples_per_ae} per model)."
               f"Total number of found AAEs is {len(aes_defs)}, "
               f"{len(aes_defs) - len(remaining_aae_defs)} models will be skipped "
               f"as samples have already been generated.")

    for i, (c1, c2, ae_id, filename) in enumerate(remaining_aae_defs):
        typer.echo(f"Sampling data for {c1}-{c2} [id = {ae_id}, "
                   f"progress {i + 1}/{len(remaining_aae_defs)}]")
        raae = R_AAE(dim=(128, 2, 784),
                     img_shape=(28, 28, 1),
                     num_labels=10,
                     learning_rate=0.001,
                     batch_size=1000,
                     num_epochs=80,
                     labels=(c1, c2))

        raae.random_id = ae_id
        raae.load_weights(f"/artifacts/{dataset}/models/{filename}")

        dens_sampler = DensityMap(raae, segments_per_dim)

        this_ae_samples = dens_sampler.draw_samples_by_segment_weight(
            n_samples=samples_per_ae,
            max_label_diff=max_label_diff,
            seed=seed + ae_id
        )

        sample_storage.store_samples_for_ae(
            r_aae=raae, split_folder=split_folder, samples=this_ae_samples
        )


@app.command()
def build_dataset(dataset: str = typer.Argument(..., help="(mnist/fmnist)"),
                  split: str = typer.Argument(..., help="Name of the split to sample (e.g. 'test')"),
                  num_samples: int = typer.Argument(..., help="Number of samples in split", min=1),
                  seed: Optional[int] = typer.Argument(0, help="Random seed")):
    """Build & Package (csv & numpy) the final dataset from the sampled data (persistet on fs)."""
    dataset = _check_valid_dataset(dataset)
    split = _check_valid_split(split)

    samples: List[Sample] = sample_storage.load_all_samples_for_split(
        os.path.join("/artifacts", dataset, split)
    )
    rng = Random(seed)
    rng.shuffle(samples)

    selected_samples = samples[:num_samples]

    imgs, prob_labels = to_numpy_arrays(selected_samples)
    det_labels = draw_det_labels(selected_samples, seed=seed)

    save_numpy_arrays(images=imgs,
                      prob_labels=prob_labels,
                      det_labels=det_labels,
                      datasets_folder=os.path.join("/artifacts", dataset, split),
                      dataset=dataset,
                      split=split)

    save_as_csv(samples=selected_samples,
                imgs=imgs,
                det_labels=det_labels,
                datasets_folder=os.path.join("/artifacts", dataset, split),
                dataset=dataset,
                split=split)


class SamplingStrategy(str, Enum):
    """Whether to sample specific indexes, or random points from the dataset."""
    index_list = "index_list"
    random = "random"


@app.command()
def draw_examples(
        dataset: str,
        split: str,
        num_samples: int = typer.Argument(..., help="Number of samples to draw", min=1),
        strategy: SamplingStrategy = typer.Option(SamplingStrategy.random, case_sensitive=False),
        clear_examples: bool = typer.Argument(False, help="Clear examples folder before drawing"),
        seed: int = typer.Option(0, help="Random seed. Ignored for index-list sampling."),
        indexes: Optional[List[int]] = typer.Option(None, help="Indexes to draw. Ignored for random sampling.")):
    """Visualizes samples from the pre-built datasets (created by `build_dataset`) as png."""
    dataset = _check_valid_dataset(dataset)
    split = _check_valid_split(split)

    arrs = load_numpy_arrays(datasets_folder=os.path.join("/artifacts", dataset, split),
                             dataset=dataset, split=split)
    imgs, det_labels, pred_labels = arrs

    folder = os.path.join("/artifacts", dataset, split, "examples")

    if clear_examples:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

    if strategy == SamplingStrategy.random:
        rng = Random(seed)
        indexes = rng.sample(list(range(len(imgs))), num_samples)
    else:
        assert indexes is not None, "--indexes must be provided for index-list sampling." \
                                    "See --help for more information."

    for i in tqdm.tqdm(indexes, desc="Drawing examples"):
        real_class = det_labels[i]
        real_class_likelihood = pred_labels[i][real_class]
        pred_copy = pred_labels[i].copy()
        pred_copy[real_class] = 0
        other_class = np.argmax(pred_copy)
        other_class_likelihood = pred_copy[other_class]

        if real_class > other_class:
            p_segment = f"p_{real_class}_{real_class_likelihood:.2f}_p_{other_class}_{other_class_likelihood:.2f}"
        else:
            p_segment = f"p_{other_class}_{other_class_likelihood:.2f}_p_{real_class}_{real_class_likelihood:.2f}"

        img_name = f"{i}_label_{det_labels[i]}_{p_segment}.png"
        img_path = os.path.join(folder, img_name)

        # Invert colors and save image
        inverted_img = 255 - imgs[i]
        plt.imsave(img_path, inverted_img, cmap="gray")


def _mukh_det_labels(prob_labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Draws deterministic labels for the samples."""
    rng = random.Random(seed)
    labels = []
    labels_candidates = list(range(10))
    for i in range(len(prob_labels)):
        labels.append(rng.choices(labels_candidates, weights=prob_labels[i]))
    return np.array(labels).flatten()


def _load_mukhoti(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mukh_file_loc = "../resources"
    if split == "training":
        images = np.load(os.path.join(mukh_file_loc, "mukh_x_train.npy"))
        prob_labels = np.load(os.path.join(mukh_file_loc, "mukh_y_train.npy"))
        det_labels = _mukh_det_labels(prob_labels, seed=0)
    elif split == "test":
        images = np.load(os.path.join(mukh_file_loc, "mukh_x_test.npy"))
        prob_labels = np.load(os.path.join(mukh_file_loc, "mukh_y_test.npy"))
        det_labels = _mukh_det_labels(prob_labels, seed=0)
    else:
        raise ValueError(f"Unknown split: {split}")
    return images, det_labels, prob_labels


def _eval_ambiguity_for_ds(dataset: str,
                           clean: bool,
                           result_table: pd.DataFrame,
                           run_id: int):
    if dataset == "mnist" or dataset == "mukhoti":
        (x_train, y_train), (x_test, y_test) = _load_mnist(dataset)
    elif dataset == "fmnist":
        (x_train, y_train), (x_test, y_test) = _load_fmnist(dataset)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    if not clean:
        folder = os.path.join("/artifacts", dataset, "training")

        if dataset == "mnist" or dataset == "fmnist":
            ambi_x, ambi_y, _, = load_numpy_arrays(datasets_folder=folder,
                                                   dataset=dataset,
                                                   split="training")
        elif dataset == "mukhoti":
            ambi_x, ambi_y, _, = _load_mukhoti(split="training")
        else:
            raise ValueError(f"Unknown dataset {dataset}")

        x_train = np.concatenate((x_train, ambi_x), axis=0)
        y_train = np.concatenate((y_train, ambi_y), axis=0)

    model = eval_model.train_or_load_model(
        run_id=run_id,
        dataset=dataset,
        x_train=x_train,
        y_train=y_train,
        clean=clean,
    )

    if dataset == "mnist" or dataset == "fmnist":
        ambi_test_data = load_numpy_arrays(datasets_folder=os.path.join("/artifacts", dataset, "test"),
                                           dataset=dataset, split="test")
    elif dataset == "mukhoti":
        ambi_test_data = _load_mukhoti(split="test")
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    eval_runner.evaluate(model=model,
                         dataset=dataset,
                         clean=clean,
                         nominal_test_data=(x_test, y_test),
                         ambiguous_test_data=ambi_test_data,
                         result_table=result_table)


TYPER_ARG_AMB_RUNS = typer.Option(..., help="Run number(s) (valid values: 0,1,2,3)")


@app.command()
def evaluate_ambiguity(run: List[int] = TYPER_ARG_AMB_RUNS):
    """Evaluate the ambiguity of the dataset."""
    # If we sort the table to get rid of this warning,
    # The latex table will not be sorted as we like it best.
    # As the table is super small, we can ignore the performance problem.
    warnings.filterwarnings("ignore", ".*past lexsort depth may impact performance.*")
    # Noted (not relevant for us):
    warnings.filterwarnings("ignore", ".*to_latex` is expected to utilise.*")

    result_folder = os.path.join("/artifacts", "ambiguity_results")
    latex_paper_result_folder = os.path.join(result_folder, "latex_tables")
    if not os.path.exists(latex_paper_result_folder):
        os.makedirs(latex_paper_result_folder)

    if run == (-1,):
        typer.echo("Skipping re-evaluation. Simply generating plots from previous runs.")
        run = []

    for run_id in run:
        typer.echo(f"Evaluating ambiguity for run {run_id}. This will take a while...")
        if run_id > 1:
            typer.echo("(For run id 2 and 3 this takes even longer than for lower runs, as the models are much larger.")
        res_table = eval_runner.create_empty_result_table()
        _eval_ambiguity_for_ds("mukhoti", clean=False, result_table=res_table, run_id=run_id)
        _eval_ambiguity_for_ds("mukhoti", clean=True, result_table=res_table, run_id=run_id)
        _eval_ambiguity_for_ds("mnist", clean=False, result_table=res_table, run_id=run_id)
        _eval_ambiguity_for_ds("mnist", clean=True, result_table=res_table, run_id=run_id)
        _eval_ambiguity_for_ds("fmnist", clean=False, result_table=res_table, run_id=run_id)
        _eval_ambiguity_for_ds("fmnist", clean=True, result_table=res_table, run_id=run_id)
        run_results_folder = os.path.join(result_folder, "runs", f"run_{run_id}")
        if not os.path.exists(run_results_folder):
            os.makedirs(run_results_folder)

        eval_runner.print_full_results_latex(res_table, latex_paper_result_folder, run_id)
        eval_runner.print_results_as_latex_table(res_table, run_results_folder)
        eval_runner.write_full_csv_results(res_table, run_results_folder)
        # save pandas dataframe as pickle
        res_table.to_pickle(os.path.join(run_results_folder, "results.pkl"))

    # Create overall average
    typer.echo("Creating overall average tables (averaging the four model architectures)")
    res_tables = []
    for run_id in range(4):
        run_results_folder = os.path.join(result_folder, "runs", f"run_{run_id}")
        res_tables.append(pd.read_pickle(os.path.join(run_results_folder, "results.pkl")))
    overall_res_table = pd.concat(res_tables).groupby(level=(0, 1, 2)).mean()
    eval_runner.print_full_results_latex(overall_res_table, latex_paper_result_folder, run=None)
    eval_runner.print_results_as_latex_table(overall_res_table, result_folder)
    eval_runner.write_full_csv_results(overall_res_table, result_folder)


class SupervisorGroup(str, Enum):
    """Misclassification Predictors"""
    all = "all"
    softmax_and_dropout = "softmax_and_dropout"
    deep_ensemble = "deep_ensemble"
    surprises = "surprises"
    vae = "vae"
    dissector = "dissector"
    none = "none"  # Only generating artifacts from previous, raw results


TYPER_ARG_SUPERVISOR_GROUP = typer.Argument(..., help="Which group of supervisors to evaluate")
TYPER_ARG_SUPERVISOR_RUNS = typer.Option(..., help="Run number(s) (valid values: 0,...,19)")


@app.command()
def evaluate_supervisor(group: SupervisorGroup = TYPER_ARG_SUPERVISOR_GROUP,
                        run: List[int] = TYPER_ARG_SUPERVISOR_RUNS):
    """Evaluate the specified supervisors and generated the latex and csv output tables.

    Specifying one or multiple run numbers (using multiple --run flags)
    allows to select the run (and thus model architecture) to evaluate."""
    # Noted (not relevant for us):
    warnings.filterwarnings("ignore", ".*to_latex` is expected to utilise.*")

    for dataset in ["mnist", "fmnist"]:
        test_sets = supervisor_benchmark.test_set.get(dataset, '/artifacts')
        for _run in run:
            # Write status to stdout
            if group != SupervisorGroup.none:
                typer.echo(f"Evaluating {group} for {dataset}  and run {_run} "
                           f"(architecture: {type(architecture_choice(_run)).__name__})")
            # Perform work
            if group == SupervisorGroup.softmax_and_dropout or group == SupervisorGroup.all:
                runner_softmax_and_dropout.run(test_sets, run_id=_run)
            if group == SupervisorGroup.deep_ensemble or group == SupervisorGroup.all:
                runner_deep_ensemble.run(test_sets, run_id=_run)
            if group == SupervisorGroup.surprises or group == SupervisorGroup.all:
                runner_surprise.run(test_sets, run_id=_run)
            if group == SupervisorGroup.vae or group == SupervisorGroup.all:
                runner_vae.run(test_sets, run_id=_run)
            if group == SupervisorGroup.dissector or group == SupervisorGroup.all:
                runner_dissector.run(test_sets, run_id=_run)

    # Generate per-run results
    for _run in run:
        result_files.plot(artifacts_folder='/artifacts', run_id=_run)

    # Generate averaged results
    plot_aggregate(artifacts_folder='/artifacts')

    typer.echo(f"""
🎉🎉🎉
Completed evaluation of supervisors '{group.name}' for run(s) {run}."
Artifact outputs (.tex and .csv) tables were updated.  
🎉🎉🎉    """)


@app.command(hidden=True)
def create_adversarial_data(run: List[int] = TYPER_ARG_SUPERVISOR_RUNS):
    """Manually re-create the adversarial data for a given run.

    This is not needed in principle, as the adversarial data is created
     automatically when the model is trained, but it allows to re-create
     the data during development.
    """
    for dataset in ["mnist", "fmnist"]:
        for _run in run:
            test_sets = supervisor_benchmark.test_set.get(dataset, '/artifacts')
            architecture = model_architectures.architecture_choice(_run)
            model = architecture.train_or_load_model(test_sets, _run)
            adversarial_attack.attack(test_sets, run_id=_run, model=model._inner_sequential)


if __name__ == "__main__":
    # Launch Typer-Generated CLI
    app()
