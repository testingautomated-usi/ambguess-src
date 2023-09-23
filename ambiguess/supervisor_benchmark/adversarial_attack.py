"""Creates adversarial examples for the models of the different runs.

Based on
https://raw.githubusercontent.com/bethgelab/foolbox/master/
    examples/multiple_attacks_pytorch_resnet18.py
"""
import os

import foolbox.attacks as fa
import numpy
import numpy as np
import tensorflow as tf
import tqdm
from foolbox import accuracy, TensorFlowModel, Model
from sklearn import utils

from supervisor_benchmark import model_architectures
from supervisor_benchmark.test_set import TestSets


def attack(dataset: TestSets, run_id: int, model: tf.keras.Sequential):
    """Create the adversarial examples"""
    fmodel: Model = TensorFlowModel(model, bounds=(0, 1), preprocessing=dict())

    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    x, y = dataset.nominal_test_data()

    # Shuffle dataset: Needed as the four attacks are applied in four consectuive blocks of the datasets
    # and multiple runs should corrupt different inputs with different attacks
    x, y = utils.shuffle(x, y, random_state=run_id)

    x = model_architectures.preprocess_x(x)
    y = y.astype(np.int32)
    images, labels = tf.constant(x), tf.constant(y)

    # images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    # apply the attack
    attacks = [
        fa.FGSM(),
        fa.LinfPGD(),
        fa.LinfBasicIterativeAttack(),
        fa.LinfDeepFoolAttack(),
    ]
    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]

    per_attack_chosens = []
    for attack_i, attack in enumerate(attacks):
        print(attack)
        block_size = images.shape[0] // len(attacks)
        block_img = images[attack_i * block_size: (attack_i + 1) * block_size]
        block_lab = labels[attack_i * block_size: (attack_i + 1) * block_size]
        img_badges = np.array_split(block_img, 10)
        lab_badges = np.array_split(block_lab, 10)
        _, clipped_advs, success = [], [], []
        for i in tqdm.tqdm(range(len(img_badges)), desc=f"Applying attack {str(attack.__class__)}"):
            _, _clipped_advs, _success = attack(fmodel,
                                                tf.constant(img_badges[i]),
                                                tf.constant(lab_badges[i]),
                                                epsilons=epsilons)
            clipped_advs.append(_clipped_advs)
            success.append(_success)
        clipped_advs = np.concatenate(clipped_advs)
        success = np.concatenate(success)

        min_successful_eps_idxs = np.argmax(success, axis=0)
        chosen = []
        for sample, eps_idx in enumerate(min_successful_eps_idxs):
            chosen.append(clipped_advs[eps_idx][sample])
        per_attack_chosens.append(chosen)

    adv_data = np.concatenate(per_attack_chosens, axis=0)

    # Measure the accuracy of the adversarial examples on our model
    #   (I'm deliberately not using foolboxes accuracy function here,
    #    as this would rely on fmodel, which may hide some unexpected
    #    preprocessing or similar)
    adv_accuracy = np.sum(np.argmax(model(adv_data), axis=1) == y) / len(y)
    print(f"adversarial accuracy:  {adv_accuracy * 100:.1f} %")

    # Store generated adversarial data
    folder = os.path.join(dataset.artifact_path, "supervisor_benchmark", "runs",
                          str(run_id), "adv_data", dataset.dataset)
    if not os.path.exists(folder):
        os.makedirs(folder)
    numpy.save(os.path.join(folder, "adversarial_x.npy"), adv_data)
    numpy.save(os.path.join(folder, "adversarial_y.npy"), y)
