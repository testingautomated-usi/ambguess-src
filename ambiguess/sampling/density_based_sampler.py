""" The AmbiGuess Gradient-Based Diveversity-Aiming Sampling Algorithm.

Note that the term "segment" in this module refers to what we call 'grid cell' in the paper.
"""

import random
from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from aae.autoencoders import R_AAE
from sampling.sample import Sample

# If not constrained by the centers of the distributions for a given axis
# (e.g., upper and lower ends of LS) we use 5 times the standard deviation
# as a the upper and lower bound for that axis
SAMPLING_BADGE_SIZE = 10
STD_SCALER = 5


class DensityMap:
    """Functionality to divide, prioritize and sample from the latent space"""

    def __init__(self, aae: R_AAE, anchors_per_dim: int):
        """
        Utils for density based sampling of a regularized 2-class latent space.

        Calling this constructor initializes the density map and
        is thus computationally expensive.
        :param aae: the regularized autoencoder from which samples should be drawn
        :param anchors_per_dim: this to the power of ndim many anchors will be set.
        """
        self.aae = aae
        self.anchors_per_dim = anchors_per_dim

        assert self.aae.z_dim == 2, "Currently, only 2D density maps are supported" \
                                    "Extension to n-D is possible, but not implemented"

        self.anchors, self.segment_frame = self.__create_anchors()
        self.gradients: np.ndarray = self.__compute_gradient_grid()
        self.anchors_prob_labels = self.__compute_anchors_pred_labels()

    def get_samples(self, ls_coords: np.ndarray) -> List[Sample]:
        """Get the samples defined by the given latent space coordinates."""
        reconstructions = self.aae.decode_(ls_coords).numpy().reshape(-1, 28, 28)
        labels = self.aae.assign_label(ls_coords)
        res = []
        for i, coords in enumerate(ls_coords):
            res.append(Sample(
                coordinates=coords,
                image=reconstructions[i],
                class_1=self.aae.class_1,
                class_2=self.aae.class_2,
                label_1=labels[i][0],
                label_2=labels[i][1],
                num_classes=self.aae.num_labels,
                autoencoder_id=self.aae.random_id
            ))
        return res

    def draw_samples_by_segment_weight(self,
                                       n_samples: int,
                                       max_label_diff: float,
                                       seed: int) -> List[Sample]:
        """Draw samples from the latent space, weighted by the segment gradients."""
        rng = random.Random(seed)

        # Identify segments where the anchor is non-ambiguous
        label_dif = np.abs(self.anchors_prob_labels[:, 0] - self.anchors_prob_labels[:, 1]).flatten()
        ignore_segments = np.where(label_dif > max_label_diff)

        # Sample segments, using the anchor gradients as weights,
        #    except for non-ambigous anchors, where the weight is 0
        segment_indexes = np.arange(self.anchors_per_dim ** 2)
        flat_grad = np.copy(self.gradients.flatten(order='C'))
        flat_grad[ignore_segments] = 0

        def next_coords_badge():
            """Generator to get the next batch of coordinates, given the segment weights"""
            while True:
                chosen_segments = rng.choices(segment_indexes, weights=flat_grad, k=SAMPLING_BADGE_SIZE)
                # The 2D index is the (remainder, quotient) tuple when dividing segment index by anchors_per_dim
                chosen_segments_2d = [divmod(seg, self.anchors_per_dim)[::-1] for seg in chosen_segments]
                chosen_anchors = [
                    (self.anchors[0][seg[0]], self.anchors[1][seg[1]])
                    for seg in chosen_segments_2d
                ]

                # Draw point uniformly withing each segment
                ten_coords = []
                for anchor_x, anchor_y in chosen_anchors:
                    x_min, x_max = anchor_x - self.segment_frame[0] / 2, anchor_x + self.segment_frame[0] / 2
                    y_min, y_max = anchor_y - self.segment_frame[1] / 2, anchor_y + self.segment_frame[1] / 2
                    ten_coords.append((rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)))
                yield np.array(ten_coords)

        # Now we use the sample generator to draw samples,
        #   but we only keep the samples for which the label is ambiguous enough,
        #   i.e., the label difference is below the `max_label_diff` threshold
        ambigous_coords = []
        attempts = 0
        coords_gen = next_coords_badge()
        while len(ambigous_coords) < n_samples:
            coords = coords_gen.__next__()
            labels = self.aae.assign_label(np.array(coords))
            label_dif = np.abs(labels[:, 0] - labels[:, 1])
            selected_idx = label_dif < max_label_diff
            ambigous_coords.extend(coords[selected_idx])
            attempts += coords.shape[0]
            # Eventually, we give up and warn the user
            if attempts > n_samples * 100:
                print("Could not find enough samples"
                      f"for {self.aae.class_1}-{self.aae.class_2} "
                      f"(id = {self.aae.random_id}),"
                      "try increasing the max_label_diff"
                      f"(currently {max_label_diff})")
                break
        # Cutting the selected points down to the number of desired samples
        selected_coords = np.array(ambigous_coords[:min(n_samples, len(ambigous_coords))])
        if len(selected_coords) == 0:
            return []
        return self.get_samples(selected_coords)

    # docstr-coverage:excused `private`
    def __create_anchors(self):
        mu1, mu2, std1, std2 = self.aae.mu_1, self.aae.mu_2, self.aae.std_1, self.aae.std_2

        scaled_std_0 = max(std1[0], std2[0]) * STD_SCALER
        scaled_std_1 = max(std1[1], std2[1]) * STD_SCALER

        bound_axis_0 = (-scaled_std_0, scaled_std_0) if mu1[0] == mu2[0] else (
            mu1[0] + 2 * std1[0], mu2[0] - 2 * std2[0])
        bound_axis_1 = (-scaled_std_1, scaled_std_1) if mu1[1] == mu2[1] else (
            mu1[1] + 2 * std1[1], mu2[1] - 2 * std2[1])

        segment_frame = (
            abs(bound_axis_0[0] - bound_axis_0[1]) / self.anchors_per_dim,
            abs(bound_axis_1[0] - bound_axis_1[1]) / self.anchors_per_dim
        )

        axis_0_anchors = np.linspace(bound_axis_0[0] + segment_frame[0] / 2,
                                     bound_axis_0[1] - segment_frame[0] / 2,
                                     self.anchors_per_dim)
        axis_1_anchors = np.linspace(bound_axis_1[0] + segment_frame[1] / 2,
                                     bound_axis_1[1] - segment_frame[1] / 2,
                                     self.anchors_per_dim)

        return (axis_0_anchors, axis_1_anchors), segment_frame

    # docstr-coverage:excused `private`
    def __gradient_at(self, x: float, y: float) -> float:
        with tf.GradientTape(persistent=True) as tape:
            z = tf.Variable([[x, y]], trainable=True)
            reconstruction = self.aae.decode_(z)
            euclid_dist = tf.reduce_sum(reconstruction ** 2)

        grad = tape.gradient(euclid_dist, z).numpy()
        return np.linalg.norm(grad)

    # docstr-coverage:excused `private`
    def __compute_gradient_grid(self):
        gradients = np.zeros((self.anchors_per_dim, self.anchors_per_dim), dtype=np.float32)

        with tqdm(total=self.anchors_per_dim ** 2, desc="Calculating Gradients") as pbar:
            for i, x in enumerate(self.anchors[0]):
                for j, y in enumerate(self.anchors[1]):
                    gradients[i, j] = self.__gradient_at(x, y)
                    pbar.update(1)

        return gradients

    # docstr-coverage:excused `private`
    def __compute_anchors_pred_labels(self):
        """Predictions for the first label for anchors, allowing to filter non-ambiguous ones"""
        anchors = []
        for i, x in enumerate(self.anchors[0]):
            for j, y in enumerate(self.anchors[1]):
                anchors.append((x, y))

        anchors_labels = self.aae.assign_label(np.array(anchors))
        return anchors_labels
