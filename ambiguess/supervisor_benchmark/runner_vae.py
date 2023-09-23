"""Evaluates the VAE-reconstruction-loss based misclassification predictor."""
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from supervisor_benchmark import test_set, result_files
from supervisor_benchmark.binary_classification_utils import Evaluation
from supervisor_benchmark.model_architectures import _run_folder
from supervisor_benchmark.test_set import TestSets

LATENT_DIM = 2


def run(data: TestSets,
        run_id: int):
    """Perform the evaluation and persist the results."""
    vae = _load_or_train_vae(data, run_id)
    nominal_rl = vae.reconstruction_losses(data.nominal_test_data()[0])
    evaluator = Evaluation(nominal_rl)
    for test_set_name, ts in data.non_nominal_as_dict(run_id=run_id).items():
        ts_rl = vae.reconstruction_losses(ts[0])
        auc_roc = evaluator.auc_roc(ts_rl)
        result_files.register_results(
            dataset_and_split=(data.dataset, test_set_name),
            supervisor="Autoencoder",
            metric="auc_roc",
            value=auc_roc,
            run_id=run_id,
            artifacts_folder=data.artifact_path)


def _load_or_train_vae(dataset: test_set.TestSets,
                       run_id: int):
    model_path = os.path.join(_run_folder(run_id), "vaes", dataset.dataset)
    if os.path.exists(model_path):
        encoder = keras.models.load_model(os.path.join(model_path, "encoder"))
        decoder = keras.models.load_model(os.path.join(model_path, "decoder"))
        return VAE(encoder, decoder)
    else:
        model = _train_vae(dataset)
        model.decoder.save(os.path.join(model_path, "decoder"))
        model.encoder.save(os.path.join(model_path, "encoder"))
    return model


def _train_vae(dataset: test_set.TestSets):
    encoder = _create_encoder()
    decoder = _create_decoder()

    x_train, _, = dataset.mixed_ambiguous_training_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(x_train, epochs=30, batch_size=128)
    return vae


def _create_decoder():
    latent_inputs = keras.Input(shape=(LATENT_DIM,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


def _create_encoder():
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    # docstr-coverage: inherited
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    """Variational Autoencoder, based on https://keras.io/examples/generative/vae/"""

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        """The three metrics of the VAE."""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    # docstr-coverage:inherited
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def reconstruction_losses(self, inputs):
        """Returns reconstruction losses for given inputs."""
        if np.any(inputs > 1.1) or inputs.dtype == np.uint8:
            inputs = np.expand_dims(inputs, -1).astype("float32") / 255.
        z_mean, _, _ = self.encoder(inputs)
        reconstruction = self.decoder(z_mean)
        losses = tf.reduce_sum(
            keras.losses.binary_crossentropy(inputs, reconstruction), axis=(1, 2)
        )
        return losses
