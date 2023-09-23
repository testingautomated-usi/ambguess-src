"""Primary class to handle the rAAEs in AmbiGuess."""
import math
import os.path
import random
from abc import abstractmethod
from typing import Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from aae.decoder import Decoder
from aae.discriminator import Discriminator
from aae.encoder import Encoder
from aae.loss import RegularizedAaeLoss


class AE(tf.keras.Model):
    """Base autoencoder class (subtyped by AAE and rAAE)."""

    def __init__(self, dim, img_shape, **kwargs):
        """
        Autoencoder Interface
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        """
        self.h_dim = dim[0]
        self.z_dim = dim[1]
        self.image_size = dim[2]
        self.original_size = int(math.sqrt(self.image_size))
        super(AE, self).__init__(**kwargs)
        self.preprocessing = lambda x: x / 255.
        self.postprocessing = lambda x: x * 255
        self.preformat = lambda x: x.reshape((len(x),
                                              np.prod(x.shape[1:])))
        """
        self.postformat = lambda x: x.reshape((len(x),
                                               self.original_size,
                                               self.original_size))
        """

        self.postformat = lambda x: x.reshape((len(x), img_shape[0], img_shape[1], img_shape[2]))
        self.concact_index = 0  # Index to concat batch size
        self.img_shape = img_shape
        self.random_id = random.randint(0, 100000)

    @abstractmethod
    def encode(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        pass

    @abstractmethod
    def decode(self, z):
        """
        Decode with activation function sigmoid
        :param z: latent space
        :return: output model
        """
        pass

    @abstractmethod
    def call_(self, inputs, training=None, mask=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return
        """
        pass

    def call(self, inputs, training=False, mask=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :use_batch: use batch or not
        :batch_size: size of the batch
        :return
        """
        outputs = self.call_(inputs, training=training, mask=mask)
        return outputs

    @abstractmethod
    def load_weights_model(self, list_path):
        """
        Load the weights of the model
        """
        pass

    @abstractmethod
    def save_weights_model(self, list_path):
        """
        Save the weights of the model
        """
        pass


class AAE(AE):
    """Standard (non regularized) adversarial autoencoder"""

    def __init__(self, dim, num_labels, img_shape, dropout=0.5, regularize=True, **kwargs):
        """
        Wrapper for the Adversal AutoEncoder (AAE)
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param num_labels: Number of labels for regularization
        :param dropout: Noise dropout [0,1]
        """
        # Define the hyperparameters
        super(AAE, self).__init__(dim, img_shape, **kwargs)
        self.num_labels = num_labels
        self.dropout = dropout
        # Initialize the models
        self.encode_ = Encoder([self.h_dim, self.z_dim],
                               self.dropout)
        self.decode_ = Decoder([self.h_dim, self.image_size],
                               self.dropout)
        self.discriminator_ = Discriminator([self.h_dim],
                                            self.dropout)
        # Build the models
        self.encode_.build(input_shape=(4, self.image_size))
        self.decode_.build(input_shape=(4, self.z_dim))
        if regularize:
            self.discriminator_.build(input_shape=(4, self.z_dim + \
                                                   self.num_labels))
        else:
            self.discriminator_.build(input_shape=(4, self.z_dim))

    def load_weights_model(self, list_path):
        """
        Load the weights of the model
        :param path_encoder: path of the encoder weights (.h5)
        :param path_decoder: path of the decoder weights (.h5)
        :param path_discriminator: path of the discriminator weights (.h5)
        :return:
        """
        [path_encoder, path_decoder, path_discriminator] = list_path
        self.encode_.load_weights(path_encoder)
        self.decode_.load_weights(path_decoder)
        self.discriminator_.load_weights(path_discriminator)

    def save_weights_model(self, list_path):
        """
        Save the weights of the model
        """
        [path_encoder, path_decoder, path_discriminator] = list_path
        self.encode_.save_weights(path_encoder)
        self.decode_.save_weights(path_decoder)
        self.discriminator_.save_weights(path_discriminator)

    def encode(self, x):
        """
        Encode input
        :param x: input
        :return: input in the latent space
        """
        return self.encode_(x)

    def discriminator(self, x):
        """
        Discriminator input
        :param x: input
        :return: input in the latent space
        """
        return self.discriminator_(x)

    def decode(self, z):
        """
        Decode with activation function sigmoid
        :param z: latent space
        :return: output model
        """
        return self.decode_(z)

    def call_(self, inputs, training=None, mask=None, index=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return
        """
        return self.decode_(self.encode_(inputs))


class R_AAE(AAE):
    """Regularized Adversarial Autoencoder

    Implementation partially based on MIT-Licensed
    https://github.com/Mind-the-Pineapple/adversarial-autoencoder"""

    def __init__(self, dim, img_shape, num_epochs, batch_size,
                 learning_rate,
                 # db,
                 num_labels,
                 labels: Tuple[int, int],
                 train_buf=60000, dropout=0.1, **kwargs):
        """
        Wrapper for the Adversal AutoEncoder (AAE)
        :param dim: hyperparameters of the model [h_dim, z_dim, real_dim]
        :param num_labels: Number of labels for regularization
        :param dropout: Noise dropout [0,1]
        :param value_1: label 1
        :param value_2: label 2
        """
        super(R_AAE, self).__init__(dim=dim,
                                    num_labels=num_labels,
                                    img_shape=img_shape,
                                    dropout=dropout, **kwargs)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_buf = train_buf
        self.build(input_shape=(4, self.image_size))
        self.class_1 = labels[0]
        self.class_2 = labels[1]

        self._init_fake_distr()

    @staticmethod
    def create_fake(class_1, class_2, size, z_dim,
                    mu_1, mu_2, std_1, std_2):
        """
        Create two fake independent gaussian distributions
        :param value1: label first distribution
        :param value2: label second distribution
        :param size: Size of sample
        :param z_dim: Size of the latent space
        :param mu_1: mean first distribution
        :param mu_2: mean second distribution
        :param std_1: standard desviation first distribution
        :param std_2: standart desviation second distribution
        :return: data and labels
        """
        part_1 = int(size / 2)
        part_2 = size - part_1
        a = tf.random.normal([part_1, z_dim], mean=mu_1, stddev=std_1)
        b = tf.random.normal([part_2, z_dim], mean=mu_2, stddev=std_2)
        labels = tf.concat([np.ones(part_1) * class_1,
                            np.ones(part_2) * class_2], -1)
        data = tf.concat([a, b], 0)
        d = list(zip(data, labels))
        random.shuffle(d)
        data, labels = list(zip(*d))
        data = tf.convert_to_tensor(data)
        labels = tf.convert_to_tensor(labels)
        return data, labels

    # docstr-coverage:inherited
    def train_step(self, batch_x, batch_y):
        # Autoencoder
        with tf.GradientTape() as ae_tape:
            encoder_output = self.encode_(batch_x, training=True)
            decoder_output = self.decode_(encoder_output, training=True)
            # Autoencoder loss
            ae_loss = self.loss.autoencoder_loss(batch_x, decoder_output)
        ae_grads = ae_tape.gradient(ae_loss,
                                    self.encode_.trainable_variables + \
                                    self.decode_.trainable_variables)
        self.ae_optimizer.apply_gradients(zip(ae_grads,
                                              self.encode_.trainable_variables + \
                                              self.decode_.trainable_variables))
        # Discriminator
        with tf.GradientTape() as dc_tape:
            ################
            data, labels = R_AAE.create_fake(self.class_1, self.class_2, batch_x.shape[0],
                                             self.z_dim, self.mu_1, self.mu_2, self.std_1, self.std_2)
            real_distribution = self._disc_input(data, labels)
            #################

            encoder_output = self.encode_(batch_x, training=True)

            dc_real = self.discriminator_(real_distribution, training=True)

            ################
            batch_y = tf.one_hot(tf.dtypes.cast(batch_y, tf.int32), 10)
            encoder_output = tf.concat([encoder_output, batch_y], -1)
            ################
            dc_fake = self.discriminator_(encoder_output, training=True)

            # Discriminator Loss
            dc_loss = self.loss.discriminator_loss(dc_real, dc_fake)
            # Discriminator Acc
            dc_acc = self.loss.accuracy(tf.concat([tf.ones_like(dc_real),
                                                   tf.zeros_like(dc_fake)], axis=0),
                                        tf.concat([dc_real, dc_fake], axis=0))

        dc_grads = dc_tape.gradient(dc_loss, self.discriminator_.trainable_variables)
        self.dc_optimizer.apply_gradients(zip(dc_grads,
                                              self.discriminator_.trainable_variables))
        # Generator (Encoder)
        with tf.GradientTape() as gen_tape:
            encoder_output = self.encode_(batch_x, training=True)
            encoder_output = tf.concat([encoder_output, batch_y], -1)
            dc_fake = self.discriminator_(encoder_output, training=True)

            # Generator loss
            gen_loss = self.loss.generator_loss(dc_fake)

        gen_grads = gen_tape.gradient(gen_loss, self.encode_.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.encode_.trainable_variables))

        return ae_loss, dc_loss, dc_acc, gen_loss

    def _filter_dataset(self, x, y):
        idxs = np.logical_or(y == self.class_1, y == self.class_2)
        return x[idxs], y[idxs]

    def train(self, x_train, y_train, verbose=False):
        """
        Start training
        :param x_train: data training
        :param model_save: path to save model
        :param model_name: name model

        :return:
        """
        x_train, y_train = self._filter_dataset(x_train, y_train)

        x_train = self.preprocessing(x_train)
        x_train = self.preformat(x_train)
        dataset = tf.data.Dataset.from_tensor_slices((x_train,
                                                      y_train))
        dataset = dataset.shuffle(buffer_size=self.train_buf)
        dataset = dataset.batch(self.batch_size)

        self.ae_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.dc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = RegularizedAaeLoss()

        for epoch in range(self.num_epochs):
            epoch_ae_loss_avg = tf.metrics.Mean()
            epoch_dc_loss_avg = tf.metrics.Mean()
            epoch_dc_acc_avg = tf.metrics.Mean()
            epoch_gen_loss_avg = tf.metrics.Mean()
            for batch, (batch_x, batch_y) in tqdm(enumerate(dataset)):
                ae_loss, dc_loss, dc_acc, gen_loss = self.train_step(batch_x, batch_y)
                epoch_ae_loss_avg(ae_loss)
                epoch_dc_loss_avg(dc_loss)
                epoch_dc_acc_avg(dc_acc)
                epoch_gen_loss_avg(gen_loss)

            if verbose:
                print(f"Epoch {epoch} AE loss: {epoch_ae_loss_avg.result():.2e}, "
                      f"DC loss: {epoch_dc_loss_avg.result():.2e}, "
                      f"DC acc: {epoch_dc_acc_avg.result():.5f}, "
                      f"Gen loss: {epoch_gen_loss_avg.result():.2e}")

        return epoch_dc_acc_avg.result()

    def save_raae_with_random_id(self, parent_folder):
        """Save the three models (enc, dec, disc) with random id"""
        folder = os.path.join(parent_folder, f"{self.class_1}-{self.class_2}-{self.random_id}")

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.encode_.save_weights(os.path.join(folder, f"encoder.h5"))
        self.decode_.save_weights(os.path.join(folder, f"decoder.h5"))
        self.discriminator_.save_weights(os.path.join(folder, f"discriminator.h5"))

    def _init_fake_distr(self):
        reference_1 = [-3, 0, -3, 0]
        reference_2 = [3, 0, 3, 0]
        self.std_1 = np.ones(self.z_dim)
        self.std_2 = np.ones(self.z_dim)
        mu_1 = []
        mu_2 = []
        j = 0
        for _ in range(self.z_dim):
            mu_1.append(reference_1[j])
            mu_2.append(reference_2[j])
            j += 1
            if j >= len(reference_1):
                j = 0
        self.mu_1 = np.array(mu_1)
        self.mu_2 = np.array(mu_2)

    def load_weights(self, folder):
        """Load weights from folder"""
        self.encode_.load_weights(os.path.join(folder, f"encoder.h5"))
        self.decode_.load_weights(os.path.join(folder, f"decoder.h5"))
        self.discriminator_.load_weights(os.path.join(folder, f"discriminator.h5"))

    def assign_label(self, latent_space_samples: np.ndarray):
        """Calculate a probabilisitc ground truth for the given latent space samples"""
        inp_1 = self._disc_input(latent_space_samples, self.class_1)
        disc_pred_class_1 = self.discriminator(inp_1).numpy()
        inp_2 = self._disc_input(latent_space_samples, self.class_2)
        disc_pred_class_2 = self.discriminator(inp_2).numpy()
        combined_pred = np.concatenate([disc_pred_class_1, disc_pred_class_2], axis=1)
        res = tf.keras.layers.Softmax()(combined_pred).numpy()
        return res

    def _disc_input(self, z, disc_label):
        if isinstance(disc_label, int):
            disc_label = np.ones((z.shape[0])) * disc_label
        oh_labels_1 = tf.one_hot(tf.dtypes.cast(disc_label, tf.int32), self.num_labels)
        return tf.concat([z, oh_labels_1], -1)

    def eval_class_accuracy(self, x_test, y_test):
        """The accuracy of the discriminator to discriminate between the two classes"""
        x_test, y_test = self._filter_dataset(x_test, y_test)
        encoded = self.encode(x_test.reshape(-1, 28 * 28))
        prob_labels = self.assign_label(encoded)
        det_01_labels = np.argmax(prob_labels, axis=1)
        det_real_labels = np.full_like(det_01_labels, -1)
        det_real_labels[det_01_labels == 0] = self.class_1
        det_real_labels[det_01_labels == 1] = self.class_2
        assert np.count_nonzero(det_real_labels == -1) == 0
        correctly_classified = np.count_nonzero(det_real_labels == y_test)
        return correctly_classified / len(y_test)
