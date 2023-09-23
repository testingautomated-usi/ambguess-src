"""Loss function used in the regularized adversarial autoencoder."""
import tensorflow as tf


class RegularizedAaeLoss:
    """Loss function used in the regularized adversarial autoencoder."""

    def __init__(self, ae_loss_weight=1.,
                 gen_loss_weight=1.,
                 dc_loss_weight=1):
        """
        Loss functions of Adversal AutoEncoder
        :param ae_loss_weight: weight of AE loss
        :param gen_loss_weight: weight of General loss
        :param dc_loss_weight: weight of Discriminator
        """
        self.ae_loss_weight = ae_loss_weight
        self.gen_loss_weight = gen_loss_weight
        self.dc_loss_weight = dc_loss_weight
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.accuracy = tf.keras.metrics.BinaryAccuracy()

    def autoencoder_loss(self, inputs, reconstruction):
        """
        Loss of the AutoEncoder (AE)
        :param inputs: Input data
        :param reconstruction: Reconstruction from the AE
        :return: MSE loss
        """
        return self.ae_loss_weight * self.mse(inputs, reconstruction)
        # return tf.image.ssim(tf.reshape(inputs, [100, 28, 28]),
        #                                tf.reshape(reconstruction, [100, 28, 28]), 1.0)

    def discriminator_loss(self, real_output, fake_output):
        """
        Loss of the discriminator
        :param real_output: Real distribution
        :param fake_output: Fake Distribution
        :return: cross entropy loss
        """
        loss_real = self.cross_entropy(tf.ones_like(real_output),
                                       real_output)
        loss_fake = self.cross_entropy(tf.zeros_like(fake_output),
                                       fake_output)
        return self.dc_loss_weight * (loss_fake + loss_real) / (real_output.shape[0] + fake_output.shape[0])

    def generator_loss(self, fake_output):
        """
        Generator loss
        :param fake_output: Fake Distribution
        :return: cross entropy loss
        """
        return self.gen_loss_weight * \
               self.cross_entropy(tf.ones_like(fake_output), fake_output) / fake_output.shape[0]
