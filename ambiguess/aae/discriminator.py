"""The discriminator used in our AmbiGuess autoencoders."""
import tensorflow as tf


class Discriminator(tf.keras.Model):
    """The discriminator used in our AmbiGuess autoencoders."""

    def __init__(self, dim, dropout, **kwargs):
        """
        Discriminator model
        :param dim: hyperparameters of the model [h_dim]
        :param dropout: Noise dropout [0,1]
        :param kwargs: Keras parameters (Optional)
        """
        h_dim = dim[0]
        super(Discriminator, self).__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(h_dim, name="disc_hidden_1")
        self.fc2 = tf.keras.layers.Dense(h_dim, name="disc_hidden_2")
        self.fc3 = tf.keras.layers.Dense(1, name="disc_output")
        self.lru1 = tf.keras.layers.LeakyReLU()
        self.lru2 = tf.keras.layers.LeakyReLU()
        self.d1 = tf.keras.layers.Dropout(dropout)
        self.d2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None, mask=None):
        """
        Function that works as __call__
        :param inputs: input data
        :param training: (Not use)
        :param mask: (Not use)
        :return: model output
        """
        h = self.lru1(self.fc1(inputs))
        h = self.d1(h)
        h = self.lru2(self.fc2(h))
        h = self.d2(h)
        x = self.fc3(h)
        return x
    