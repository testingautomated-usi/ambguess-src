"""The encoder used in our AmbiGuess autoencoders."""
import tensorflow as tf


class Encoder(tf.keras.Model):
    """The encoder used in our AmbiGuess autoencoders."""

    def __init__(self, dim, dropout, **kwargs):
        """
        Encoder model
        :param dim: hyperparameters of the model [h_dim, z_dim]
        :param dropout: Noise dropout [0,1]
        :param kwargs: Keras parameters (Optional)
        """
        h_dim = dim[0]
        z_dim = dim[1]
        super(Encoder, self).__init__(**kwargs)

        self.fc1 = tf.keras.layers.Dense(h_dim)
        # self.fc2 = tf.keras.layers.Dense(z_dim)
        self.fc2 = tf.keras.layers.Dense(h_dim)
        self.fc3 = tf.keras.layers.Dense(z_dim)
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
        z = self.fc3(h)
        return z
        # h = tf.nn.relu(self.fc1(inputs))
        # h = self.d1(h)
        # return self.fc2(h)
