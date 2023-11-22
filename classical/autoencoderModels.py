import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


class ConvAutoencoder_256(Model):
    def __init__(self, latent_dim, image_size):
        super(ConvAutoencoder_256, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([

            layers.Conv2D(input_shape=(image_size[0], image_size[1], image_size[2]), filters=64, kernel_size=3,
                          padding='same', name='enc'),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D((4, 4)),  # highligh most present features while reducing size

            layers.Conv2D(32, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D((4, 4)),

            layers.Conv2D(16, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(1, 1, padding='same'),  # 1x1 filter to map 3 channels onto one pixel
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense((latent_dim))

        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense((latent_dim)),
            layers.Reshape((4, 4, 1)),

            layers.Conv2D(input_shape=(4, 4, 1), filters=1, kernel_size=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),
            layers.UpSampling2D((2, 2)),

            layers.Conv2D(16, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),
            layers.UpSampling2D((2, 2)),

            layers.Conv2D(32, 3, padding='same'),
            layers.LeakyReLU(alpha=0.3),
            layers.UpSampling2D((4, 4)),

            layers.Conv2D(64, 3, padding='same'),
            layers.LeakyReLU(alpha=0.3),
            layers.UpSampling2D((4, 4)),

            layers.Conv2D(3, 3, padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvAutoencoder_64(Model):
    def __init__(self, latent_dim, image_size):
        super(ConvAutoencoder_64, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([

            layers.Conv2D(input_shape=(image_size[0], image_size[1], image_size[2]), filters=64, kernel_size=3,
                          padding='same', name='enc'),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D((2, 2)),  # highligh most present features while reducing size

            layers.Conv2D(32, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(16, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(1, 1, padding='same'),  # 1x1 filter to map 3 channels onto one pixel
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense((latent_dim))

        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense((latent_dim)),
            layers.Reshape((4, 4, 1)),

            layers.Conv2D(input_shape=(4, 4, 1), filters=1, kernel_size=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),
            layers.UpSampling2D((2, 2)),

            layers.Conv2D(16, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.3),
            layers.UpSampling2D((2, 2)),

            layers.Conv2D(32, 3, padding='same'),
            layers.LeakyReLU(alpha=0.3),
            layers.UpSampling2D((2, 2)),

            layers.Conv2D(64, 3, padding='same'),
            layers.LeakyReLU(alpha=0.3),
            layers.UpSampling2D((2, 2)),

            layers.Conv2D(3, 3, padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class SimpleAutoencoder_64(Model):
    def __init__(self, latent_dim):
        super(SimpleAutoencoder_64, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(2048, activation='sigmoid'),
            layers.Reshape((2048, 1))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class SimpleAutoencoder_256(Model):
    def __init__(self, latent_dim):
        super(DeepAutoencoder_256, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(32768, activation='sigmoid'),
            layers.Reshape((32768, 1))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DeepAutoencoder_64(Model):
    def __init__(self, latent_dim):
        super(DeepAutoencoder_64, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(2048, activation='sigmoid'),
            layers.Reshape((2048, 1))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DeepAutoencoder_256(Model):
    def __init__(self, latent_dim):
        super(DeepAutoencoder_256, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(4096, activation='relu'),
            layers.Dense(32768, activation='sigmoid'),
            layers.Reshape((32768, 1))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
