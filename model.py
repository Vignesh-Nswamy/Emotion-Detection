import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input


class Model:
    def __init__(self):
        self.he_normal = tf.keras.initializers.he_normal(seed=0)
        self.lecun_normal = tf.keras.initializers.lecun_normal(seed=0)
        self.model = Sequential()

    def add_conv_stack(self, out_channels, kernel_size, strides, padding='same'):
        self.model.add(Conv2D(out_channels,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              kernel_initializer=self.he_normal))

        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())

    def get_model(self):
        self.model.add(Input(shape=(48, 48, 1)))

        self.add_conv_stack(256, kernel_size=3, strides=1)
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.add_conv_stack(128, kernel_size=3, strides=1)
        self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.add_conv_stack(64, kernel_size=3, strides=1)
        self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.model.add(Flatten())

        self.model.add(Dense(1024, kernel_initializer=self.he_normal))
        self.model.add(Dropout(0.3))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())

        self.model.add(Dense(1024, kernel_initializer=self.he_normal))
        self.model.add(Dropout(0.3))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())

        self.model.add(Dense(7, activation='softmax', kernel_initializer=self.lecun_normal))

        return self.model


if __name__ == '__main__':
    model = Model().get_model()
    model.summary()
