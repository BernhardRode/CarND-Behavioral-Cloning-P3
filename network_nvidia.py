from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Convolution2D, MaxPooling2D


def Nvidia(model):
    model.add(BatchNormalization(axis=1, name="Normalise"))
    model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))