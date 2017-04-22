from keras.models import Sequential
from keras.layers.core import Lambda, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, Activation

def Nvidia(model, dropout=0.5):
    model.add(Conv2D(24, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))