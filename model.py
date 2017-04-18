################################
# HYPER PARAMETERS
################################

DATASETS = {
    'track_01': {
        'fast_lap_001': True,
        'safety_lap_001': False,
        'safety_lap_002': False,
        'safety_lap_003': True,
        'safety_lap_004': True,
        'multiple_drives': False,
        'bridge_exit_001': True,
        'reverse_lap_001': True,
        'poc': False,
        'udacity': True,
    },
    'track_02': {
        'safety_lap_001': False,
        'safety_lap_002': False,
        'reverse_lap_001': False
    }
}
STEERING_CORRECTION = 0.18
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
IMAGE_SIZE = (160, 320, 3)
CROPPING_TB = (70, 15)
CROPPING_LR = (0, 0)
EPOCHS = 50 # can be big, as we stop early, when network performs well enough
EARLY_STOPPING_DELTA = 0.0001
EARLY_STOPPING_PATIENCE = 1
VERBOSE = 1

################################
# DO NOT CHANGE ANYTHING BELOW
################################

import os
import csv
import time
import json
import math
import cv2
import random
import numpy as np

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# set start time
start = time.time()

#
# Setup useful helper functions
#
SAMPLES = []
def read_image(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img

def lambda_read_image(path):
    return lambda : read_image(path)

def lambda_read_image_flipped(path):
    return lambda : np.fliplr(read_image(path))
#
# Read Drive Log and prepare samples
#
for track in DATASETS:
    for recording in DATASETS[track]:
        if DATASETS[track][recording]:
            basepath = './recordings/' + track + '/' + recording
            with open(basepath + '/driving_log.csv') as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:

                    center, left, right, = line[0], line[1], line[2]
                    center = basepath + '/IMG/' + center.split('/')[-1]
                    left = basepath + '/IMG/' + left.split('/')[-1]
                    right = basepath + '/IMG/' + right.split('/')[-1]
                    steering = float(line[3])
                    throttle = float(line[4])
                    brake = float(line[5])
                    speed =  float(line[6])

                    SAMPLES.append({
                        'lambda_image': lambda_read_image(center), 
                        'steering': steering, 
                        'brake': brake,
                        'speed': speed, 
                        'throttle': throttle
                    })
                    SAMPLES.append({
                        'lambda_image': lambda_read_image(left), 
                        'steering': steering + STEERING_CORRECTION, 
                        'brake': brake,
                        'speed': speed, 
                        'throttle': throttle
                    })
                    SAMPLES.append({
                        'lambda_image': lambda_read_image(right), 
                        'steering': steering - STEERING_CORRECTION, 
                        'brake': brake,
                        'speed': speed, 
                        'throttle': throttle
                    })
                    # SAMPLES.append({
                    #     'lambda_image': lambda_read_image_flipped(center), 
                    #     'steering': -steering, 
                    #     'brake': brake,
                    #     'speed': speed, 
                    #     'throttle': throttle
                    # })

print('Number of samples:', len(SAMPLES))

#
# Create Train / Validation Split
#
TRAIN_SAMPLES, VALIDATION_SAMPLES = train_test_split(
    SAMPLES,
    test_size=TEST_SIZE
)
print('Number of train samples:', len(TRAIN_SAMPLES))
print('Number of valid samples:', len(VALIDATION_SAMPLES))

print('#################################################')

def generator(samples, batch_size=32):
    '''Samples Generator'''
    num_samples = len(samples)

    while True:
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X, y = [], []

            for batch_sample in batch_samples:
                image, steering = batch_sample['lambda_image'](), batch_sample['steering']
                X.append(image)
                y.append(steering)

            X = np.array(X)
            y = np.array(y)

            X, y = shuffle(X, y)
            yield (X, y)
#
# MODEL
#
model = Sequential()
# cropping
model.add(Cropping2D(cropping=(CROPPING_TB, CROPPING_LR), input_shape=IMAGE_SIZE))
# normalize
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#
# Networks
#

# LeNet architecture
# from network_lenet import LeNet
# LeNet(model)

# AlexNet architecture
# from network_alexnet.py import AlexNet
# AlexNet(model)

# VGG architecture
# from network_vgg import VGG
# VGG(model)

# NVidia architecture
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model.add(BatchNormalization(axis=1, name="Normalise"))
model.add(Conv2D(24, (3, 3), strides=(2,2), name="Conv1", activation="relu"))
model.add(MaxPooling2D(name="MaxPool1"))
model.add(Conv2D(48, (3, 3), strides=(1,1), name="Conv2", activation="relu"))
model.add(MaxPooling2D(name="MaxPool2"))
model.add(Conv2D(72, (3, 3), strides=(1,1), name="Conv3", activation="relu"))
model.add(MaxPooling2D(name="MaxPool3"))
model.add(Dropout(0.2, name="Dropout1"))

model.add(Flatten(name="Flatten"))
model.add(Dense(100, activation="relu", name="FC2"))
model.add(Dropout(0.5, name="Dropout2"))
model.add(Dense(50, activation="relu", name="FC3"))
model.add(Dropout(0.2, name="Dropout3"))
model.add(Dense(10, activation="relu", name="FC4"))

model.add(Dense(1, name="Steering", activation='linear'))

#
# NOT IMPLEMENTED
#
# GoogLeNet architecture 
# from network_googlenet.py import GoogLeNet
# GoogLeNet(model)

#
# COMPILE
#
model.compile(loss='mse', optimizer='adam')

#
# Values
#
training_steps = int(math.ceil(len(TRAIN_SAMPLES) / BATCH_SIZE))
validation_steps = int(math.ceil(len(VALIDATION_SAMPLES) / BATCH_SIZE))

#
# Callbacks
#
# earlyStopper
earlyStopper = EarlyStopping(min_delta=EARLY_STOPPING_DELTA, patience=EARLY_STOPPING_PATIENCE, mode='min')
# Checkpointer
checkpointer = ModelCheckpoint(filepath="./model.cp.h5", verbose=VERBOSE, save_best_only=True)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

history = LossHistory()

# START GENERATORS
TRAIN_GENERATOR = generator(TRAIN_SAMPLES, batch_size=BATCH_SIZE)
VALIDATION_GENERATOR = generator(VALIDATION_SAMPLES, batch_size=BATCH_SIZE)
#
# FIT
#
train_generator = TRAIN_GENERATOR
steps_train = training_steps
print('Start training model')
history_obj = model.fit_generator(
    train_generator, 
    steps_train,
    validation_data = VALIDATION_GENERATOR,
    validation_steps = validation_steps,
    epochs = EPOCHS,
    verbose=VERBOSE,
    callbacks=[history, earlyStopper, checkpointer]
)

#
# Save
#
model.save('model.h5')
json.dump(model.to_json(), open('model.json', 'w'))
print('Saved model and dumped data.')

#
# PLOT
#
# https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/0f26c2e3-6feb-4ad6-a472-6312c6a3c60e

# set end time
end = time.time()

# display duration of training
duration = end - start
print('Done in ', duration / 60)