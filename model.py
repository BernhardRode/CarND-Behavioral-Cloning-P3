################################
# HYPER PARAMETERS
################################

DATASETS = {
    'track_01_hd': {
        'reverse_lap_001': False,
        'safety_lap_001': False,
        'safety_lap_002': False,
        'safety_lap_003': False,
        'recovery_lap_001': False
    },
    'track_01': {
        'recovery_lap': False,
        'reverse_lap': True,
        'safety_laps': True,
        'sharp_turn': True,
        'bridge_exit': True,
        'udacity': True
    },
    'track_02_hd': {
        'safety_lap_001': False,
        'safety_lap_002': False,
        'reverse_lap_001': False
    },
    'track_02': {
        'safety_laps': True,
        'reverse_lap': True
    }
}
VERBOSE = 1
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2
ANGLE_CORRECTION = 0.08
LEARNING_RATE = 0.001
IMAGE_SIZE = (160, 320, 3)
NVIDIA_SIZE = (66, 200, 3)
EPOCHS = 3
EARLY_STOPPING_PATIENCE = 1
EARLY_STOPPING_DELTA = 0.0001
CROPPING_TB = (50, 20)
CROPPING_LR = (0, 0)
BATCH_SIZE = 32
ZERO_ANGLE = 0.02
DROP_ZERO_ANGLE_PROB = 0.9

################################
# DO NOT CHANGE ANYTHING BELOW
################################

import time
import random
import os
import numpy as np
import math
import json
import cv2
import csv

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Flatten, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.callbacks import EarlyStopping, Callback

# set start time
start = time.time()

#
# helper functions
#
SAMPLES = []
#
# augmented_brightness
#
def augmented_brightness(image):
    augmented_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    augmented_image[:,:,2] = augmented_image[:,:,2]*random_bright
    augmented_image = cv2.cvtColor(augmented_image,cv2.COLOR_HSV2RGB)
    return augmented_image
#
# Read image adapt color and augment brightness
#
def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = augmented_brightness(image)
    shape = image.shape
    return image

def lambda_read_image(path):
    return lambda : read_image(path)

def lambda_read_image_flipped(path):
    return lambda : cv2.flip(read_image(path), 1)
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
                    angle = float(line[3])
                    throttle = float(line[4])
                    brake = float(line[5])
                    speed =  float(line[6])

                    add = True
                    rand = np.random.random()
                    if -ZERO_ANGLE < angle < ZERO_ANGLE and rand > DROP_ZERO_ANGLE_PROB:
                        add = False

                    if add:
                        SAMPLES.append({
                            'lambda_image': lambda_read_image(center), 
                            'angle': angle, 
                            'brake': brake,
                            'speed': speed, 
                            'throttle': throttle
                        })
                        SAMPLES.append({
                            'lambda_image': lambda_read_image(left), 
                            'angle': angle + ANGLE_CORRECTION, 
                            'brake': brake,
                            'speed': speed, 
                            'throttle': throttle
                        })
                        SAMPLES.append({
                            'lambda_image': lambda_read_image(right), 
                            'angle': angle - ANGLE_CORRECTION, 
                            'brake': brake,
                            'speed': speed, 
                            'throttle': throttle
                        })
                        SAMPLES.append({
                            'lambda_image': lambda_read_image_flipped(center), 
                            'angle': -angle, 
                            'brake': brake,
                            'speed': speed, 
                            'throttle': throttle
                        })

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
                image, angle = batch_sample['lambda_image'](), batch_sample['angle']
                X.append(image)
                y.append(angle)

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
model.add(Lambda(lambda x: x/127.5 - 1.0))

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
from network_nvidia import Nvidia
Nvidia(model)

#
# NOT IMPLEMENTED
#
# GoogLeNet architecture 
# from network_googlenet.py import GoogLeNet
# GoogLeNet(model)

#
# COMPILE
adam = Adam(lr=LEARNING_RATE)
model.compile(optimizer=adam, loss='mse')

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
    callbacks=[history, earlyStopper]
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