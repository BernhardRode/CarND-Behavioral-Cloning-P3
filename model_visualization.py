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
ZERO_ANGLE = 0.05
DROP_ZERO_ANGLE_PROB = 0.2

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

example_image = 'writeup/center.jpg'
image = cv2.imread(example_image)
flipped_image = cv2.flip(cv2.imread(example_image), 1)
augmented_brightness_image_1 = augmented_brightness(image)
augmented_brightness_image_2 = augmented_brightness(image)
augmented_brightness_image_3 = augmented_brightness(image)

cv2.imwrite("writeup/image.png", image);
cv2.imwrite("writeup/flipped_image.png", flipped_image);
cv2.imwrite("writeup/augmented_brightness_image_1.png", augmented_brightness_image_1);
cv2.imwrite("writeup/augmented_brightness_image_2.png", augmented_brightness_image_2);
cv2.imwrite("writeup/augmented_brightness_image_3.png", augmented_brightness_image_3);

#
# Read Drive Log and prepare samples
#
# for track in DATASETS:
#     for recording in DATASETS[track]:
#         if DATASETS[track][recording]:
#             basepath = './recordings/' + track + '/' + recording
#             with open(basepath + '/driving_log.csv') as csvfile:
#                 reader = csv.reader(csvfile)
#                 for line in reader:
#                     center, left, right, = line[0], line[1], line[2]
#                     center = basepath + '/IMG/' + center.split('/')[-1]
#                     left = basepath + '/IMG/' + left.split('/')[-1]
#                     right = basepath + '/IMG/' + right.split('/')[-1]
#                     angle = float(line[3])
#                     throttle = float(line[4])
#                     brake = float(line[5])
#                     speed =  float(line[6])

#                     add = True
#                     rand = np.random.random()
#                     if -ZERO_ANGLE < angle < ZERO_ANGLE and rand > DROP_ZERO_ANGLE_PROB:
#                         add = True

#                     if add is True:
#                         SAMPLES.append({
#                             'lambda_image': lambda_read_image(center), 
#                             'angle': angle, 
#                             'brake': brake,
#                             'speed': speed, 
#                             'throttle': throttle
#                         })
#                         SAMPLES.append({
#                             'lambda_image': lambda_read_image(left), 
#                             'angle': angle + ANGLE_CORRECTION, 
#                             'brake': brake,
#                             'speed': speed, 
#                             'throttle': throttle
#                         })
#                         SAMPLES.append({
#                             'lambda_image': lambda_read_image(right), 
#                             'angle': angle - ANGLE_CORRECTION, 
#                             'brake': brake,
#                             'speed': speed, 
#                             'throttle': throttle
#                         })
#                         SAMPLES.append({
#                             'lambda_image': lambda_read_image_flipped(center), 
#                             'angle': -angle, 
#                             'brake': brake,
#                             'speed': speed, 
#                             'throttle': throttle
#                         })

# print('Number of samples:', len(SAMPLES))

# import matplotlib.pyplot as plt
# import numpy as np
# x = []
# for sample in SAMPLES:
#   x.append(sample['angle'])

# hist, bins = np.histogram(x, bins=50)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.show()