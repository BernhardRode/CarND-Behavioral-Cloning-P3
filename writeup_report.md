# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* network_nvidia.py definition of the nvidia based model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py & network_nvidia.py files contain the code for training and saving the convolution neural network. The files show the pipeline I used for training and validating the model, and they contain comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the NVIDIA paper and start building upon it.

[https://arxiv.org/pdf/1604.07316.pdf](NVIDIA Paper)

[image1]: ./writeup/network_nvidia.png "NVIDIA Model"
!["NVIDIA Model"][image1]

During my research I tried out different model implementations (network_*.py), but the NVIDIA thing was the one, I felt most comfortable with (model.py lines 200-225).

#### 2. Final Model Architecture

I'm using a sequential keras model (model.py line 194). Before the samples go to the "real" network, I apply some cropping (get rid of sky and dashboard) and normalizing (model.py lines 195-198).

My model is based on the NVIDIA paper and is defined in network_vidia.py it consistens basically of the following layers:

* 5 Conv2D Layers with RELU activation
* 1 Flatten Layer
* 4 Dense Layers
* 2 Dropout Layers

The model contains dropout layers in order to reduce overfitting (network_nvidia.py 20 & 23). 
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 231).

#### 3. Creation of the Training Set & Training Process

At first i tried creating data with maximum image details in the simulator, that did not work out so well. These are the *_hd datasets. The other datasets have been created with "fastest" settings in the simulator.

At the end, I've ended up using several datasources (model.py lines 5-30), they can be switched on and of easily by turning the boolean flags around.

At the beginning, I started using the udacity set and added new sets when needed. I've ended up using 7 different datasets. 

* track01 - udacity - the data provided by udacity
* track01 - safety_laps - driving three laps in a row 
* track01 - bridge_exits - 5 times exiting the bridge correctly
* track01 - sharp_turns - different sharp turn drives
* track02 - safety_lap - driving two laps in a row 
* track02 - reverse_lap - driving one lap in reverse direction

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Sample Preparation

When my application starts, it first iterate sover the configured datasets reads their data (model.py line 106-120).

Because we have way more straigt driving then turn driving, i've created a simple sampling algorithm (model.py 122-127).
When the recorded steering angle is between -0.02 and +0.02 (ZERO_ANGLE), I drop roundabout 90% (DROP_ZERO_ANGLE_PROB = 0.9) of the samples. This helps generalizing the model better.

[image2]: ./writeup/figure_1.png "Histogram without drop"
!["Histogram without drop"][image2]


[image3]: ./writeup/figure_2.png "Histogram with drop"
!["Histogram with drop"][image3]

If a sample should be added, I create four different samples out of it.

* center image with no steering correction
* left image with steering correction of +0.08
* right image with steering correction of -0.08
* flipped center image with -steering

[image4]: ./writeup/center.jpg "Original"
[image5]: ./writeup/center_image.png "Unmodified"
[image6]: ./writeup/flipped_image.png "Flipped"
[image7]: ./writeup/augmented_brightness_image_1.png "Augemented Brightness 1"
[image8]: ./writeup/augmented_brightness_image_2.png "Augemented Brightness 2"
[image9]: ./writeup/augmented_brightness_image_3.png "Augemented Brightness 3"

!["Original"][image4]
!["Unmodified"][image5]
!["Flipped"][image6]
!["Augemented Brightness 1"][image7]
!["Augemented Brightness 2"][image8]
!["Augemented Brightness 3"][image9]

All the samples create a lamda function, which will when executed (in the generator later on) read and process the image.

When all the samples have been created, I split the samples up into train and validation samples with a test_size of 20%.

Number of samples: 69824
Number of train samples: 55859
Number of valid samples: 13965

##### Image processing

The lamda functions lambda_read_image & lambda_read_image_flipped (model.py 98-101) are being used to read in images from the harddrive and add some basic image pre processing.

* lambda_read_image - just reads the image with image_read
* lambda_read_image_flipped - reads the image with image_read and flips it around

The image_read function takes a filename and reads that image into memory. Then it converts the color space and adds random brightness, to prevent overfitting later on.

I also thought about adding some random noise, but at the end it wasn't needed for this project.

#### Training Process

The training process is using a generator (model.py 169-189) it batches image loading in batches of 32 samples.

It shuffles the batch of samples. Then it itereates over the batch of samples and calls the lamda function to read each image.

I've got an EarlyStopper (model.py 242-251) callback implemented, as I started to use with more then 30 epochs per training.
I'm not using it any more, because from what i've seen training of 3 epochs is totally sufficent for this project to work fine.

