# Self-Driving-Car-ND-Predict-Steering-Angle-with-CNN

Predicting the steering angle of the car using CNN with center/front image as input.

## Overview
This repo is for the submission of Project 3 'Behavioral Cloning' in Udacity's SDC ND. The goal was to use a game simulator and drive a car using CNN.

## Collecting training data
The simulator allows you to drive the car manually and it captures front, left and right images along with speed, acceleration and steering angle of the car as you drive. You then use the training data captured by manually driving and teach CNN to drive and predict driving angle.

Earlier on while playing with the manual mode I realized that the car was very difficult to control using keyboard input. I didn't have a joystick so I decided to use computer vision to drive the car first.

The CV uses the same automatic mode in the simulator but when it gets an input image it predicts where the lanes are and then calculate the steering angle. As steering angle is also a function of how fast you are driving, it uses input speed as and steering angle and also returns accelerator and break setting of the car. The only drawback of this is that only center camera images are available in automatic mode that we save for training. We don't have access to left and right camera images right now.

Please refer to [this project](https://github.com/sjamthe/Self-Driving-Car-ND-Predict-Steering-Angle-with-CV) for details on how the CV drives the car.

## Model
Most of the steering angle prediction model including the one by Nvdia use complete single image as input to predict the angle. Even after cropping the image to a smaller size as in my CV program I found the image much larger to fit on the GPU memory.

 In my CV steering angle program I had noticed that sometimes it is hard find both left and right lane markings so I had to calculate the steering angle based on only one lane. I decided to write my CNN model with this idea in mind.

 I split my image into two parts. The left images trains a network to predict a steering angle, the same network (but a different model) predicts steering angle again. I look at the probabilities of both prediction and select the one with higher probability as the final steering angle.

### Data preparation
Each captured input image is 320 x 160 x 3.
![](images/input.jpg)


