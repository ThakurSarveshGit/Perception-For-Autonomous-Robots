# Perception for Autonomous Robotics

### Authors: Akshitha Pothamshetty & Sarvesh Thakur

## Project One: Augmented Reality

This project identifies the tag id from a series of frames taken from a drone attached camera. The direct application of this project is aid the landing of a drone by recognizing the appropriate tag id from the downward facing camera. The QR tags are a simplified version that can encode ids from 0 to 15[8 bits]. The four corner points were robustly estimated across all frames, based on which homography calculations were performed to recover the camera pose. This information was later use to project a virtual cube in the scene.

<p align="center">
  <img src="AugmentedReality/PutAVirtualCube.gif?raw=true" alt="Homography and Pose Estimation"/>
</p>

## Project Two: Lane Detection for Self-Driving Cars

In this project we aim to do simple Lane Detection to mimic Lane Departure Warning systems used in SelfDriving Cars.  We are provided with two video sequences(one in normal lighting and other one in changing light intensities), taken from a self driving car.  Our task was to design an algorithm to detect lanes on the road,as well as estimate the road curvature to predict car turns.

<p align="center">
  <img src="AdvanceLaneDetection/laneDetection.gif?raw=true" alt="Advance Lane Detection for Autonomous Cars"/>
</p>

## Project Three: Underwater Buoy Detection using Gaussian Mixture Models(GMMs)

This project will introduced the concept of color segmentation using Gaussian Mixture Models and Expectation Maximization techniques. The video sequence provided has been captured underwater and shows three buoys of different
colors, namely yellow, orange and green. They are almost circular in shape and are distinctly colored. However, conventional segmentation techniques involving color thresholding will not work well in such an environment, since noise and varying light intensities will render any hard-coded thresholds ineffective.
In such a scenario, we “learn” the color distributions of the buoys and use that learned model to segment them. This project required us to obtain a tight segmentation of each buoy for the entire video sequence by applying a tight contour (in the respective color of the buoy being segmented) around each buoy.

<p align="center">
  <img src="BuoyDetectionGMM/buoyDetection.gif?raw=true" alt="Underwater Buoy Detection using Gaussian Mixture Models(GMMs)"/>
</p>

## Project Four: Lucas Kanade Object Tracker

In this project, we implemented Lucas-Kanade(LK) algorithm that minimizes the sum of squared error between the image and a template image( object to be tracked in prior image), to track an object in an image.
We Evaluated the code on three video sequence: the car sequence, the human walking and the table vase scene. We added advanced approaches in the code for implementing robustness to illumination. (Eg. used robust M-estimator to avoid outliers affecting the cost function in LK algorithm).


<p align="center">
  <img src="LucasKanadeTracker/GIF_LUCAS_KANADE.gif?raw=true" alt="Lucas Kanade Object Tracker"/>
</p>

## Project Five: Visual Odometry

Visual Odometry is a crucial concept in Robotics Perception for estimating the trajectory of the robot (the camera on the robot to be precise). The concepts involved in Visual Odometry are quite the same for SLAM which needless to say is an integral part of Perception.

In this project we have frames of a driving sequence taken by a camera in a car, and the scripts
to extract the intrinsic parameters. I implemented the different steps to estimate the 3D motion of the camera, and provide as output a plot of the trajectory of the camera.

<p align="center">
  <img src="VisualOdometry/visualodom.gif?raw=true" alt="Homography and Pose Estimation"/>
</p>


