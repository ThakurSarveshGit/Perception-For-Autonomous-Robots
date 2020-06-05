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
  <img src="AdvancedLaneDetection/laneDetection.gif?raw=true" alt="Advance Lane Detection for Autonomous Cars"/>
</p>

## Project Five: Visual Odometry

Visual Odometry is a crucial concept in Robotics Perception for estimating the trajectory of the robot (the camera on the robot to be precise). The concepts involved in Visual Odometry are quite the same for SLAM which needless to say is an integral part of Perception.

In this project we have frames of a driving sequence taken by a camera in a car, and the scripts
to extract the intrinsic parameters. I implemented the different steps to estimate the 3D motion of the camera, and provide as output a plot of the trajectory of the camera.

<p align="center">
  <img src="AugmentedReality/PutAVirtualCube.gif?raw=true" alt="Homography and Pose Estimation"/>
</p>


