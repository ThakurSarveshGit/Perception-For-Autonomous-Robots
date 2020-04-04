# Visual Odometry Implementation from Scratch
# Author: Sarvesh Thakur
# File: VisualOdometryScratch: Raw implementation of VO for Car Dashcam

# Python Libraries
import cv2
import math,glob
import numpy as np
import sympy as sp
import random as rand
from scipy import stats
import os, sys, copy, time
import matplotlib.pyplot as plt

from ReadCameraModel import *
from UndistortImage import *


# class Cells:
class Cells:
	def __init__(self):
		self.pts = list()
		self.pairs = dict()

	def randPt(self):
		return rand.choice(self.pts)

# todo: Class VisualOdometry
class VisualOdometry:
	def __init__(self):
		return


if __name__ == "__main__":

	# ----- Initialize variables ----- #
	Translation = np.zeros((3,1)) # 3x1 Column matrix
	Rotation = np.eye(3) # 3x3 matrix

	# pyplot setup
	fig = plt.figure('Figure 1', figsize=(7,5))
	fig.suptitle("Visual Odometry for Autonomous Cars")
	axis1 = fig.add_subplot(111)
	axis1.set_title("Authors: Sarvesh Thakur & Shelly Bagchi")

	# video setup
	cap = cv2.VideoCapture("Oxford_dataset/UndistortedVideoFull.avi") # Preprocessed Dashcam Video
	ret, currentFrameOriginal = cap.read()
	
	# Region of interest for VO
	imageDimensions = currentFrameOriginal.shape
	yBar, xBar = np.array(imageDimensions[:-1])/8 # Take the center area

	vo = VisualOdometry() # Class Visual Odometry
	
	# accessing first frame
	currentFrame = currentFrameOriginal.copy()
	currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
	
	index = 0 # Debuging

	# while cap.isOpened():
	while index < 30: # debuging
		index +=1
		ret, nextFrameOriginal = cap.read()
		nextFrame = nextFrameOriginal.copy()

		if ret:
			
			#------- 1. SIFT Feature Detection ----------#
			nextFrame = cv2.cvtColor(nextFrame, cv2.COLOR_BGR2GRAY) # gray
			sift = cv2.xfeatures2d.SIFT_create()
			kp_current, des_current = sift.detectAndCompute(currentFrame, None)
			kp_next, des_next = sift.detectAndCompute(nextFrame, None)

			# Extract the best matches
			bestMatches = []
			bf = cv2.BFMatcher()
			matches = bf.knnMatch(des_current, des_next, k=2)
			
			for m, n in matches:
				if m.distance < 0.5*n.distance:
					bestMatches.append(m)

			# displaying the matches
			imageMatch = cv2.drawMatches(currentFrame, kp_current, nextFrame, kp_next, bestMatches, outImg=currentFrame, matchColor=None, singlePointColor=(255, 255, 255),  flags=2)


			#-------- 2. ZHANG's 8x8 grid containing matched points --------#
			
			pointCorrespondence_cf = np.zeros((len(bestMatches), 2))
			pointCorrespondence_nf = np.zeros((len(bestMatches), 2))
			grid = np.empty((8,8), dtype=object) # This 8x8 blocks keeps the best pairs of matches
			grid[:, :] = Cells()
			
			for i, match in enumerate(bestMatches):
				j = int(kp_current[match.queryIdx].pt[0]/xBar)
				k = int(kp_current[match.queryIdx].pt[1]/yBar)
				# Assigning points to blocks in grid
				grid[j, k].pts.append(kp_current[match.queryIdx].pt)
				grid[j, k].pairs[kp_current[match.queryIdx].pt] = kp_next[match.trainIdx].pt
				
				pointCorrespondence_cf[i] = kp_current[match.queryIdx].pt[0], kp_current[match.queryIdx].pt[1]
				pointCorrespondence_nf[i] = kp_next[match.trainIdx].pt[0], kp_next[match.trainIdx].pt[1]
				
			# print(grid[4,4].pairs)

			#-------- todo: 3. Estimating the translation and rotation matrices --------#
			# F = vo.EstimateFundamentalMatrixRANSAC(pointCorrespondence_cf, pointCorrespondence_nf, matches, grid, 0.05)
			# print(F)

			# Estimating Rotation and Translation points

			# Displaying results every 10 images
			if index%10 == 0:
				plt.imshow(imageMatch)
				plt.show(block=False)
				plt.pause(0.05)
				plt.close()

