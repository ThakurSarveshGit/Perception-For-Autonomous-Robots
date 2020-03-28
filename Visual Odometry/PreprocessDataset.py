# Visual Odometry
# Author: Sarvesh Thakur & Shelly Bagchi
# File: PreprocessDataset: Creates an undistored frame video for the entire dataset, so we can process the frames directly.

import time, sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math, glob

from ReadCameraModel import *
from UndistortImage import *

# Import Images
def importImages(foldername: str)->np.ndarray:
	fileNames = os.listdir(foldername) # Getting all camera Images
	fileNames.sort() # Sorting the Images
	images = [] # List to store all the image arrays

	for i,img in enumerate(fileNames):
		print("No. of images read: ", i)
		imageName = os.path.join(foldername, img)
		images.append(cv2.imread(imageName, 0))
	return images


# Dataset Preparation
if __name__ == "__main__":
	t1 = time.time()
	images = importImages('Oxford_dataset/stereo/centre')
	width = images[0].shape[1]
	height = images[0].shape[0]
	video = cv2.VideoWriter('Oxford_dataset/UndistortedVideoFull.avi', cv2.VideoWriter_fourcc(*'XVID'), 23, (width, height));

	colorImages = []
	undistortedImages = []
	undistortedImagesGray = []

	fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model/')

	for i, image in enumerate(images):
		if i > 20:
			print("Images processed: ", i)
			
			colorImg = cv2.cvtColor(image, cv2.COLOR_BAYER_GR2BGR)
			# colorImages.append(colorImg)
			
			undistImage = UndistortImage(colorImg, LUT)
			grayImage = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
			
			video.write(undistImage)
		else:
			continue

	video.release()
	cv2.destroyAllWindows()

	t2 = time.time()

	print("Total Images processed: ",len(colorImages))
	print("TIme Elapsed: ", (t2 -t1)/60, " minutes")