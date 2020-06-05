from AugmentedReality import *
import numpy as np
import cv2
import copy

# Initiating the Augmented_Reality class.
ar_class = Augmented_Reality()

# Change these for different videos and image inputs
path_to_video = 'Tag0.mp4'
path_to_image = 'Lena.png'

# Video and image read objects
video_cap = cv2.VideoCapture(path_to_video)
image_cap = cv2.imread(path_to_image)

# template corners.
points_template = np.array([[0,image_cap.shape[0]],[image_cap.shape[1],image_cap.shape[0]],[image_cap.shape[1],0],[0,0]])


while video_cap.read():
	ret, image = video_cap.read()
	original_image = copy.deepcopy(image)

	if not ret:
		break
	
	tag_images_list, tag_corners_list = ar_class.tag_detection(original_image)

	if len(tag_images_list) > 0:
		for tag_image in tag_images_list:
			tag_image, tag_id = ar_class.get_tag_id(tag_image)
			cv2.putText(tag_image,str(tag_id),(tag_image.shape[0]/2 - 5,tag_image.shape[1]/2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1,cv2.LINE_AA)
			# cv2.imshow('Tag ID',tag_image)

	if len(tag_corners_list) > 0:
		for corner in tag_corners_list:
			for i in range(0,4):
				cv2.circle(image,(corner[i,0],corner[i,1]),3,(255,0,0),5)# Blue Corners
				resized_corner = cv2.resize(image,(0,0), fx=0.5, fy=0.5 )
				cv2.imshow('Corners', resized_corner)

			### Placing template Image on the Tag
			H = ar_class.inverse_homography(corner, points_template)
			width_at_destination = np.array([min(corner[:,0]+1), max(corner[:,0]-1)]).astype(int)
			height_at_destination = np.array([min(corner[:,1]+1), max(corner[:,1]-1)]).astype(int)
			Image_with_template = ar_class.place_template_on_tag(original_image, H, image_cap, width_at_destination, height_at_destination)
			#cv2.imshow('Lena replaces Tags', Image_with_template)			

			### Placing Cube on the Tag
			K = ar_class.calibration_matrix# Calibration Matrix
			Camera_Pose_R_T = ar_class.estimate_camera_pose(K, H)
			frame_with_cube = ar_class.put_a_cube(K, Camera_Pose_R_T, corner, points_template, original_image)

	resized_cube = cv2.resize(original_image, (0,0), fx=0.5, fy=0.5)
	cv2.imshow('Cube Placed', resized_cube)
	cv2.waitKey(1000/30)


video_cap.release()

