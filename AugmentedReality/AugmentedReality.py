# Class Augmented_Reality
# Author: Akshitha Pothamshetty & Sarvesh Thakur
# Project One Augmented reality.
import numpy as np
import cv2
import imutils	# Simplifies many opencv functionalities with ready-made functions.


# Class Augmented_Reality
# | ------- __init__()
# | ------- order_points()
# | ------- four_point_transform()
# | ------- display_frame_till_esc()
# | ------- inverse_homography()
# | ------- estimate_camera_pose()
# | ------- contours_to_corners()
# | ------- tag_detection()
# | ------- _rel_to_abs()
# | ------- get_tag_id()

image_index = 0
class Augmented_Reality:
	
	def __init__(self):
		self.calibration_matrix = np.array([[1406.08415449821,0,0],
                        [2.20679787308599, 1417.99930662800,0],
                        [1014.13643417416, 566.347754321696,1]]).T
		return
	
	def order_points(self, points):
		# Corner points returned by contour function may not be sorted;
		# order_points() returns ordered points in the following way:
		# top_left, top_right, bottom_right, bottom_left
		
		# reference: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
		rect = np.zeros((4,2), dtype="float32")
		# the top-left point will have the smallest sum, whereas the bottom-right will have 			the largest sum
		s = points.sum(axis = 1)
		rect[0] = (points[np.argmin(s)]).astype(int)
		rect[2] = (points[np.argmax(s)]).astype(int)
		# top-right point will have the smallest difference,
		# whereas the bottom-left will have the largest difference
		diff = np.diff(points, axis=1)
		rect[1] = (points[np.argmin(diff)]).astype(int)
		rect[3] = (points[np.argmax(diff)]).astype(int)
		# return the ordered coordinates.
		return rect

	def four_point_transform(self, image, pts):
		rect = self.order_points(pts)
		(tl, tr, br, bl) = rect
	 
		# construct the set of destination points to obtain a "birds eye view"
		dst = np.array([
			[0, 0],
			[80 - 1, 0],
			[80 - 1, 80 - 1],
			[0, 80 - 1]], dtype = "float32")
	 
		# compute the perspective transform matrix and then apply it
		M = cv2.getPerspectiveTransform(rect, dst)
		warped = cv2.warpPerspective(image, M, (80, 80))# tag is of size 80x80
	 
		# return the warped image
		return warped
	
	def inverse_homography(self, point_1, point_2):
		# Can even directly use cv2.findhomography() but then this is more intuitive.
		# takes in two points and returns homography for placing point_1 image on point_2
		# returns homog matrix for placing a point from Camera frame to World Frame(here video frames)
		p1 = point_1; p2 = point_2
		A  = -np.array([
			[ -p1[0][0] , -p1[0][1] , -1 , 0  , 0 , 0 , (p2[0][0]*p1[0][0]) , (p2[0][0]*p1[0][1]),(p2[0][0]) ],
			[ 0 , 0 , 0 , -p1[0][0]  , -p1[0][1] , -1 , p2[0][1]*p1[0][0] , p2[0][1]*p1[0][1] ,p2[0][1]],
			[ -p1[1][0] , -p1[1][1] , -1 , 0  , 0 , 0 , (p2[1][0]*p1[1][0]) , (p2[1][0]*p1[1][1]),(p2[1][0])],
			[ 0 , 0 , 0 , -p1[1][0]  , -p1[1][1] , -1 , p2[1][1]*p1[1][0] , p2[1][1]*p1[1][1] ,p2[1][1]],
			[ -p1[2][0] , -p1[2][1] , -1 , 0  , 0 , 0 , (p2[2][0]*p1[2][0]) , (p2[2][0]*p1[2][1]),(p2[2][0])],
			[ 0 , 0 , 0 , -p1[2][0]  , -p1[2][1] , -1 , p2[2][1]*p1[2][0] , p2[2][1]*p1[2][1] ,p2[2][1]],
			[ -p1[3][0] , -p1[3][1] , -1 , 0  , 0 , 0 , (p2[3][0]*p1[3][0]) , (p2[3][0]*p1[3][1]),(p2[3][0])],
			[ 0 , 0 , 0 , -p1[3][0]  , -p1[3][1] , -1 , p2[3][1]*p1[3][0] , p2[3][1]*p1[3][1] ,p2[3][1]],
			], dtype=np.float64)

		# SVD of A
		U, S, V = np.linalg.svd(A)
		# The elements of homography obtained from last column of V
		h = V[:][8]/V[8][8]
		H_inv = np.reshape(h, (3,3))
		H = np.linalg.inv(H_inv)
		H = H/H[2][2]; # h33 = 1
		return H


	def estimate_camera_pose(self, K, H):
		# @returns the camera pose matrix [R | t] given calibration and homography matrix
		K_inv = np.linalg.inv(K)
		Lambda = ((np.linalg.norm(np.matmul(K_inv,H[:,0]))+(np.linalg.norm(np.matmul(K_inv,H[:,1]))))/2)
		B_tilda = np.matmul(K_inv, H)

		if np.linalg.det(B_tilda) >0:
			B = B_tilda/Lambda
		else:
			B = B_tilda*(-1/Lambda)
		col1_R = B[:,0]/Lambda
		col2_R = B[:,1]/Lambda
		col3_R = np.cross(col1_R, col2_R)*Lambda
		t = np.array([B[:,2]/Lambda]).T
		R = np.array([col1_R, col2_R, col3_R]).T
		pose = np.matrix(np.hstack([R,t]))
		return pose

		
	def display_frame_till_esc(self, image):
		# Displays an image until esc key is pressed.
		global image_index
		image_index += 1
		image_name = 'Image ' + str(image_index)
		cv2.imshow(image_name, image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


	def contours_to_corners(self, quad_contour):
		# @quad_contour is in the form: [[[x1,y1]],[[x2,y2]],[[x3,y3]],[[x4,y4]]]
		# returns points in the form: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
		quad_np = np.array(quad_contour)
		quad_points = np.zeros([4,2])
		quad_points[:4,0] = quad_np[:4,0,0]# x-coordinates
		quad_points[:4,1] = quad_np[:4,0,1]# y-coordinates
		return quad_points
	

	def tag_detection(self, image):
		# return the detected tags from the image and thier corners in the form [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
		
		# Setting up blob parameters
		detector = cv2.SimpleBlobDetector()
		params = cv2.SimpleBlobDetector_Params()
		params.filterByColor = True	# Filter by Color
		params.blobColor = 255	# Detecting white blob
		params.filterByArea = True # Filter by Area
		params.maxArea = 45000 # We dont want to get the White laying sheet
		
		# Create a detector with the parameters
		detector = cv2.SimpleBlobDetector_create(params)# Now we are ready for detecting our blobs in image.
		# Resizing, Conversion and Masking.
		resized_image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# Not using Resized image.
		mask_image = cv2.inRange(gray_image, 180, 255)
		res = cv2.bitwise_and(gray_image,gray_image,mask = mask_image)# 1080x1920
		
		# Finding keypoints.
		keypoints = detector.detect(mask_image) # To access keypoints; Use keypoints.pt
		keypoints_image = cv2.drawKeypoints(res, keypoints, np.array([]), (0,0,255), 		cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		
		# Extracting tag area for further processing.
		warped_list = []# List of warped Tags
		tag_corner_absolute_list = []# List of corresponding Corners for each tag.
		for i in range(0, len(keypoints)): # Robust for Multiple Tags as well.
			# If we find a keypoint
			if (keypoints[i] != []):
				(center_x, center_y) = keypoints[i].pt # Getting keypoint's center
				offset = np.ceil(keypoints[i].size) # What's the size of the ellipse(major focus)
				tag = res[int(center_y - offset):int(center_y + offset), int(center_x - offset):int(center_x + offset)] # From Ellipse to Rectangular area.
				# Keeping track of location in the original image as well
				
		
			# Detecting Contours.
			contours = cv2.findContours(tag, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			contours = imutils.grab_contours(contours)
			contours = (sorted(contours, key = cv2.contourArea, reverse = True))[1:]

			# Finding Tag Corners both relative to the resized image and Original image
			for contour in contours:
				perimeter = cv2.arcLength(contour, True)
				approx = cv2.approxPolyDP(contour, 0.015*perimeter, True)
				if len(approx) == 4:
					tag_contour = approx
					tag_corner_relative = self.contours_to_corners(tag_contour)
					tag_corner_absolute = self._rel_to_abs(tag_corner_relative, center_x, center_y, offset)# Finding corners w.r.t original frame size
					tag_corner_absolute_list.append(tag_corner_absolute)
					warped = self.four_point_transform(tag, tag_corner_relative)
					warped_list.append(warped)
		return warped_list, tag_corner_absolute_list


	def _rel_to_abs(self, corners, x, y, offset):
		correction =  np.matmul(np.ones([4,2]), [[x,0],[0,y]]) -np.ones([4,2])*offset
		absolute_corners = (np.ceil(correction + corners)).astype(int)
		return absolute_corners


	def get_tag_id(self, tag):
		# @tag is the tag image of size 80x80
		# returns the tag id.
		
		cell_half_width = int(round(tag.shape[1]/16.0))# = 5 Pixels
		cell_half_height = int(round(tag.shape[0]/16.0))# = 5 Pixels
		
		row1 = cell_half_height*5
		col1 = cell_half_width*5
		row2 = cell_half_height*7
		col2 = cell_half_width*7
		row3 = cell_half_height*9
		col3 = cell_half_width*9
		row4 = cell_half_height*11
		col4 = cell_half_width*11
		
		# collecting center pixels from each cell(left to right, top to bottom_right)
		cells = []
		cells.append(tag[row1, col1])# Stores the center pixel from corresponding row and col1	
		cells.append(tag[row1, col2])
		cells.append(tag[row1, col3])
		cells.append(tag[row1, col4])

		cells.append(tag[row2, col1])
		cells.append(tag[row2, col2])
		cells.append(tag[row2, col3])
		cells.append(tag[row2, col4])

		cells.append(tag[row3, col1])
		cells.append(tag[row3, col2])
		cells.append(tag[row3, col3])
		cells.append(tag[row3, col4])

		cells.append(tag[row4, col1])
		cells.append(tag[row4, col2])
		cells.append(tag[row4, col3])
		cells.append(tag[row4, col4])

		for index, value in enumerate(cells):
			if value < 200:
				cells[index] = 0# Black Cells
			else:
				cells[index] = 1# White Cells.

		# Binary to decimal
		tag_id = 1*cells[5] + 2*cells[6] + 4*cells[9] + 8*cells[10]
		#print(tag_id)
		return tag, tag_id
		


	def place_template_on_tag(self, Image_1, Homography, Image_2, width, height):
		H_inv = np.linalg.inv(Homography)
		H_inv = H_inv/H_inv[2][2]
		for col in range(width[0], width[1]):
			for row in range(height[0], height[1]):
				pixel_homo_coord = np.array([col, row, 1]).T
				pixel_camera_coord = np.matmul(H_inv, pixel_homo_coord)
				pixel_camera_coord = ((pixel_camera_coord/pixel_camera_coord[2])).astype(int)
				if (pixel_camera_coord[0] < Image_2.shape[1]) and (pixel_camera_coord[1] < Image_2.shape[0]) and (pixel_camera_coord[0] >= 0) and (pixel_camera_coord[1] >= 0):
					Image_1[row, col] = Image_2[pixel_camera_coord[1], pixel_camera_coord[0]]
		return Image_1

	def put_a_cube(self, calib_mat, cam_pose, tag_corner, cube_corners, Image):
		# Getting Projection Matrix
		Proj_Matrix = np.matmul(calib_mat, cam_pose)
		Proj_Matrix = Proj_Matrix/Proj_Matrix[2,3]# 3x4 Matrix
		
		height = -200
		
		# Points on the Upper Surface of Cube - World Frame
		pt1 = np.array([cube_corners[0][0],cube_corners[0][1],height,1]).T
		pt2 = np.array([cube_corners[1][0],cube_corners[1][1],height,1]).T
		pt3 = np.array([cube_corners[2][0],cube_corners[2][1],height,1]).T
		pt4 = np.array([cube_corners[3][0],cube_corners[3][1],height,1]).T

		# Points on the the Upper Surface of the cube - Camera Frame
		pc1 = np.matmul(Proj_Matrix, pt1)
		pc2 = np.matmul(Proj_Matrix, pt2)
		pc3 = np.matmul(Proj_Matrix, pt3)
		pc4 = np.matmul(Proj_Matrix, pt4)


		# Drawing Lines between those points:
		
		# Drawing Lines between the points on the floor(tag_corner)
		cv2.line(Image, tuple(tag_corner[0]),tuple(tag_corner[1]),(255,0,0),4)
		cv2.line(Image, tuple(tag_corner[1]),tuple(tag_corner[2]),(255,0,0),4)
		cv2.line(Image, tuple(tag_corner[2]),tuple(tag_corner[3]),(255,0,0),4)
		cv2.line(Image, tuple(tag_corner[3]),tuple(tag_corner[0]),(255,0,0),4)
		
		# Drawing Vertical Lines of the Cube
		cv2.line(Image, tuple(tag_corner[0]),tuple([int(pc1[0,0]/pc1[0,2]),int(pc1[0,1]/pc1[0,2])]),(0,0,0),4)
		cv2.line(Image, tuple(tag_corner[1]),tuple([int(pc2[0,0]/pc2[0,2]),int(pc2[0,1]/pc2[0,2])]),(0,0,0),4)
		cv2.line(Image, tuple(tag_corner[2]),tuple([int(pc3[0,0]/pc3[0,2]),int(pc3[0,1]/pc3[0,2])]),(0,0,0),4)
		cv2.line(Image, tuple(tag_corner[3]),tuple([int(pc4[0,0]/pc4[0,2]),int(pc4[0,1]/pc4[0,2])]),(0,0,0),4)

		# Drawing Lines of on the Upper Surface.
		cv2.line(Image,tuple([int(pc1[0,0]/pc1[0,2]),int(pc1[0,1]/pc1[0,2])]),tuple([int(pc2[0,0]/pc2[0,2]),int(pc2[0,1]/pc2[0,2])]), (255, 0, 0), 4)
		cv2.line(Image,tuple([int(pc2[0,0]/pc2[0,2]),int(pc2[0,1]/pc2[0,2])]),tuple([int(pc3[0,0]/pc3[0,2]),int(pc3[0,1]/pc3[0,2])]), (255, 0, 0), 4)
		cv2.line(Image,tuple([int(pc3[0,0]/pc3[0,2]),int(pc3[0,1]/pc3[0,2])]),tuple([int(pc4[0,0]/pc4[0,2]),int(pc4[0,1]/pc4[0,2])]), (255, 0, 0), 4)
		cv2.line(Image,tuple([int(pc4[0,0]/pc4[0,2]),int(pc4[0,1]/pc4[0,2])]),tuple([int(pc1[0,0]/pc1[0,2]),int(pc1[0,1]/pc1[0,2])]), (255, 0, 0), 4)		

		return Image

































