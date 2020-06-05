import cv2
import glob

def store_100_frames(path_to_video):
    # Video has around 200 frames only.
    read_video = cv2.VideoCapture(path_to_video)
    
    count = 0
    name_index = 1
    while True:
        ret, frame = read_video.read()
        if ret == True:
            if count%2 == 0 and name_index <=70:
                name = '100Frames/train/frame' + str(name_index) + '.png'
                cv2.imwrite(name, frame)
                name_index += 1
            elif count%2 == 0 and name_index>=70:
                name = '100Frames/test/frame' + str(name_index-70) + '.png'
                cv2.imwrite(name, frame)
                name_index += 1
            else:
                pass
        else:
            break
        
        count += 1
    return

# store_100_frames("detectbuoy.avi")

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)


# load the image, clone it, and setup the mouse callback function
images = glob.glob("100Frames/train/*.png")
images.sort()
index = 1
for image in images:
	image = cv2.imread(image)
	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)
	 
	# keep looping until the 'q' key is pressed
	while True:
		# display the image and wait for a keypress
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the 'r' key is pressed, reset the cropping region
		if key == ord("r"):
			image = clone.copy()
	 
		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			break

		elif key == ord("n"):
			break
	 
	# if there are two reference points, then crop the region of interest
	# from teh image and display it
	if len(refPt) == 2:
		roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		name = "100Frames/train/orange/frame" + str(index) +'.png'
		index +=1
		cv2.imwrite(name, roi)
	# 	cv2.imshow("ROI", roi)
	# 	cv2.waitKey(0)
	 
	# # close all open windows
	# cv2.destroyAllWindows()

