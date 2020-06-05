#!/usr/bin/env python
# coding: utf-8

# # Project 6 
# 
# * Shelly Bagchi
# * Sarvesh Thakur


import numpy as np
import cv2
import cv2.ml as ml

import os
import bisect
import matplotlib.pyplot as plt



# Which signs to detect
signs = ['00001', '00014', '00017', '00019', '00021', '00035', '00038', '00045'];
sign_ref = {}
print("Detecting only these signs:", signs)


bin_n = 16 # Number of bins for HOG



# --- Functions for HOG features and data reading ---

# See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html 
def hog(img):
    img = cv2.resize(img, (64,64))

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist


def get_data(path):
    subfolders = os.listdir(path)
    subfolders.sort()

    #global sign_ref

    data = np.empty((1,64), dtype=np.float32)
    labels = np.empty(1, dtype=np.int)

    for classID in signs:  # read limited signs as described in assignment
        # read only image files
        files = [f for f in os.listdir(os.path.join(path,classID)) if f.endswith('.ppm')]
        files.sort()
        for num,file in enumerate(files):
            img = cv2.imread(os.path.join(path, classID, file))
            #cv2.imshow("frame",img)

            # Save first img as reference
            if num==0:
                sign_ref[int(classID)] = cv2.resize(img, (64,64))

            # Should we preprocess image - denoise, increase contrast?

            # Extract HOG features
            features = hog(img)

            # Save data into format appropriate for cv2.ml.svm (ROW_SAMPLE)
            data = np.vstack((data, [features]))
            # Responses should be class ID for each row/col in sample
            labels = np.append(labels, int(classID))

        
    # Make sure to remove placeholder from array creation
    # And reshape for svm input
    data = (data[1:,:]).reshape((-1,64)).astype(np.float32)
    labels = (labels[1:]).reshape((-1,1)).astype(np.int)
    return data, labels



# --- Functions for MSER ---

def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    '''
        Matlab imadjust implementation in Python3
    '''
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2 ,'Input image should be 2-dims'  # One Channel Only!

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, 255): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst


# See https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
def sortContours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)



def findContours(img, mask, color=(0, 255, 0)):
    # Find contours & draw on frame
    # NOTE:  Tweak this threshold as needed!  *****
    _, thresh = cv2.threshold(mask.astype('uint8'), 0.6*255, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by size
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    box = (0,0,0,0)
    #cnts, boundingBoxes = sortContours(cnts)
    #for c,box in zip(cnts, boundingBoxes):
    for c in cnts[:1]:
        #print("contour size:", cv2.contourArea(c))

        # compute the center of the contour
        #M = cv2.moments(c)
        #cX = int(M["m10"] / M["m00"])
        #cY = int(M["m01"] / M["m00"])
 
        # draw the contour on the image
        #cv2.drawContours(img, [c], -1, color, 1)
        #cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)

        # Compute & draw the bounding box
        box = cv2.boundingRect(c)
        x,y,w,h = box
        cv2.rectangle(img, (x,y),(x+w,y+h), color, 2)

    return img, box



def runMSER(img):
    # Denoise image
    img = cv2.fastNlMeansDenoisingColored(img, None, 10,10,7,21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # only for display use
    # Split color channels
    #b,g,r = cv2.split(dst)
    b,g,r = img[:,:,0],img[:,:,1],img[:,:,2]
    # Contrast normalization
    b,g,r = cv2.equalizeHist(b),cv2.equalizeHist(g),cv2.equalizeHist(r)
    B,G,R = imadjust(b),imadjust(g),imadjust(r)

    # Normalize intensities for each type of sign
    # Note:  maximum and minimum work element-wise over an array 
    # *** Invert this??
    #res_b = 255*np.maximum(0, np.minimum(B-R, B-G) / R+G+B )
    #_,res_b = cv2.threshold(res_b, 1, 1, cv2.THRESH_BINARY)
    #res_r = 255*np.maximum(0, np.minimum(R-B, R-G) / R+G+B )
    #_,res_r = cv2.threshold(res_r, 1, 1, cv2.THRESH_BINARY)
    
# See paper:
# https://www.researchgate.net/publication/261331878_Traffic_sign_recognition_using_MSER_and_Random_Forests
    omega_rb = 255*np.maximum(R/(R+G+B), B/(R+G+B) )
    #_,omega_rb = cv2.threshold(omega_rb, 0.5, 255, cv2.THRESH_BINARY)

    
    #dst = findContours(img, res_b, color=(255, 0, 0))
    #dst = findContours(img, res_r, color=(0, 0, 255))
    dst,box = findContours(img.copy(), omega_rb, color=(0, 255, 0))
    
    # Save sign as crop of original img (**retval)
    if box == (0,0,0,0):
        sign = img
    else:
        x,y,w,h = box
        sign = img[y:y+h, x:x+w]
        #cv2.imshow("sign", sign)


    # Display imgs in separate windows
    #cv2.imshow("frame",dst)
    #cv2.imshow("frame",gray)
    #cv2.imshow("omega_rb", omega_rb.astype(np.uint8))

    # Display imgs side by side
    #displayResult1 = np.hstack((gray, omega_rb.astype(np.uint8)))
    #displayResult1 = cv2.resize(displayResult1, None, displayResult1.shape, 0.5,0.5)
    #cv2.imshow("frame, omega_rb", displayResult1)

    #plt.imshow(cv2.cvtColor(displayResult, cv2.COLOR_BGR2RGB))
    #plt.imshow(displayResult, cmap='gray')
    #plt.show()

    # Uncomment if you want to wait after every frame is shown
    #cv2.waitKey(0)
    # Press ESC key to exit
    #if cv2.waitKey(25) & 0xFF == 27:  # ESC
    #    return

    return dst, box, sign



try:
    dir = os.path.dirname(__file__)
except:
    dir = os.path.dirname('Project6_MSER.py')

# Data locations
train_path = os.path.join(dir, 'Training')
test_path = os.path.join(dir, 'Testing')
input_path = os.path.join(dir, 'input')  #####
output_path = os.path.join(dir, 'output')

# Set up classifier
svm = ml.SVM_create()
svm.setType(ml.SVM_C_SVC)
svm.setKernel(ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

# Read in training data
TRAIN = 1
if TRAIN:
    print("Reading training images...")
    # Extract features from training set using HOG
    train_data, train_labels = get_data(train_path)

    # Train SVM
    print("Training SVM using HOG features...")
    svm.train(train_data, ml.ROW_SAMPLE, train_labels)
    #svm.save(os.path.join(dir, 'svm_data.dat'))
    #print("Model saved to file:  svm_data.dat")

# Read in testing data
TEST = 0
if TEST:
    print("Reading test images...")
    # Extract features from testing set
    test_data, test_labels = get_data(test_path)

    # Run classifier prediction and compare to test labels
    print("Getting predictions from SVM...")
    # import results from previously created datafile, 'svm_data.dat'
    #print("Reading trained model from svm_data.dat...")
    #svm.load(os.path.join(dir, 'svm_data.dat'))
    result = svm.predict(test_data)[1]
    # Check accuracy of results
    mask = result==test_labels
    correct = np.count_nonzero(mask)
    print("Percent correct:", correct*100.0/result.size)


# Run sign detection on input frames
INPUT = 1
if INPUT:
    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(os.path.join(output_path, 'Project6FinalOutput'), fourcc, 20., (1628,1236));


    print("Identifying signs with MSER...")
    files = os.listdir(input_path)
    files.sort()
    for num,file in enumerate(files[:]):
        print("> Frame", num)
        img = cv2.imread(os.path.join(input_path, file))
        
        # Using MSER method to detect red and blue signs
        dst, box, sign = runMSER(img)

        if box==(0,0,0,0):
            print("No sign detected in frame")
        else:
            print("Sign detected!")
            # Extract HOG features
            features = hog(sign)
            # Run classifier prediction
            result = svm.predict(np.array([features], dtype=np.float32))[1]
            print("Class result is:", result)
            # Draw result on frame
            x,y,w,h = box
            #cv2.putText(dst, "Sign #{}".format(int(result[0,0])), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            try:
                dst[y:y+64,x-64:x] = sign_ref[int(result[0,0])]
            except:
                pass

            
        # Write frame to video
        video.write(dst);
        
        #cv2.imshow("frame",img)
        cv2.imshow("Frame",img)
        # cv2.imshow("sign",sign)
    
        #Press ESC key to exit
        if cv2.waitKey(25) & 0xFF == 27:  # ESC
           break

    
    print("Finished execution.");
    print("Find video file output.avi in output folder.");
    video.release()
