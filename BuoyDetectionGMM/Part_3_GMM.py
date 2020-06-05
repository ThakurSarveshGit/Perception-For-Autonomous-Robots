import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn
from Part_2_EM import *


########################################################################################
########################################################################################
#Set Train = 1 for training
Train = 1
K=7

#Set values here
testfolder = (glob.glob("100Frames/test/finaltest/*.png")).sort()
outfolder = "EM Output"
########################################################################################
########################################################################################

# Simple Gaussian
def gauss_function(x, amp, x0, sigma):
	return amp*np.exp(-(x-x0)**2./(2.*sigma**2))

def gaussian_pdf(x, mu, sigma):
	y = np.power(10,15)*(1/(np.sqrt(2*np.pi*np.power(sigma,2))))*(np.power(np.e, -(np.power((x-mu),2)/(2*np.power(sigma,2)))))
	return np.array(y)

def im2double(image):
	info = np.iinfo(image.dtype)
	return image.astype(np.float)/info.max

def morph_image(image, iterations=10):
	# performs binary closure(dilation formed by erosion) for iter iterations
	clone = image.copy()
	kernel = np.ones((5,5),np.uint8)
	for i in range(iterations):
		dilation = cv2.dilate(clone,kernel,iterations = 1)
		erosion = cv2.erode(dilation, kernel, iterations = 1)
		clone = erosion.copy()
	return clone

def img_grey(data):
    return Image.fromarray(data * 255, mode='L').convert('1')

def stats_of_channel(channel_pixels_from_all_images):
	'''
	return means, variance and standard deviation vector.
	'''
	mean = np.mean(channel_pixels_from_all_images)
	variance = np.var(channel_pixels_from_all_images)
	standard_deviation = np.sqrt(variance)
	return mean, variance, standard_deviation

orange_images = np.load("100Frames/train/orange/all_orange.npy")
orange_buoy_pixels_R = orange_images[:,2:3]# Contains R channel value for orange buoy
orange_buoy_pixels_G = orange_images[:,1:2]# and so on...
orange_buoy_pixels_B = orange_images[:,0:1]

green_images = np.load("100Frames/train/green/all_green.npy")
green_buoy_pixels_R = green_images[:,2:3]
green_buoy_pixels_G = green_images[:,1:2]
green_buoy_pixels_B = green_images[:,0:1]

yellow_images = np.load("100Frames/train/yellow/all_yellow.npy")
yellow_buoy_pixels_R = yellow_images[:,2:3]
yellow_buoy_pixels_G = yellow_images[:,1:2]
yellow_buoy_pixels_B = yellow_images[:,0:1]

[orange_B_mean, orange_B_var, orange_B_sd] = stats_of_channel(orange_buoy_pixels_B)
[orange_G_mean, orange_G_var, orange_G_sd] = stats_of_channel(orange_buoy_pixels_G)
[orange_R_mean, orange_R_var, orange_R_sd] = stats_of_channel(orange_buoy_pixels_R)

[green_B_mean, green_B_var, green_B_sd] = stats_of_channel(green_buoy_pixels_B)
[green_G_mean, green_G_var, green_G_sd] = stats_of_channel(green_buoy_pixels_G)
[green_R_mean, green_R_var, green_R_sd] = stats_of_channel(green_buoy_pixels_R)

[yellow_B_mean, yellow_B_var, yellow_B_sd] = stats_of_channel(yellow_buoy_pixels_B)
[yellow_G_mean, yellow_G_var, yellow_G_sd] = stats_of_channel(yellow_buoy_pixels_G)
[yellow_R_mean, yellow_R_var, yellow_R_sd] = stats_of_channel(yellow_buoy_pixels_R)



# Loading the test and training datas.
test_images = (glob.glob("100Frames/test/finaltest/*.png")).sort()
train_images = (glob.glob("100Frames/test/finaltest/*.png")).sort()


# Loading the cropped pixel information for each buoy.
orange_images = np.load("100Frames/train/orange/all_orange.npy")
orange_buoy_pixels_R = orange_images[:,2:3]# Contains R channel value for orange buoy
orange_buoy_pixels_G = orange_images[:,1:2]# and so on...
orange_buoy_pixels_B = orange_images[:,0:1]

green_images = np.load("100Frames/train/green/all_green.npy")
green_buoy_pixels_R = green_images[:,2:3]
green_buoy_pixels_G = green_images[:,1:2]
green_buoy_pixels_B = green_images[:,0:1]

yellow_images = np.load("100Frames/train/yellow/all_yellow.npy")
yellow_buoy_pixels_R = yellow_images[:,2:3]
yellow_buoy_pixels_G = yellow_images[:,1:2]
yellow_buoy_pixels_B = yellow_images[:,0:1]


# Creating Data Mixture that we will be applying our GMM on
pixel_mixture = orange_buoy_pixels_R.copy()
pixel_mixture = np.append(pixel_mixture, green_buoy_pixels_G)
pixel_mixture = np.append(pixel_mixture, yellow_buoy_pixels_G)
pixel_mixture = np.append(pixel_mixture, yellow_buoy_pixels_R)


# Fitting a GMM of 3 Gaussians
gmm = GaussianMixture(n_components=3, covariance_type="full", tol=0.001)
gmm = gmm.fit(X = np.expand_dims(pixel_mixture, 1))

# Evaluate GMM
gmm_x = np.linspace(0,255,255)
gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1,1)))

# Make regular Histogram
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8,5])
ax.hist(pixel_mixture, bins=50, normed=True, alpha=0.5, color="#0070FF")
ax.plot(gmm_x, gmm_y, color='crimson', lw=4, label="GMM")

# Annotate diagrams
ax.set_ylabel('Probability Density')
ax.set_xlabel('Pixel Value')

plt.legend()

# plt.show()


print(gmm.means_)
print(gmm.covariances_)
# Iterating through each test image
test_images_path = glob.glob("100Frames/test/*.png")
count_test_image = 0

test_images_path = glob.glob("100Frames/test/*.png")
binary_images_path = "Results_Part1/.Binary_Images/"

count_test_image = 0
for image_path in test_images_path:
	count_test_image += 1
	print("Reading test image {}".format(count_test_image))
	image = cv2.imread(image_path)
	image = cv2.GaussianBlur(image, (5,5), 0)# Gaussian Blurring to reduce noise.
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# Extracting R, G and B planes from the test image
	B = im2double(image[:,:,0])
	G = im2double(image[:,:,1])
	R = im2double(image[:,:,2])

	# A 2D array for storing the probability for each pixel in the image.
	# probO stores the probability of a pixel being in orange buoy
	probO = np.zeros((image.shape[0], image.shape[1]))
	probY = np.zeros((image.shape[0], image.shape[1]))
	probG = np.zeros((image.shape[0], image.shape[1]))

	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			r = R[i,j]
			g = G[i,j]
			b = B[i,j]

			# Storing prob. of a pixel to be in orange buoy region based on gaussian defined
			# based on R channel intensity only.
			probO[i,j] = gaussian_pdf(r, orange_R_mean, orange_R_sd)
			
			# Storing prob. of a pixel to be in green buoy region based on gaussian defined
			# based on G channel intensity only.
			probG[i,j] = gaussian_pdf(g, green_G_mean, green_G_sd)

			# Storing prob. of a pixel to be in yellow buoy region based on gaussian defined
			# based on Green and Red channels intensities.
			probY[i,j] = gaussian_pdf(b, (yellow_R_mean+yellow_G_mean)/2, (yellow_R_sd+yellow_G_sd)/2)



	# Classifying Pixel locations.
	probY = probY/np.linalg.norm(probY)
	probG = probG/np.linalg.norm(probG)
	probO = probO/np.linalg.norm(probO)

	####
	# 	Yellow
	# std: 0.00013913947087200373
	# mean_min, mean_max: (0.0013517204647477737, 0.002862990590755389)

	# Green
	# 4.287537593498933e-05
	# (0.0016843979241664728, 0.001985788738100027)
	
	# Orange
	# 0.00037531415793453087
	# (0.0005805714007867209, 0.003371176673615435)
	# print('yellow')
	# print(np.max(probY), np.min(probY))
	yellow_1 = probY*1000 < 2.5# Tuning for Yellow Buoy
	yellow_2 = probY*1000 > 1.8# Tuning for Yellow Buoy
	yellow = np.multiply(yellow_1,yellow_2) # Taking and 
	yellow = yellow.astype(np.uint8)
	yellow *= 255
	# print('orange')
	# print(np.max(probO), np.min(probO))
	orange = probO*1000 > 3 # Tuning for Orange Buoy
	orange = orange.astype(np.uint8)
	orange *= 255
	# print('green')
	# print(np.max(probG), np.min(probG))
	green = probG*1000 > 1.8 # Tuning for Green Buoy
	green = green.astype(np.uint8)
	green *= 255

	# Binary Images.
	orange_image = morph_image(orange, iterations=10)
	yellow_image = morph_image(yellow, iterations=10)
	green_image = morph_image(green, iterations=10)

	binary_image = orange_image + yellow_image + green_image

	# Saving Binary Images. Run Once
	name = binary_images_path + 'binary_' + str(count_test_image) + '.jpg'
	
	# Morphological_operations to get better images.
	

	cv2.imwrite(name, binary_image)

# print(np.max(prob), np.min(prob), np.std(prob))