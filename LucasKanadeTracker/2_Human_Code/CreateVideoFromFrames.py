import cv2
import glob

def create_video(path_frames, video_name, format_type=".avi"):
	out = cv2.VideoWriter(video_name, cv2.cv.FOURCC('M', 'J', 'P', 'G'), 20, (320,240))
	path_name = path_frames + '/*' + ".jpg"
	all_images = glob.glob(path_name)
	all_images.sort()

	for image_path in all_images:
		image = cv2.imread(image_path)
		out.write(image)

	# print("Video written to: {}".format(video_name))
	# print("Frames taken from: {}".format(path_name))
	out.release()

	print("Video Written!")
	return


if __name__ == "__main__":
	path_frames = 'data_enpm673/results/human'
	video_name = 'tracking_human.avi'
	create_video(path_frames, video_name)