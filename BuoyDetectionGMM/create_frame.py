import cv2

video_path = cv2.VideoCapture('Part0.mp4')
frame_number = 0
while True:
	ret,frame = cap.read()
	name = 'result_frame_' + str(frame_number) + '.jpg'
	cv2.imwrite(name, frame)

	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

video_path.release()
cv2.destroyAllWindows()