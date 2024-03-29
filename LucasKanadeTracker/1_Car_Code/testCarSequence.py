import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os
import cv2
import glob
import LucasKanade

if __name__ == '__main__':
    result_dir = '../data_enpm673/results/car/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    cars_data = glob.glob('../data_enpm673/car/*.jpg')
    cars_data.sort()
    frame = cv2.imread(cars_data[0])
    frame = cv2.resize(frame, (320, 240))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect_list = []
    rect = np.array([60, 60, 140, 120])
    rect_list.append(rect)
    for i in range(1, len(cars_data)):
        next_frame = cv2.imread(cars_data[i])
        next_frame = cv2.resize(next_frame, (320,240))
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        p = LucasKanade.LucasKanade(frame, next_frame, rect)

        rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
        rect_list.append(rect)

        # show the image
        tmp_img = next_frame.copy()
        cv2.rectangle(tmp_img, (int(round(rect[0])), int(round(rect[1]))), (int(round(rect[2])), int(round(rect[3]))), color=(0,255,255), thickness=2)
        cv2.imshow('image', tmp_img)
        cv2.waitKey(1)

        if i in range(len(cars_data)):
            cv2.imwrite(os.path.join(result_dir, 'car_frame_{}.jpg'.format(i)), tmp_img)

        frame = next_frame
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rect_list = np.array(rect_list)
    np.save(os.path.join('carseqrects.npy'), rect_list)
