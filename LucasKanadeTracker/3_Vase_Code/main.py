import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os
import cv2
import glob
from CreateVideoFromFrames import *

import LucasKanade

if __name__ == '__main__':
    result_dir = '../data_enpm673/results/vase/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    cars_data = glob.glob('../data_enpm673/vase/*.jpg')
    cars_data.sort()
    frame = cv2.imread(cars_data[0])
    frame = cv2.resize(frame, (320, 240))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect_list = []
    rect = np.array([125, 95, 170, 150])# Hard Coded
    rect_list.append(rect)
    pre_p = None
    update_th = 0.5

    # get the rect without template correction
    for i in range(1, len(cars_data)):
        # ori_rects = np.load("vaseseqrects.npy")
        next_frame = cv2.imread(cars_data[i])
        next_frame = cv2.resize(next_frame, (320,240))
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        p = LucasKanade.LucasKanade(frame, next_frame, rect)

        rect_new = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
        rect_list.append(rect_new)

        # show the image
        tmp_img = next_frame.copy()
        copy_temp_img = tmp_img.copy()
        copy_temp_img = cv2.cvtColor(copy_temp_img, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(copy_temp_img, (125, 95), (170, 150),color=(0,255,0), thickness=1)
        cv2.rectangle(copy_temp_img, (int(round(rect_new[0])), int(round(rect_new[1]))), (int(round(rect_new[2])), int(round(rect_new[3]))),
                      color=(0, 255, 255), thickness=1)
        cv2.imshow('image', copy_temp_img)
        cv2.waitKey(1)

        if i in range(len(cars_data)):
            cv2.imwrite(os.path.join(result_dir, 'vase_frame_corrected_{}.jpg'.format(i)), copy_temp_img)

        # decide whether to update the template
        if pre_p is None:
            frame = next_frame
            rect = rect_new
            pre_p = np.copy(p)
        elif abs(np.sqrt(np.sum((p-pre_p)**2))) < update_th:
            frame = next_frame
            rect = rect_new
            pre_p = np.copy(p)

    rect_list = np.array(rect_list)
    np.save('vaseseqrects-wcrt.npy', rect_list)

    path_frames = '../data_enpm673/results/vase'
    video_name = 'tracking_vase.avi'
    create_video(path_frames, video_name)
