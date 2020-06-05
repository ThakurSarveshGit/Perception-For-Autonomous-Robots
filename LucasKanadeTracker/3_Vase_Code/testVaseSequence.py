import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os
import cv2
import glob
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
    rect = np.array([125, 95, 170, 150])
    rect_list.append(rect)
    for i in range(1, len(cars_data)):
        try:
            next_frame = cv2.imread(cars_data[i])
            next_frame = cv2.resize(next_frame, (320,240))
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            p = LucasKanade.LucasKanade(frame, next_frame, rect)
            if p is None:
                p = np.zeros(2)

            rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
            rect_list.append(rect)

            # show the image
            tmp_img = next_frame.copy()
            cv2.rectangle(tmp_img, (int(round(rect[0])), int(round(rect[1]))), (int(round(rect[2])), int(round(rect[3]))), color=(0,255,255), thickness=1)
            cv2.imshow('image', tmp_img)
            cv2.waitKey(1)

            if i in range(len(cars_data)):
                cv2.imwrite(os.path.join(result_dir, 'vase_frame_{}.jpg'.format(i)), tmp_img)

            frame = next_frame
        except:
            print('exception raised in testVaseSequence')
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rect_list = np.array(rect_list)
    np.save(os.path.join('vaseseqrects.npy'), rect_list)
