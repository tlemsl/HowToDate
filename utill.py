import cv2
import numpy as np
import matplotlib.pylab as plt
import time
import pyautogui
from PIL import ImageGrab

class SimilarityChecker:
    def __init__(self, normal_img, bad_img, threshold = 0.001):
        self._normal_img = normal_img
        self._bad_img = bad_img
        self._normal_hist = self._histogramization(self._normal_img)
        self._bad_hist = self._histogramization(self._bad_img)
        self._threshold= threshold

    def _histogramization(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    
    def ending_check(self, img):
        if isinstance(img, tuple):
            img = tuple2numpy(img)
        type = 0 # 0 -> not an ending, 1-> bad ending, 2->normal ending 
        hist = self._histogramization(img)

        if cv2.compareHist(hist, self._bad_hist, 4) < self._threshold:
            type = 1
        elif cv2.compareHist(hist, self._normal_hist, 4) < self._threshold:
            type = 2
        return type
    
    def check(self, img1, img2):
        imgs = [img1, img2]
        hists = []
        for img in imgs:
            start_time = time.time()
            if isinstance(img, tuple):
                img = tuple2numpy(img)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            hists.append(hist)
        ret = cv2.compareHist(hists[0], hists[1], 4)

        return ret < self._threshold, ret 
            


    


def numpy2tuple(array):
    if isinstance(array, np.ndarray):
        return tuple(numpy2tuple(sub_array) for sub_array in array)
    else:
        return array

def tuple2numpy(tpl):
    if isinstance(tpl, tuple):
        return np.array([tuple2numpy(sub_tpl) for sub_tpl in tpl])
    else:
        return tpl

def PIL2OpenCV(pil_image):
    numpy_image= np.array(pil_image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def reset_position():
    #love choice click
    while True:
        fore=pyautogui.getActiveWindow()
        print(fore.title)
        if fore.title=="LoveChoice":
            break
        time.sleep(1)
    print(fore.size)
    img=ImageGrab.grab((fore.left+8, fore.top+31, fore.right-8, fore.bottom-8))
    return [fore.left+8, fore.top+31, fore.right-8, fore.bottom-8], img
    
def cal_ending(img):
    img_arr=np.array(img) #튜플인지 확인
    R=0
    for i in range(len(img_arr)):
        for j in range(len(img_arr[0])):
            R+=img_arr[i][j][0]

    R=R/(len(img_arr)*len(img_arr[0]))
    print(R)
    if R>244:
        return 20
    else: 
        return 10

