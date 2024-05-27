import cv2
import numpy as np
import matplotlib.pylab as plt

def similiaity(img1, img2):
    imgs = [img1, img2]
    hists = []
    for img in imgs:
        if isinstance(img, tuple):
            img = tuple2numpy(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        hists.append(hist)
    return cv2.compareHist(hists[0], hists[1], 4) 


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