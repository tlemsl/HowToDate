import cv2
import numpy as np
import matplotlib.pylab as plt

def similiaity(img1, img2):
    imgs = [img1, img2]
    hists = []
    for img in imgs:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        hists.append(hist)
    return cv2.compareHist(hists[0], hists[1], 4) 
