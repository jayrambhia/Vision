import cv2
import cv2.cv as cv
import numpy as np

def sobel(filename, gray=True, dx=1, dy=1, ksize=3):
    if gray:
        img = cv2.imread(filename, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        img = cv2.imread(filename)
    dst = cv2.Sobel(img, ddepth=cv.CV_32F, dx=dx, dy=dy, ksize=ksize)
    minv = np.min(dst)
    maxv = np.max(dst)
    cscale = 255/(maxv-minv)
    shift =  -1*(minv)
    t = np.zeros(img.shape,dtype='uint8')
    t = cv2.convertScaleAbs(dst,t,cscale,shift/255.0)
    return t

