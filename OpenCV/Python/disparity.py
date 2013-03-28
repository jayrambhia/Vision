import cv2
import cv2.cv as cv
import sys
import numpy as np

def getDisparity(imgLeft, imgRight, method="BM"):

    gray_left = cv2.cvtColor(imgLeft, cv.CV_BGR2GRAY)
    gray_right = cv2.cvtColor(imgRight, cv.CV_BGR2GRAY)
    print gray_left.shape
    c, r = gray_left.shape
    if method == "BM":
        sbm = cv.CreateStereoBMState()
        disparity = cv.CreateMat(c, r, cv.CV_32F)
        sbm.SADWindowSize = 9
        sbm.preFilterType = 1
        sbm.preFilterSize = 5
        sbm.preFilterCap = 61
        sbm.minDisparity = -39
        sbm.numberOfDisparities = 112
        sbm.textureThreshold = 507
        sbm.uniquenessRatio= 0
        sbm.speckleRange = 8
        sbm.speckleWindowSize = 0

        gray_left = cv.fromarray(gray_left)
        gray_right = cv.fromarray(gray_right)

        cv.FindStereoCorrespondenceBM(gray_left, gray_right, disparity, sbm)
        disparity_visual = cv.CreateMat(c, r, cv.CV_8U)
        cv.Normalize(disparity, disparity_visual, 0, 255, cv.CV_MINMAX)
        disparity_visual = np.array(disparity_visual)

    elif method == "SGBM":
        sbm = cv2.StereoSGBM()
        sbm.SADWindowSize = 9;
        sbm.numberOfDisparities = 96;
        sbm.preFilterCap = 63;
        sbm.minDisparity = -21;
        sbm.uniquenessRatio = 7;
        sbm.speckleWindowSize = 0;
        sbm.speckleRange = 8;
        sbm.disp12MaxDiff = 1;
        sbm.fullDP = False;

        disparity = sbm.compute(gray_left, gray_right)
        disparity_visual = cv2.normalize(disparity, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8U)

    return disparity_visual

imgLeft = cv2.imread(sys.argv[1])
imgRight = cv2.imread(sys.argv[2])
try:
    method = sys.argv[3]
except IndexError:
    method = "BM"

disparity = getDisparity(imgLeft, imgRight, method)
cv2.imshow("disparity", disparity)
cv2.imshow("left", imgLeft)
cv2.imshow("right", imgRight)
cv2.waitKey(0)