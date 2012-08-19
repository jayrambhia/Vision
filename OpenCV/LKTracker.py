import cv2
import cv2.cv as cv
import numpy as np
import itertools

lk_params = dict( winSize  = (10, 10), 
                  maxLevel = 10, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))   

feature_params = dict( maxCorners = 4000, 
                       qualityLevel = 0.6,
                       minDistance = 2,
                       blockSize = 2)

class LK:
    def __init__(self, lk_params, feature_params, source=None, bb=None):
        self.lk_params = lk_params
        self.feature_params = feature_params
        self.mouse_p1 = None
        self.mouse_p2 = None
        self.mouse_drag = False
        self.bb = None
        self.img = None
        if source:
            self.cam = cv2.VideoCapture(source)
        else:
            self.cam = cv2.VideoCapture(0)
        if not bb:
            self.start()
        else:
            self.bb = bb
            _, self.img = self.cam.read()
            self.lk()
        
    def start(self):
        _, self.img = self.cam.read()
        cv2.imshow("img", self.img)
        cv.SetMouseCallback("img", self.__mouseHandler, None)
        if not self.bb:
            _, self.img = self.cam.read()
            cv2.imshow("img", self.img)
            cv2.waitKey(30)
        cv2.waitKey(0)
    
    def __mouseHandler(self, event, x, y, flags, params):
        _, self.img = self.cam.read()
        if event == cv.CV_EVENT_LBUTTONDOWN and not self.mouse_drag:
            self.mouse_p1 = (x, y)
            self.mouse_drag = True
        elif event == cv.CV_EVENT_MOUSEMOVE and self.mouse_drag:
            cv2.rectangle(self.img, self.mouse_p1, (x, y), (255, 0, 0), 1, 8, 0)
        elif event == cv.CV_EVENT_LBUTTONUP and self.mouse_drag:
            self.mouse_p2 = (x, y)
            self.mouse_drag=False
        cv2.imshow("img",self.img)
        cv2.waitKey(30)
        if self.mouse_p1 and self.mouse_p2:
            cv2.destroyWindow("img")
            xmax = max((self.mouse_p1[0],self.mouse_p2[0]))
            xmin = min((self.mouse_p1[0],self.mouse_p2[0]))
            ymax = max((self.mouse_p1[1],self.mouse_p2[1]))
            ymin = min((self.mouse_p1[1],self.mouse_p2[1]))
            self.bb = [xmin,ymin,xmax-xmin,ymax-ymin]
            self.lk()
    
    def lk(self):
        bb = self.bb
        oldg = cv2.cvtColor(self.img, cv2.cv.CV_BGR2GRAY)
        old_pts = None
        while True:
            try:
                _, img = self.cam.read()
                img1 = img[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2]]
                g = cv2.cvtColor(img1, cv2.cv.CV_BGR2GRAY)
                pt = cv2.goodFeaturesToTrack(g, **self.feature_params)
                if type(pt) == type(None):
                    cv2.imshow("img", img)
                    cv2.waitKey(30)
                    oldg = newg
                    _, img = self.cam.read()
                    newg = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
                    continue
                for i in xrange(len(pt)):
                    pt[i][0][0] = pt[i][0][0]+bb[0]
                    pt[i][0][1] = pt[i][0][1]+bb[1]
                newg = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
                p0 = np.float32(pt).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(oldg, newg, p0, None, **self.lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(newg, oldg, p1, None, **self.lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_pts = []
                for pts, val in itertools.izip(p1, good):
                    if val:
                        new_pts.append([pts[0][0], pts[0][1]])
                        cv2.circle(img, (pts[0][0], pts[0][1]), 2, thickness=2, color=(255,255,0))
                bb = self.__predictBB(bb, old_pts, new_pts)
                print bb
                if bb[0]+bb[2] >= img.shape[1]:
                    bb[0] = img.shape[1] - bb[2] - 10
                if bb[1]+bb[3] >= img.shape[0]:
                    bb[1] = img.shape[0] - bb[3] - 10
                old_pts = new_pts
                oldg = newg
                cv2.rectangle(img, (bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]), color=(255,0,0))
                cv2.imshow("img", img)
                k=cv2.waitKey(30)
                if k==27:
                    cv2.destroyAllWindows()
                    break
            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break
                
    def __predictBB(self, bb0, pt0, pt1):
        if not pt0:
            pt0 = pt1
        dx=[]
        dy=[]
        for p1, p2 in itertools.izip(pt0, pt1):
            dx.append(p2[0]-p1[0])
            dy.append(p2[1]-p1[1])
        if not dx or not dy:
            return bb0
        cen_dx = round(sum(dx)/len(dx))/2
        cen_dy = round(sum(dy)/len(dy))/2
        bb = [int(bb0[0]+cen_dx), int(bb0[1]+cen_dy), bb0[2], bb0[3]]
        if bb[0] <= 0:
            bb[0] = 10
        if bb[1] <= 0:
            bb[1] = 10
        return bb
LK(lk_params, feature_params)
