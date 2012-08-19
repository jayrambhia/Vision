import cv2
import cv2.cv as cv
import numpy as np

class CAMShift:
    def __init__(self, source=None, bb=None):
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
            self.bb = (xmin,ymin,xmax-xmin,ymax-ymin)
            self.camshift()
        
    def camshift(self):
        self.imgs=[]
        while True:
            try:
                hsv = cv2.cvtColor(self.img, cv.CV_BGR2HSV)
                mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                x0, y0, w, h = self.bb
                x1 = x0 + w -1
                y1 = y0 + h -1
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
                hist_flat = hist.reshape(-1)
                self.imgs.append(hsv)
                prob = cv2.calcBackProject(self.imgs, [0], hist_flat, [0, 180], 1)
                prob &= mask
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                self.ellipse, self.bb = cv2.CamShift(prob, self.bb, term_crit)
                cv2.rectangle(self.img, (self.bb[0], self.bb[1]), (self.bb[0]+self.bb[2], self.bb[1]+self.bb[3]), color=(255,0,0))
                cv2.imshow("CAMShift", self.img)
                k = cv2.waitKey(30)
                if k==27:
                    cv2.destroyAllWindows()
                    break
                if k==114:
                    cv2.destroyAllWindows()
                    self.start()
                    break
                _, self.img = self.cam.read()
            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break
                
#c = CAMShift()

