import cv2
import numpy as np

class SignDetector:
    # stale z zakresami dla konkretnych kolorów znaków
    A_FAMILY = {'lower': [3, 30, 30], 'upper': [38, 255, 255]}

    def __init__(self, filename):
        self.img = cv2.imread(filename)
        # self.img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        return str(len(approx)), approx

    def find(self, family_type):
        upper = np.array(family_type['upper'])
        lower = np.array(family_type['lower'])
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower, upper)
        res = cv2.bitwise_and(self.img, self.img, mask=mask)
        ret,res = cv2.threshold(res,7,255,cv2.THRESH_BINARY)

        img = cv2.GaussianBlur(res, (3,3), 0)
 
        laplacian = cv2.Canny(img, 100, 500)
        i,contours,z = cv2.findContours(laplacian, cv2.RETR_TREE, 1)
        
        for c in contours:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
            M = cv2.moments(c)
            if not M["m00"]:
                M["m00"] = 0.1
            cX = int((M["m10"] / M["m00"]) )
            cY = int((M["m01"] / M["m00"]) )
            shape, approx = self.detect(c)

            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            c = c.astype("int")
            if len(c)>100 and len(approx) > 2 and int(shape) < 10:
                cv2.rectangle(self.img, (cX-50, cY-50),(cX+50, cY+50), (0,0,200), 2 )
                cv2.drawContours(self.img, [approx], -1, (0, 255, 0), 2)
                cv2.putText(self.img, shape, (cX, cY+10), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 255), 1)


        cv2.imshow('frame', laplacian)
        cv2.imshow('img', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





if __name__ == '__main__':
    SignDetector('test4.png').find(SignDetector.A_FAMILY)