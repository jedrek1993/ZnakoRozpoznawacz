import cv2
import numpy as np

class ShapeDetector:
    def __init__(self):
        pass
 
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


def main():
    img = cv2.imread('znak.png')
    img = cv2.GaussianBlur(img,(11,11),0)

    upper = np.array([180, 220, 220])
    lower = np.array([170, 0, 0])

    upper1 = np.array([25, 220, 220])
    lower1 = np.array([0, 100, 0])

    upper2 = np.array([179, 20, 255])
    lower2 = np.array([0, 0, 130])
    

    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(h, lower, upper)
    mask1 = cv2.inRange(h, lower1, upper1)

    mask += mask1

    # mask = cv2.inRange(h, lower2, upper2)

    res = cv2.bitwise_and(img,img, mask= mask)

    laplacian = cv2.Canny(res, 100, 200)
    i,contours,z = cv2.findContours(laplacian, cv2.RETR_TREE, 2)
    sd = ShapeDetector()

    for c in contours:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        if not M["m00"]:
            M["m00"] = 0.1
        cX = int((M["m10"] / M["m00"]) )
        cY = int((M["m01"] / M["m00"]) )
        shape, approx = sd.detect(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c = c.astype("int")
        if len(c)>50 and len(approx) > 2 and shape == '3':
            cv2.rectangle(img, (cX-50, cY-50),(cX+50, cY+50), (0,0,200), 2 )
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
            cv2.putText(img, shape, (cX, cY+10), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 255, 255), 1)

    cv2.imshow('res', res)
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def edge():
    img = cv2.imread('znak.png')
    img = cv2.GaussianBlur(img,(5,5),0)
    h = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    

    laplacian = cv2.Canny(h, 100, 200)

    i,contours,z = cv2.findContours(laplacian, cv2.RETR_TREE, 2)
    sd = ShapeDetector()


    for c in contours:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        if not M["m00"]:
            M["m00"] = 0.1
        cX = int((M["m10"] / M["m00"]) )
        cY = int((M["m01"] / M["m00"]) )
        shape, approx = sd.detect(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c = c.astype("int")
        if len(c)>50 and len(approx) > 2:
            cv2.rectangle(img, (cX-32, cY-32),(cX+32, cY+32), (0,0,200), 2 )
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
            cv2.putText(img, shape, (cX, cY+10), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 255, 255), 1)
                    

    final = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    avg_a = np.average(final[:, :, 1])
    avg_b = np.average(final[:, :, 2])

    for x in range(final.shape[0]):
        for y in range(final.shape[1]):
            l, a, b = final[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            final[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            final[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)

    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)


    cv2.imshow('img', img)
    cv2.imshow('frame', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def circle():
    img = cv2.imread('znak.png')
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Canny(cimg, 100, 200)

    circles = cv2.HoughCircles(laplacian,cv2.HOUGH_GRADIENT,1.2,100, minRadius=20,maxRadius=255)

    circles = np.uint16(np.around(circles))
    for i in circles[0,-20:]:
        # draw the outer circle
        cv2.circle(laplacian,(i[0],i[1]),i[2],255,2)
        # draw the center of the circle
        cv2.circle(laplacian,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('img', img)
    cv2.imshow('frame', laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    edge()