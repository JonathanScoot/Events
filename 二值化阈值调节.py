import cv2
import imutils

img = cv2.imread('/Users/jonathan/Desktop/自动涂胶机图像/shoe.JPG')
img = imutils.resize(img,width=500, height=500)
img = cv2.GaussianBlur(img, (21, 21), 2)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def nothing():
    pass

cv2.namedWindow('image')
cv2.createTrackbar('Yuzhi', 'image', 0, 255, nothing)

while True:

    yuzhi = cv2.getTrackbarPos('Yuzhi', 'image')

    ret, thresh = cv2.threshold(imgray, yuzhi, 255, 0)

    #adatpthresh2 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    mediabluthresh = cv2.medianBlur(thresh,7)

    edge = cv2.Canny(mediabluthresh, 100, 255)

    image, cnts, hierarchy = cv2.findContours(edge, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contoursimg = cv2.drawContours(edge, cnts, 3, (0, 255, 0), 3)

     #adatpedge = cv2.Canny(adatpthresh2, 100, 255)

    cv2.imshow('Original', img)
    cv2.imshow('thresh', thresh)
    cv2.imshow('mdediathresh', mediabluthresh)
    #cv2.imshow('adatpthresh2', adatpthresh2)
    #cv2.imshow('adatpedge', adatpedge)
    cv2.imshow('edge', edge)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
