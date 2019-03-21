import  imutils
import  cv2

img = cv2.imread('/Users/wangjie/Desktop/bike.jpeg')
img = imutils.resize(img, width=500)
while True:

    imggary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cv2.imshow('imggary', imggary)
    cv2.imshow('img', img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()