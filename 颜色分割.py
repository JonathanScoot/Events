
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

class ball:
    def __init__(self, hh=0, hs=0, hv=0, lh=0, ls=0, lv=0):
        self.lower = np.array([hh, hs, hv])
        self.upper = np.array([lh, ls, lv])

darkball = ball(0, 0, 0, 100, 100, 100)



while True:
    ret, frame = cap.read()

    hsvimge = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 43, 46])
    upper_black = np.array([10, 255, 255])

    mask = cv2.inRange(hsvimge, lower_black, upper_black)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('res', res)
    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()



