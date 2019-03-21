
import cv2
import numpy as np

cap = cv2.imread('/Users/wangjie/Desktop/road1.jpg', 0)

while True:
    displayimage = cv2.imshow('road', cap)
    k=cv2.waitKey(5) &0xFF
    if k==27:
        break
cv2.destroyAllWindows()

cv2.line()