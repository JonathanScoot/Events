import cv2

mofangimg = cv2.imread('/Users/jonathan/Desktop/Jetimg/ball2.jpeg')   #读取图片

imgpixelB = mofangimg[100, 200, 0]
imgpixelG = mofangimg[100, 300, 1]
imgpixelR = mofangimg[200, 300, 2]

print(imgpixelR + imgpixelG)
print(imgpixelG)
print(imgpixelB)
