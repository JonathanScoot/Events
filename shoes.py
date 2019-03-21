import cv2
import imutils
import numpy as np

number = 0
count = 0


def nothing():
    pass


cap = cv2.VideoCapture(0)

while True:

    '''
    im.set(cv2.cv2.CAP_PROP_FRAME_WIDTH, 500)
    im.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT, 500)
    '''
    ret, frame = cap.read()
    img = cv2.resize(frame, (500, 500))
    img = cv2.GaussianBlur(img, (21, 21), 2)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    yuzhi = 145
    ret, thresh = cv2.threshold(imgray, yuzhi, 255, 0)
    # adatpthresh2 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    mediabluthresh = cv2.medianBlur(thresh, 7)
    edge = cv2.Canny(mediabluthresh, 100, 255)
    image, cnts, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
    colorzuobiao = []
    zuobiao = [0, 0]

    # 寻找轮廓改变原始图像，按照设置的像素间隔在绿色轮廓中寻找红点
    for i in range(0, 495, 10):
        for j in range(495):
            if img.item(i, j, 1) == 255 and img.item(i, j, 0) == 0 and img.item(i, j, 2) == 0:
                img.itemset((i, j, 0), 0)
                img.itemset((i, j, 1), 0)
                img.itemset((i, j, 2), 255)
                zuobiao = [i, j]
                colorzuobiao.append(zuobiao)
                count = count + 1

    number = number + 1
    if number == 1:
        print(colorzuobiao)
        print(count)
    # 图像显示
    cv2.imshow('img', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

'''
    for i in range(0, 499, 5):
        for j in range(500):
            if [i, j] in contuorlist:
                print([i,j])


    #adatpedge = cv2.Canny(adatpthresh2, 100, 255)

    cv2.imshow('thresh', thresh)
    cv2.imshow('mdediathresh', mediabluthresh)
    #cv2.imshow('adatpthresh2', adatpthresh2)
    #cv2.imshow('adatpedge', adatpedge)
    cv2.imshow('edge', edge)
    cv2.imshow('image',image)
    cv2.imshow('img', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
'''
