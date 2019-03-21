import cv2
import numpy as np
import imutils
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=2, help="minimum area size")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)                      #打开摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)         #设置摄像头横向500个像素点
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)        #设置摄像头纵向500个像素点

#台球类
class ball:
    def __init__(self, lH=0, hH=0, lS=0, hS=0, lV=0, hV=0):
        self.lower = np.array([lH, lS, lV])
        self.upper = np.array([hH, hS, hV])         #每个球的HSV范围


#设置不同颜色台球HSV阈值用于提取颜色

blueball = ball(100, 124, 43, 255, 46, 255)      #可用
pinkball = ball(150, 200, 60, 170, 150, 255)     #可用
darkball = ball(0, 180, 0, 255, 0, 46)
yelloball = ball(25, 35, 43, 255, 205, 255)
greenball = ball(30, 90, 120, 255, 100, 180)
brownball = ball(0, 13, 63, 195, 140, 165)       #可用


while True:
    text = 'No ball exist'
    radius = None
    cntss = None
    ballradius = 40
    ret, frame = cap.read()                               #获取摄像头图像
    frame = imutils.resize(frame, width=500)              #调整
    frame = cv2.GaussianBlur(frame, (21, 21), 0)          #高斯模糊
    ballhsvimge = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  #将RGB转为HSV

    #设置不同颜色掩模
    bluemask = cv2.inRange(ballhsvimge, blueball.lower, blueball.upper)
    greenmask = cv2.inRange(ballhsvimge, greenball.lower, greenball.upper)
    pinkmask = cv2.inRange(ballhsvimge, pinkball.lower, pinkball.upper)
    darkmask = cv2.inRange(ballhsvimge, darkball.lower, darkball.upper)
    yellowmask = cv2.inRange(ballhsvimge, yelloball.lower, yelloball.upper)
    brownmask = cv2.inRange(ballhsvimge, brownball.lower, brownball.upper)

    mixmask = cv2.bitwise_or(bluemask, greenmask)
    mixmask = cv2.bitwise_or(mixmask, pinkmask)
    mixmask = cv2.bitwise_or(mixmask, darkmask)
    mixmask = cv2.bitwise_or(mixmask, yellowmask)
    mixmask = cv2.bitwise_or(mixmask, brownmask)



    #设置不同颜色显示
    '''
    blueres = cv2.bitwise_and(frame, frame, mask=bluemask)
    greenres = cv2.bitwise_and(frame, frame, mask=greenmask)
    pinkres = cv2.bitwise_and(frame, frame, mask=pinkmask)
    darkres = cv2.bitwise_and(frame, frame, mask=darkmask)
    yellowres = cv2.bitwise_and(frame, frame, mask=yellowmask)
    brownres = cv2.bitwise_and(frame, frame, mask=brownmask)
    '''
    im, cnts, hierarchy = cv2.findContours(greenmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  #找轮廓

    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue

        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        if radius >= ballradius:
            cv2.circle(frame, center, radius, (160, 110, 30), 2)
            text = (int(x), int(y))

    cv2.putText(frame, "ball's coordinate: {}, ball's radius :{}".format(text, radius), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    cv2.imshow('mixmask', mixmask)
    #显示
    '''
    cv2.imshow('blue', blueres)
    cv2.imshow('green', greenres)
    cv2.imshow('pink', pinkres)
    cv2.imshow('dark', darkres)
    cv2.imshow('yellow', yellowres)
    cv2.imshow('brown', brownres)
    cv2.imshow('frame', frame)
    '''

    #ESC键结束显示
    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break

#摧毁所有窗口
cv2.destroyAllWindows()