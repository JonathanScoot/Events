import cv2
import numpy as np
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=2, help="minimum area size")
args = vars(ap.parse_args())

mofangimg = cv2.imread('/Users/jonathan/Desktop/ImageForJet/sinuoke001.jpeg')   #读取图片
mofangimg = imutils.resize(mofangimg,500,500)
#设置图片
gaussmofangimg = cv2.GaussianBlur(mofangimg, (15, 15), 0)    #图像高斯模糊


hsvgaussmofang = cv2.cvtColor(gaussmofangimg, cv2.COLOR_BGR2GRAY)  #图像颜色格式从RGB转为HSV
ret, finalmofangimg = cv2.threshold(hsvgaussmofang, 100, 255, cv2.THRESH_BINARY)  #图像二值化 灰度高于100的像素点全部设置为255

# 自适应二值化
adatpthresh2 = cv2.adaptiveThreshold(hsvgaussmofang, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
#中值滤波去除椒盐杂质
adatpthresh2 = cv2.medianBlur(adatpthresh2, 3)


#二值化图像Canny算法边沿检测
edge = cv2.Canny(finalmofangimg, 100, 255)
adatpedge = cv2.Canny(adatpthresh2, 100, 255)

while True:

    #全局变量定义
    text = 'Empty'
    radius = None
    cntss = None
    ballradiusmin = 30
    ballradiusmax = 70
    ballnumber = 0
    mofangnumber = 0

    im, cnts, hierarchy = cv2.findContours(adatpedge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    centerlist = []
    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue

        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        centerlist.append(center)
        radius = int(radius)

        if radius >= ballradiusmin and radius <= ballradiusmax:

            if radius < 70:

                cv2.circle(mofangimg, center, radius, (0, 255, 0), 2)
                cv2.putText(mofangimg, "center:{}".format(center), (int(x - 2 * radius), int(y - radius)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0),1)
                cv2.putText(mofangimg, "ball's radius :{}".format(radius), (int(x - 2 * radius) , int(y - 1.3 * radius)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0), 1)


            if radius >= 70:
                cv2.circle(mofangimg, center, radius, (0, 255 ,0), 2)
                cv2.putText(mofangimg, "center:{}".format(center), (int(x - 1.5 *radius), int(y - radius)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
                cv2.putText(mofangimg, "MoFang's radius :{}".format(radius), (int(x - 1.5 * radius), int(y - 1.2 * radius)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)


    # 显示图像

    cv2.imshow('mofangimg', mofangimg)
    cv2.imshow('adatpedge', adatpedge)
    cv2.imshow('adaptimg', adatpthresh2)

    '''
    cv2.imshow('guassmofangimg',gaussmofangimg)
    cv2.imshow('edge', edge)
    cv2.imshow('thresh', finalmofangimg)
    cv2.imshow('adatpedge', adatpedge)
    '''
    key = cv2.waitKey(5) &0xFF
    if key == 27:
        break

cv2.destroyAllWindows()