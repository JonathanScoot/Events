import argparse
import datetime
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=2, help="minimum area size")
args = vars(ap.parse_args())

imagelist = ['image1','image2','image3']

cv2.namedWindow(imagelist[0])
cv2.namedWindow(imagelist[1])
cv2.namedWindow(imagelist[2])

if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.15)


else:
    camera = cv2.VideoCapture(args["video"])

firstFrame = None

while True:
    grabbed, frame = camera.read()
    text = 'safe'

    if not grabbed:
        break

    # 高斯模糊
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # 初始化
    if firstFrame is None:
        firstFrame = gray
        continue

    # 计算当前帧和第一帧的不同
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 160, 255, cv2.THRESH_BINARY)[1]

    # 扩展阀值图像填充孔洞，找到阀值图像上的轮廓
    thresh = cv2.dilate(thresh, None, iterations=2)
    im, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue

        (x, y), radius = cv2.minEnclosingCircle(c)
        text = (x, y)
        center = (int(x), int(y))
        radius = int(radius)
        img = cv2.circle(frame, center, radius, (0, 255, 0), 2)

    cv2.putText(frame, "ball's coordinate: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("image1", frame)
    cv2.imshow("image2", thresh)
    cv2.imshow("image3", frameDelta)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()

