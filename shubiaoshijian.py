import cv2
import serial

ser = serial.Serial('/dev/tty.usbserial',115200)

def draw(event,x,y,flags,param):
    global ix, iy
    if event==cv2.EVENT_LBUTTONDBLCLK:
        ix, iy = int(x),int(y)
        print("X=", x)
        print("Y=", y)
        center = str(ix*1000+iy)
        ser.write(center.encode())
        ser.flushInput()
        ser.flushOutput()
        ser.close()

    else:
        pass


img = cv2.imread("/Users/jonathan/Desktop/ImageForJet/ball6.jpeg")  #加载图片

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)


while(1):

    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()