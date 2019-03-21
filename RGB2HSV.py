import cv2
import numpy as np
print('-'*50)
print('                    RGB - HSV')
print('-'*50)
R = input('R:')
G = input('G:')
B = input('B:')

color = np.uint8([[[R, G, B]]])
HSV = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
H = HSV[0][0][0]
S = HSV[0][0][1]
V = HSV[0][0][2]

print('H:%d' %H, end=' ')
print('S:%d' %S, end=' ')
print('V:%d' %V)

