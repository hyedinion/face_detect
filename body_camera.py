import cv2
import numpy as np
from matplotlib import pyplot as plt

try:
	capture = cv2.VideoCapture(0)
except:
	print('카메라로딩실패')
	
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'./haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'./haarcascade_frontalface_default.xml')

while True:
	ret, frame = capture.read()
	grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	body = body_cascade.detectMultiScale(grayImage, 1.01, 10)
	face = face_cascade.detectMultiScale(grayImage, 1.01, 10)
	for (x,y,w,h) in face:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
		
		
	cv2.imshow('VideoFrame',frame)
	if cv2.waitKey(1) > 0: break
	

capture.release()
cv2.destroyAllWindows()
