import cv2
import numpy as np
from matplotlib import pyplot as plt

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)
	
	image = cv2.imread(frame)
	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'./haarcascade_frontalface_default.xml')
	body = body_cascade.detectMultiScale(grayImage, 1.01, 10)

	for (x,y,w,h) in body:
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)

	plt.figure(figsize=(12,12))
	plt.imshow(image, cmap='gray')
	plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	plt.show()
	
    if cv2.waitKey(1) > 0: break

capture.release()
cv2.destroyAllWindows()
