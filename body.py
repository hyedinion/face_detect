import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("./test2.jpg")
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'./haarcascade_frontalface_default.xml')
body = body_cascade.detectMultiScale(grayImage, 1.01, 10)

for (x,y,w,h) in body:
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)

plt.figure(figsize=(12,12))
plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

cv2.destroyAllWindows()
