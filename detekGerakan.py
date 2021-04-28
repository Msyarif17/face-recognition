import cv2 as cv
import numpy as np 
camera  = cv.VideoCapture(0,cv.CAP_DSHOW)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
fgbg = cv.createBackgroundSubtractorKNN()

while True:
	ret, f = camera.read()
	if ret == True:
		
	fgmask = fgbg.apply(f)
	fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

	cv.imshow('test',fgmask)

	if cv.waitKey(1) == 13:
		break
camera.relase()
cv.destroyAllWindows()