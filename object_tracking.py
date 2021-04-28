import cv2 as cv
import numpy as np 
camera  = cv.VideoCapture(0,cv.CAP_DSHOW)

ret, frame = camera.read()
print(type(frame))

r, h, c,w = 240, 100, 400,160
track_windows = (c,r,w,h)

roi = frame[r:r+h,c:c+w]

hsv_roi = cv.cvtColor(roi,cv.COLOR_BGR2HSV)

lower_purple = np.array([123,0,0])
upper_purple = np.array([175,255,255])
mask = cv.inRange(hsv_roi, lower_purple, upper_purple)
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist, roi_hist,0,255,cv.NORM_MINMAX)
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10,1)
while True:
	ret,frame = camera.read()
	if ret==True:
		hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
		dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
		ret, track_windows = cv.meanShift(dst,track_windows,term_crit)
		x,y,w,h = track_windows
		img2 = cv.rectangle(frame,(x,y),(x+w,y+h),255,2)
		cv.imshow("Test Object Tracking",img2)
		if cv.waitKey(1) == 13:
			break
	else:
		break
camera.relase()
cv.destroyAllWindows()
