#######################################
### Sample code to track green ball ###
#######################################
# https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
# https://github.com/jrosebr1/imutils

import cv2
import numpy as np
from collections import deque

#######################################
### Capture color image from camera ###
#######################################
cap = cv2.VideoCapture(0)

##################################
### OpenCV GUI color threshold ###
##################################
def nothing(x): pass # For cv2.createTrackbar
cv2.namedWindow('gui', cv2.WINDOW_NORMAL)
cv2.resizeWindow('gui', 640, 480)
for i in range(6): cv2.createTrackbar(str(i),'gui',0,255,nothing)
lower_green = np.array([20, 125, 25])
upper_green = np.array([99, 255, 255])
thres_low   = lower_green.copy()
thres_high  = upper_green.copy()

def reset_trackbars():
	thres_low   = lower_green
	thres_high  = upper_green
	for i in range(3): cv2.setTrackbarPos(str(i),'gui',thres_low[i])
	for i in range(3): cv2.setTrackbarPos(str(i+3),'gui',thres_high[i])

reset_trackbars()

# Length of trailing tail
tail = deque(maxlen=32)
tail2= deque(maxlen=32)

while True:
	# Capture new image
	ret, img = cap.read()

	# Convert to hsv for extracting green mask and contour
	hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	for i in range(3): thres_low[i]  = cv2.getTrackbarPos(str(i), 'gui')
	for i in range(3): thres_high[i] = cv2.getTrackbarPos(str(i+3), 'gui')
	mask = cv2.inRange(hsv, thres_low, thres_high)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[1] # Note: OpenCV3 length of contours tuple is 3 but CV2 length is 2
	# contours = contours[0] # Note: OpenCV4 use this

	center = None

	# Only proceed if at least one contour
	if len(contours) > 0:
		contours = sorted(contours, key=cv2.contourArea)
		# Find largest contour in the mask
		# c = max(contours, key=cv2.contourArea)
		c = contours[-1]
		# Compute the minimum enclosing circle
		((x, y), radius) = cv2.minEnclosingCircle(c)
		# Find center of circle more refined as compared to (x, y)
		M = cv2.moments(c)
		if M['m00'] > 0:
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			center = (cx, cy)
 
		# Only draw large enough circle
		if radius > 20:
			cv2.circle(img, (int(x), int(y)), int(radius), (0,255,255), 2)
			cv2.circle(img, center, 5, (0,0,255), -1)
			# Update the tail queue
			tail.appendleft(center)	

		# if len(contours) > 1:		
		# 	# Find second largest contour in the mask
		# 	c2 = contours[-2]
		# 	# Compute the minimum enclosing circle
		# 	((x, y), radius) = cv2.minEnclosingCircle(c2)
		# 	# Find center of circle more refined as compared to (x, y)
		# 	M = cv2.moments(c2)
		# 	if M['m00'] > 0:
		# 		cx = int(M['m10']/M['m00'])
		# 		cy = int(M['m01']/M['m00'])
		# 		center2 = (cx, cy)			

		# 	# Only draw large enough circle
		# 	if radius > 20:
		# 		cv2.circle(img, (int(x), int(y)), int(radius), (255,0,255), 2)
		# 		cv2.circle(img, center2, 5, (255,0,0), -1)
		# 		# Update the tail queue
		# 		tail2.appendleft(center2)				

	##############################
	### Draw the trailing tail ###
	##############################
	for i in range(1, len(tail)):
		# If either of the tracked points are None, ignore them
		if tail[i - 1] is None or tail[i] is None: 
			continue
		# Otherwise, compute the thickness of the line and draw the connecting lines
		thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
		cv2.line(img, tail[i - 1], tail[i], (0,0,255), thickness)	

	# for i in range(1, len(tail2)):
	# 	# If either of the tracked points are None, ignore them
	# 	if tail2[i - 1] is None or tail2[i] is None:
	# 		continue
	# 	# Otherwise, compute the thickness of the line and draw the connecting lines
	# 	thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
	# 	cv2.line(img, tail2[i - 1], tail2[i], (255,0,0), thickness)	

	cv2.imshow("img", img)
	cv2.imshow("mask", mask)

	key = cv2.waitKey(30)
	if key==27: 
		break
	if key==ord('r'):
		reset_trackbars()

# Cleanup
cap.release()
cv2.destroyAllWindows()


# Some reference color threshold for camera
# lower_red_low  = np.array([0, 170, 35]) # Note: S>170 so as not to overlap with skin
# upper_red_low  = np.array([9, 255, 255])
# lower_yellow   = np.array([17, 80, 100]) # Note: S>140 so as not to overlap with skin
# upper_yellow   = np.array([40, 255, 255])
# lower_blue     = np.array([100, 170, 15]) # Note: S>200 as blue starts later
# lower_green    = np.array([20, 125, 25])
# upper_green    = np.array([99, 255, 255])
# upper_blue     = np.array([119, 255, 255])
# lower_violet   = np.array([120, 50, 25])
# upper_violet   = np.array([150, 200, 255]) # Note: S<200 as violet did not go beyond
# lower_red_high = np.array([160, 170, 50]) # Note: S>170 so as not to overlap with skin
# upper_red_high = np.array([179, 255, 255])
