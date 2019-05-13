# This code detects lines at the towel's edges and assess the quality of 
# folding based on line cues. 

import numpy as np
import argparse
import imutils
import cv2
#import matplotlib.pyplot as plt
 
font = cv2.FONT_HERSHEY_COMPLEX

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
 

# define the lower and upper boundaries of the colors in the HSV color space
lower = {'towel':(20,90,100)}
upper = {'towel':(60,160,160)}

# define standard colors around the object (RGB)
colors = {'towel':(0,255,0)}
 
# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

# declare empty dictionary to keep trajectory data (how many colors?)
traj = {}
for key, value in upper.items(): traj[key]=[] 

def nothing(x):
    # if nothing happens then pass
    pass

# create a GUI window containing sliders for edge detection and hough algo
#cv2.namedWindow("Trackbars")
#cv2.createTrackbar("MinLength", "Trackbars", 1, 300, nothing)
#cv2.createTrackbar("MaxGap", "Trackbars", 1, 200, nothing)
#cv2.createTrackbar('thrs1', 'Trackbars', 50, 500, nothing)
#cv2.createTrackbar('thrs2', 'Trackbars', 150, 500, nothing)

lines = []

# keep looping
t = float(0)
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
  
    #minLineLenth = cv2.getTrackbarPos("MinLength", "Trackbars")
    #maxLineGap   = cv2.getTrackbarPos("MaxGap", "Trackbars")
    #thrs1 = cv2.getTrackbarPos("thrs1", "Trackbars")
    #thrs2 = cv2.getTrackbarPos("thrs2", "Trackbars")
  
    # resize the frame, convert it to the HSV color space, THEN blur it!
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    edges = cv2.Canny(frame,50,150,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,0.5,np.pi/180,100)#,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

   # show the frame to our screen
    cv2.imshow("Frame", frame)
   
    # display the image and wait for a keypress
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop.
    if key == ord("q"):
        #print(traj)
        break
    
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
