# This code detects the towel and how the shape changes as it is folded.
# You define the one or more colors and the shape (rectangle) to detect. 

import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
 
font = cv2.FONT_HERSHEY_COMPLEX

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
 

# define the lower and upper boundaries of the colors in the HSV color space
#upper = {'towel':(190,80,190)}
#lower = {'towel':(0,40,130)}
upper = {'towel':(115,255,210), 'hand':(110,90,240)}
lower = {'towel':( 95,130,140), 'hand':( 95,10,180)} 

#upper = {'towel':(120,150,150), 'right':(10,255,255), 'left':(90, 90, 110)} 
#lower = {'towel':(90, 90,100), 'right':(0, 170,120), 'left':(40, 30, 50)}

# define standard colors around the object (RGB)
colors = {'towel':(255,0,0), 'hand':(0,0,255), 'left':(0,255,0)}

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

# declare empty dictionary to keep trajectory data (how many colors?)
traj = {}
for item, value in upper.items(): traj[item]=[] 

# prepare the plot properties
area = 0
plt.show() 
        
# If you want to load a still image instead....
#frame = cv2.imread('C:/Users/ananda.sidarta/Desktop/5.jpg')
#grabbed = True

# keep looping
t = float(0)
while True:
    # grab the current frame, comment this for a still image
    (grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
  
    # resize the frame, convert it to the HSV color space, THEN blur it!
    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (11, 11), 0)

    # for each color in dictionary, check object in the frame
    for item, value in upper.items():
        # construct a mask for the specified color from dictionary, 
        # then perform a series of dilations and erosions to remove 
        # any small blobs left in the mask
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(blurred, lower[item], upper[item])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
               
        # find contours in the mask and initialize the current centroid
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #center = None
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # find the LARGEST contour in the mask, then use it to compute 
            # the minimum enclosing circle and centroid
            M = cv2.moments(area)
            #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            # Extract contour feature: how many corners does it have? If you reduce the 
            # 0.01 value then you can extract more (refined) corners.
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True), True)
            #print(len(approx))

            x, y = approx.ravel()[0], approx.ravel()[1]

            if area > 100:# and len(approx) == 4:
            # if exceeds minimum size, draw perimeter according to shape
                cv2.drawContours(frame, [approx], 0, colors[item], 2)
                cv2.putText(frame, item, (x, y-10), font, 0.5, colors[item])
 
        # create a plot showing the area of the towel surface
        #plt.plot(t,area,'bo',markersize=2)
        # display the plot in real-time and adjust the X-axis limit
        #plt.xlim([(t//100-1)*t+t%100, (t//100-1)*t+t%100+100])
        #plt.ylim([9000,60000])
        #plt.pause(0.01)
        #plt.draw()  # redraw the figure
        t+=1
             
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
