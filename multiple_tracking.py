# This code detects different colors and tracks their motion in space.
# You can either load a video or get input from a webcam.


# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
 
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
 

# define the lower and upper boundaries of the colors in the HSV color space
#lower = {'red':(166, 84, 141),'green':(66, 122, 129),'blue':(97, 100, 117),'yellow':(23, 59, 119),'orange':(0, 50, 80),'white':(0, 0, 180)} 
lower = {'red':(0,170,230),'green':(30,85,6),'blue':(97,100,117),'yellow':(23,59,119)}
#upper = {'red':(186,255,255),'green':(86,255,255),'blue':(117,255,255),'yellow':(54,255,255),'orange':(20,255,255),'white':(5, 5, 255)}
upper = {'red':(35,245,245),'green':(165,255,255),'blue':(117,255,255),'yellow':(54,255,255)}

# define standard colors for circle around the object (RGB)
#colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(255,255,0), 'orange':(255,165,0), 'white':(100,100,100)}
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(255,255,0)}
 #pts = deque(maxlen=args["buffer"])
 
# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])


# declare empty dictionary to keep trajectory data (how many colors?)
traj = {}
for key, value in upper.items(): traj[key]=[] 


# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
  
    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # flag to check if both colors appear, to draw connecting line
    connect = True

    # for each color in dictionary, check object in the frame
    for key, value in upper.items():
        # construct a mask for the specified color from dictionary, 
        # then perform a series of dilations and erosions to remove 
        # any small blobs left in the mask
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
               
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        connect = connect and (len(cnts)>0)
        
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use it to compute 
            # the minimum enclosing circle and centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            #print("For %s color: %s" %(key,center))

            # keep the center location of the contours in the dictionary
            traj[key].append(center)

            # only proceed if the radius meets a minimum size. Correct this value for your object's size
            if radius > 0.5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                cv2.putText(frame, key, (int(x-radius),int(y-radius)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)

    # create a line connecting two colors, if only both colors appear
    #if connect :
    #    cv2.line(frame, traj["red"][-1], traj["green"][-1], (255,0,0), 5)    
     
    # show the frame to our screen
    cv2.imshow("Frame", frame)
   
    # display the image and wait for a keypress
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop.
    if key == ord("q"):
        #print(traj)
        break

# after quiting, plot the trajectory only if there is any to plot
# draw each color first before showing them simultaneously
for key, value in traj.items():
    if len(value) > 0:
        plt.scatter(*zip(*traj[key]), s=5, color=key)
        plt.xlim([0, 600])
        plt.ylim([0, 600])
plt.show()
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
