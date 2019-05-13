# I often have difficulty in getting the color of an object, which changes
# based on the illumination. Here, you obtain the color of a pixel based on
# the location of a mouse cursor.
 
# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
 
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
 

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

    
# function to handle mouse click, containing 5 arguments
def myPrint(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        r,g,b = frame[y,x]
        hsv = cv2.cvtColor(np.uint8([[[r,g,b]]]), cv2.COLOR_BGR2HSV)
        print("Color (HSV space): ", hsv)
    
# setup a mouse click event on "Frame"
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", myPrint )

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
  
    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=400)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # show the frame to our screen, display it in HSV space
    cv2.imshow("Frame", hsv)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop.
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
