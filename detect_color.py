# I often have difficulty in getting the color of an object, which changes
# based on the illumination. Here, you obtain the color of a pixel based on
# the location of a mouse cursor.
# Update: Here, we assume you point to homogenous color while clicking
 
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

# If you want to load a still image, comment the camera.read() line in while-loop.
#frame = cv2.imread('C:/Users/ananda.sidarta/Desktop/5.jpg')
#grabbed = True

hsvArray = [] # To contain list of color values in HSV space

# function to handle mouse click anywhere in the frame, containing 5 arguments
def myPrint(event, x, y, flags, params):
    global hsvArray
    xpixels = 15
    ypixels = 7
    if event == cv2.EVENT_LBUTTONDOWN:
        # compute the color values along surrounding x,y pixel coord 
        for i in range(-xpixels,xpixels):
           for j in range(-ypixels,ypixels):
              r,g,b = frame[y+j,x+i]
              hsv = cv2.cvtColor(np.uint8([[[r,g,b]]]), cv2.COLOR_BGR2HSV)
              print("Color (HSV space): ", hsv)
              hsvArray.append(hsv)
        # then compute the mean, lower, and upper bound of the color!
        getRange()
        
    
# setup a mouse click event on "Frame"
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", myPrint )


# function to compute the average and display +/-15 range.
def getRange():
    global hsvArray
    if len(hsvArray):
        hsvMean = np.mean(hsvArray, axis=0)   # careful which axis!
        hsvMean = np.around(hsvMean, decimals=0)  # round the decimals
        print('\nAverage    :  ', hsvMean[0][0])
        print('Lower limit:  ', hsvMean[0][0] - 15)
        print('Upper limit:  ', hsvMean[0][0] + 15)
        print('\n')
        hsvArray = []
        

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
  
    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=900)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # show the frame to our screen, display it in HSV space
    cv2.imshow("Frame", hsv)
    key = cv2.waitKey(1) & 0xFF

    # if the 't' key is pressed, compute the mean, upper and lower boundary
    if key == ord("t"):
        getRange()

    # if the 'q' key is pressed, stop the loop.
    if key == ord("q"):
        break



# once quit the loop, we average the hsv color value and present +/-20 limits.


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
