
# This code detects different colors and tracks their motion in space.
# You can either load a video or get input from a webcam.

# Revision: I'm adding a graph that can track and plot objects moving in real time.
# I added calibration component to correct for camera tilt w.r.t to the floor.
# Object coordinate will be displayed in real time on the plot.


# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import yaml
 


# Define the calibration file (YAML format)
mypath = 'C:/Users/ananda.sidarta/Documents/myPython/'
myfile = 'myRealSense.yaml'

# Load calibration file and convert calibration params as arrays.
with open(mypath+myfile) as f:
    data = yaml.load(f)
    print("Loading the calibration file....")
    matrix  = np.array( data['homo_matrix'] )    # Homogenous matrix
    cam_mat = np.array( data['cam_matrix'] )  # Intrinsic matrix


# This function performs back-projection to the world coordinate system through analytical solutions.
# The purpose is to be able to know (x,y) is the real world. Note that the world position is w.r.t the 
# origin identified during the calibration with a chessboard.

def backProject(x, y):  # The function, I don't return any value?!
    ## Stage 0: Prepare the necessary variables
    t = matrix[0:3,3]
    Rmat = matrix[0:3,0:3]
    Rzx, Rzy, Rzz = Rmat[0:3,2]
    cx, cy = cam_mat[0:2,2]
    fx, fy = cam_mat[0,0], cam_mat[1,1]
    ## Stage 1: Using planar equation to solve Zc, we compute RHS first
    RHS = np.dot(np.array([Rzx,Rzy,Rzz]), t)
    LHS = Rzx*(x-cx)/fx + Rzy*(y-cy)/fy + Rzz
    Zc = RHS/LHS
    ## Stage 2: Compute Xc, Yc in terms of Zc
    Xc = Zc*(x - cx)/fx
    Yc = Zc*(y - cy)/fy
    worldcam = np.array([Xc,Yc,Zc])  # World coordinate in camera system
    #print("World position ", worldcam)
    ## Stage 3: Calculate position in world reference
    Rmat_inv = np.linalg.inv( Rmat )     # Inverse of Rmat
    worldpos = np.matmul(Rmat_inv,(worldcam-t))
    #print("World position ", worldpos)
    worldposition.append(worldpos[0:2])


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
 

# define the lower and upper boundaries of the colors in the HSV color space
#lower = {'red':(150,140,220) }
#upper = {'red':(190,190,245) }
lower = {'red':(0,140,120) }
upper = {'red':(20,200,150) }

# define standard colors for circle around the object (RGB)
colors = {'red':(0,0,255)}
 #pts = deque(maxlen=args["buffer"])
 
# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(1)
# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])


# declare empty dictionary to keep trajectory data (how many colors?)
traj = {}
for i in range(1,5): traj[i]=[] 

t = float(0)



# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
  
    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=1100)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # flag to check if both colors appear, to draw connecting line
    connect = True

    # for each color in dictionary, check object in the frame
    for color, value in upper.items():
        # construct a mask for the specified color from dictionary, 
        # then perform a series of dilations and erosions to remove 
        # any small blobs left in the mask
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv, lower[color], upper[color])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
               
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        connect = connect and (len(cnts)>0)
        #print ('I see', len(cnts), 'circles')
        
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            circle_id = 1
            # cycle through the contours that have been identified
            for c in cnts:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                #print("For %s color: %s" %(color,center))
                
                # keep the center location of the contours in the dictionary
                traj[circle_id].append(center)
                circle_id += 1
                
                # only proceed if the radius meets a minimum size. 
                if radius > 0.5:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius), colors[color], 2)
                    cv2.putText(frame, color, (int(x-radius),int(y-radius)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[color],2)

                # create a plot showing the movement of the colored objects...
                # this should be updated in real-time.
                plt.scatter(*center, s=20, color=color)
                plt.pause(0.01)
    
    # redraw then clear the current entire figure with the object;
    # have to do this to create updating effect!
    plt.draw()
    plt.clf();
    plt.axis([0, 600, 0, 600])
    
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



# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
