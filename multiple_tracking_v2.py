
#------------------------------------------------------------------------------------------------
# In this code, I use Intel RealSense library, 'pyrealsense', to control the camera specifically. 
# So, once you are able to grab the image using this library, the rest of the tracking algorithm 
# remains the same in OpenCV. We no longer use openCV to open/stop the camera streamong.
#------------------------------------------------------------------------------------------------


# import the necessary packages
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import yaml

# Specific for Intel RealSense camera
from utils_camera_rs import *
import pyrealsense2 as rs

camera_config['enable_color'] 		= True	# Calibrate using color
camera_config['enable_infra'] 		= False	# Calibrate using the left IR image that is perfectly aligned to depth
camera_config['enable_depth'] 		= False # Disable depth as default is True
camera_config['enable_ir_emitter'] 	= False # Disable to IR emitter for calibration only
camera_config['preset'] 			= 2 	# D415 1:Default 2:Hand 3:HighAccuracy 4:HighDensity 5:MediumDensity 6:RemoveIRPattern
camera_config['width'] 				= 1280	# Set to max resolution
camera_config['height'] 			= 720	# Set to max resolution
camera_config['fps'] 				= 6		# Lower the fps so that 1280 by 720 can still work on USB2.1 port

# start detecting any cameras connected...
devices = rs.context().query_devices(); 
print('Number of devices detected', len(devices))
# cycle through devices. Instantiate the CameraRS class for each device
cameras = [ CameraRS(devices[i], camera_config) for i in range(len(devices)) ]


# Define the calibration file (YAML format)
mypath = 'C:/Users/ananda.sidarta/Documents/myPython/'
myfile = 'myRealSense4Dec.yaml'

# Load calibration file and convert calibration params as arrays.
with open(mypath+myfile) as f:
    data = yaml.load(f)
    print("Loading the calibration file....")
    matrix  = np.array( data['homo_matrix'] )    # Homogenous matrix
    cam_mat = np.array( data['cam_matrix'] )  # Intrinsic matrix


# define the lower and upper boundaries of the colors in the HSV color space
#lower = {'red':(163, 183, 186), 'green':(58,96,158) }
#upper = {'red':(193, 213, 216), 'green':(88,126,188) }
lower = {'red':( 0,165,189), 'green':(40,0,240),  'blue':(70,230,140) }
upper = {'red':(39,195,209), 'green':(90,30,255), 'blue':(119,255,180) }

# define standard colors for circle around the object (BGR)
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0)}



# This function performs back-projection to the world coordinate system through analytical solutions.
# The purpose is to be able to know (x,y) is the real world. Note that the world position is w.r.t the 
# origin identified during the calibration with a chessboard.

def backProject(x, y):  # Return two numbers!
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
    ## Stage 2: Compute Xc, Yc in terms of Zcq
    Xc = Zc*(x - cx)/fx
    Yc = Zc*(y - cy)/fy
    worldcam = np.array([Xc,Yc,Zc])  # World coordinate in camera system
    #print("World position ", worldcam)
    ## Stage 3: Calculate position in world reference
    Rmat_inv = np.linalg.inv( Rmat )     # Inverse of Rmat
    worldpos = np.matmul(Rmat_inv,(worldcam-t))
    #print("World position ", worldpos)
    # NOTE: The world position should be converted to cm unit!
    return worldpos[0]*100,worldpos[1]*100


# function to handle mouse click anywhere in the frame, containing 5 arguments
def myClick(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        #print("Clicking %.2f, %.2f" %(x,y))
        x1,y1 = backProject(x,y)
        print("World pos %.2f, %.2f" %(x1,y1))

# setup a mouse click event on "Frame" on the image window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", myClick )

# declare empty dictionary to keep trajectory data (how many colors?)
traj = {}
for color, value in upper.items(): traj[color]=[] 

t = float(0)


# keep looping
while True:
    # grab the current camera frame
    for c in cameras:
        if c.cc['enable_color']:
            frame = c.get_color_img() # Get infra image from camera
            
        # if we are viewing a video and there is no image grabbed, bail out the loop
        if frame.size < 1: break
  
        # resize the frame, blur it, and convert it to the HSV color space
        frame = imutils.resize(frame, width=1300)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

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
            #print ('I see', len(cnts), 'circles')
            
            if len(cnts) > 0:
                # find the largest contour in the mask, then use it to compute 
                # the minimum enclosing circle and centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                #print("For %s color: %s" %(color,center))
                
                # Perform back-projection from the camera to world coordinates
                center1 = backProject(center[0], center[1])
                # keep the center location of the contours in the dictionary
                traj[color].append(center1)
                
                # only proceed if the radius meets a minimum size. Correct this value for your object's size
                if radius > 0.5:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius), colors[color], 2)
                    cv2.putText(frame, color, (int(x-radius),int(y-radius)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[color],2)
                    
            if len(traj[color]) > 0: #and len(traj['green']) > 0:
                # create a plot showing the movement of the colored objects, updated in real time
                # note I always take the last item of the list!
                plt.scatter(*traj[color][-1], s=20, color=color)

        # redraw then clear the current entire figure with the object;
        # important to place here to create updating effect!
        plt.axis([-100, 100, -80, 120])
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

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
for color, value in traj.items():
    if len(value) > 0:
        plt.scatter(*zip(*traj[color]), s=20, color=color)  # print coordinate in cm unit
        plt.axis([-100, 100, -80, 120])
        plt.axis('equal')   # axis square!
        
        

# cleanup the camera and close any open windows
cv2.destroyAllWindows()
print ("\nProgram quitting......")
for c in cameras:   
	c.pipeline.stop()   # Stop streaming from RealSense

