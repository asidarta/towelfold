
##############################################################################
# Note: In practice you can either load a static image containing chessboard,
# or use real-time video capture using OpenCV functions. Actually if you use
# Intel RealSense camera, there is a pyRealsense package for python.
##############################################################################


import numpy as np
import yaml
import cv2
import imutils
import glob


print("\nThis program does camera calibration and color calibration.") 
print("For the color calibration, click on the homogenous areas....\n") 

# Number of squares in the chessboard pattern and size!
pattern = (10,7)
chessboard_sq_size = 0.035

# prepare coordinates, like (0,0,0), (1,0,0), ..., (0,1,0), ...
objp = np.zeros((pattern[0]*pattern[1],3), np.float32)
objp[:,:2] = np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2)
objp = objp * chessboard_sq_size  # to convert square size to metre

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# termination criteria used for cornerSubPix function below
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)


# For projecting 3D points into 2D image
def project_3Daxis_to_2D(img, camera_matrix, dist_coeffs, rvec, tvec, length):
    # create lines for 3D axes with a length (metre)
    obj_axis_3D = np.float32([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]) * length
    # project the axes into 2D taking camera info
    obj_axis_2D = cv2.projectPoints(obj_axis_3D, rvec, tvec, camera_matrix, dist_coeffs)[0].reshape(-1, 2)
    if obj_axis_2D.shape[0] == 4:
        colours = [(255,0,0),(0,255,0),(0,0,255)]  # R,G,B
        for i in range(1,4):
            (x0, y0), (x1, y1) = obj_axis_2D[0], obj_axis_2D[i]
            # draw the lines, thickness = 3  
            cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), colours[i-1], 3) 
        cv2.putText(img, "Origin", (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,0] ,2)

    #print("Creating axes at origin")
    return img


# important function to calibrate and draw grids and lines
def calib_image_proc(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners, must define correct #squares 9x6. It will return True if found!
    ret, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                                     cv2.CALIB_CB_FAST_CHECK)
       
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        frame = cv2.drawChessboardCorners(frame, pattern, corners2, ret)            

        # Then get necessary camera params from the calibration image
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
  		# solvePnp will return the transformation matrix to transform 3D model coordinate to 2D coordinate
          # Koon used the following function?!?!
        #ret, rvecs, tvecs = cv2.solvePnP(objpoints[0], corners2, mtx, dist)
        h,  w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        homo_matrix = np.hstack((cv2.Rodrigues(rvecs[0])[0], tvecs[0]))   # 3 by 4 R|t matrix
        homo_matrix = np.vstack((homo_matrix, [0,0,0,1]))      # becomes 4 by 4 R|t matrix
        
        # draw the long axis at the origin, on the image 'img'. Note the dim of rvecs and tvecs!!
        frame = project_3Daxis_to_2D(frame, mtx, dist, rvecs[0], tvecs[0], 0.08)
    else:
        mtx, homo_matrix = [], []
    
    # return the frame with grid lines and axes at origin
    return frame, mtx, homo_matrix
      
        
hsvArray = [] # To contain list of color values in HSV space

# function to handle mouse click anywhere in the frame, containing 5 arguments
def myClick(event, x, y, flags, params):
    global hsvArray
    xpixels, ypixels = 2,2#15, 7
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicking %.2f, %.2f" %(x,y))
        # compute the color values along surrounding x,y pixel coord 
        for i in range(-xpixels,xpixels):
           for j in range(-ypixels,ypixels):
              r,g,b = frame[y+j,x+i]
              hsv = cv2.cvtColor(np.uint8([[[r,g,b]]]), cv2.COLOR_BGR2HSV)
              #print("Color (HSV space): ", hsv)
              hsvArray.append(hsv)
        # then compute the mean, lower, and upper bound of the color!
        getRange()
      
        
# function to compute the average and display +/-15 range.
def getRange():
    global hsvArray
    if len(hsvArray):
        hsvMean = np.mean(hsvArray, axis=0)   # careful which axis!
        hsvMean = np.around(hsvMean, decimals=0)  # round the decimals
        #print('\nAverage    :  ', hsvMean[0][0])
        print('Lower limit:  ', hsvMean[0][0] - 15)
        print('Upper limit:  ', hsvMean[0][0] + 15)
        print('\n')
        hsvArray = []        
  


              
##############################################################################
# Main loop: either we load a chessboard image; or stream from the RGB camera
##############################################################################
    
if False:
    # We load an image used during calibration. Define image path.
    mypath  = 'D:/'
    myfile  = 'calib_chessboard.PNG'
    myimage = mypath + myfile
    images = glob.glob(myimage)

    for fname in images:
        img = cv2.imread(fname)
        img, mtx, homo_matrix = calib_image_proc(img) # Run calibration proc
        # undistort the image
        #dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # crop the image
        #x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w]
        cv2.imshow('Frame', img)
    while True:
        # if the 'q' key is pressed, stop the loop.
        key = cv2.waitKey(10) & 0xFF  
        if key == ord("q"):
            #cv2.imwrite('calib_result.png', dst)
            break


else:
    # Specific for Intel RealSense camera
    from utils_camera_rs import *
    import pyrealsense2 as rs

    # Declare a dict listing RealSense camera config
    camera_config['enable_color'] 		= True	# Calibrate using color
    camera_config['enable_infra'] 		= False	# Calibrate using the left IR image that is perfectly aligned to depth
    camera_config['enable_depth'] 		= False # Disable depth as default is True
    camera_config['enable_ir_emitter'] 	= False # Disable to IR emitter for calibration only
    camera_config['preset'] 			= 2 	# D415 1:Default 2:Hand 3:HighAccuracy 4:HighDensity 5:MediumDensity 6:RemoveIRPattern
    camera_config['width'] 				= 1280	# Set to max resolution
    camera_config['height'] 			= 720	# Set to max resolution
    camera_config['fps'] 				= 6		# Lower the fps so that 1280 by 720 can still work on USB2.1 port


    # Start detecting any cameras connected...
    devices = rs.context().query_devices(); 
    print('Number of devices detected', len(devices))
    # Cycle through devices. Instantiate the CameraRS class for each device
    cameras = [ CameraRS(devices[i], camera_config) for i in range(len(devices)) ]

    # setup a mouse click event on "Frame" on the image window
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", myClick )

    # Enter the loop if cameras are connected  
    while len (cameras):

        # Here I state the camera assuming we use RealSense camera, not webcam!
        for c in cameras:
            if c.cc['enable_color']:
                frame = c.get_color_img() # Grab colored image from camera
            
            # Run calibration proc and show the frame after calibration
            frame = imutils.resize(frame, width=1300)
            frame, mtx, homo_matrix = calib_image_proc(frame)
            cv2.imshow('Frame', frame)

        # if the 'q' key is pressed, stop the loop.
        key = cv2.waitKey(1) & 0xFF  
        if key == ord("q"):
            #cv2.imwrite('calib_result.png', dst)
            break

            # Once quit, we see if there is any camera calibration then save the calibration info.
            if len(mtx) > 0:
                # Create a dictionary containing matrices first
                calib_dict = dict( cam_matrix  = mtx.tolist(),
                                  homo_matrix = homo_matrix.tolist() )

                # Then save the dict as YAML structure.
                with open('C:/Users/ananda.sidarta/Documents/myPython/' + 'myRealSense4Dec' + '.yaml', 'w') as outfile:
                    yaml.dump(calib_dict, outfile, default_flow_style=False)
                    print('Saved model2camera_matrix.....!')
                    #print ("Camera intrinsic: \n", mtx)
                    #print ("Camera extrinsic: \n", homo_matrix)
        else:
            print("WARNING! No chessboard found!")


# Quit and finished! Close the streaming.
cv2.destroyAllWindows()
print ("\nProgram quitting......")


