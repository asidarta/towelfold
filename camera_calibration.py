
##############################################################################
# Note: In practice you can either load a static image containing chessboard,
# or use real-time video capture using OpenCV functions. Actually if you use
# Intel RealSense camera, there is a pyRealsense package for python.
##############################################################################


import numpy as np
import cv2
import imutils
import glob


# We load an image used during calibration. Define image path.
#mypath  = 'C:/Users/ananda.sidarta/Documents/GitHub/towelfold/'
mypath  = 'D:/'
myfile  = 'calib_chessboard.PNG'
myimage = mypath + myfile
pattern = (10,7)   # Number of squares in the chessboard pattern


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((pattern[0]*pattern[1],3), np.float32)
objp[:,:2] = np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(myimage)

if False:
  for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners, must define correct #squares 9x6. It will return True if found!
    ret, corners = cv2.findChessboardCorners(gray, pattern, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, pattern, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
         
        # Then get necessary camera params from the calibration image
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        homo_matrix = np.hstack((cv2.Rodrigues(rvecs[0])[0], tvecs[0]))   # 3 by 4 R|t matrix

        # undistort the image
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]




# Assume don't load any image from the calibration. We directly capture from the camera itself.
        
camera  = cv2.VideoCapture(1)  # Change for webcam/USB camera!!!
camera.set(3,1280)   # Set frame height
camera.set(4,1080)   # Set frame width


# setup a mouse click event on "Frame"
cv2.namedWindow("Frame")
  
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
  
    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=1200)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners, must define correct #squares 9x6. It will return True if found!
    ret, corners = cv2.findChessboardCorners(gray, pattern, None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        frame = cv2.drawChessboardCorners(frame, pattern, corners2, ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        h,  w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        homo_matrix = np.hstack((cv2.Rodrigues(rvecs[0])[0], tvecs[0]))   # 3 by 4 R|t matrix

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop.
    if key == ord("q"):
        #cv2.imwrite('calib_result.png', dst)
        break



# Quit and finished! 
cv2.destroyAllWindows()
print ("\nProgram quitting......")
print ("Camera intrinsic: \n", mtx)
print ("Camera extrinsic: \n", homo_matrix)
