## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

########################################################
##      Open CV and RealSense library integration     ##
########################################################

import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import cv2

font = cv2.FONT_HERSHEY_COMPLEX
# define the lower and upper boundaries of the colors in the HSV color space
lower = {'towel':(95, 140, 60)}
upper = {'towel':(110,240,100)}


# Configure depth and color streams
height, width = 1920, 1080
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, height,width, rs.format.z16, 30)
config.enable_stream(rs.stream.color, height,width, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

t=0
area=0

try:  ## some error handling
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # resize the frame, convert it to the HSV color space, THEN blur it!
        frame = color_image
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (11, 11), 0)

        # for each color in dictionary, check object in the frame
        for key, value in upper.items():
            # construct a mask for the specified color from dictionary, 
            # then perform a series of dilations and erosions to remove 
            # any small blobs left in the mask
            kernel = np.ones((9,9),np.uint8)
            mask = cv2.inRange(blurred, lower[key], upper[key])
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

                if area > 400:# and len(approx) == 4:
                    # if exceeds minimum size, draw perimeter according to shape
                    cv2.drawContours(frame, [approx], 0, (255,0,0), 2)
                    cv2.putText(frame, "Detected", (x, y), font, 0.5, (255,0,0))
 
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
        if key == ord("q"): break

finally:

    # Stop streaming
    pipeline.stop()