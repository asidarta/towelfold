##############################################################
### Sample program to stream depth, infra and color frames ###
##############################################################
import cv2
import numpy as np
import pyrealsense2 as rs

########################
### RealSense config ###
########################
# User defined variables
width, height, fps = 640, 480, 30
config = rs.config()
config.enable_stream(rs.stream.infrared, width, height, rs.format.y8, fps) # Configure the pipeline to stream
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps) 
pipeline = rs.pipeline()
pipeline.start(config) # Start streaming

while True:
    ##############################
    ### Get images from camera ###
    ##############################
    frames = pipeline.wait_for_frames() # Wait for coherent frames
    infra_frame = frames.get_infrared_frame()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not infra_frame or not depth_frame or not color_frame:
        continue

    infra_img = np.asanyarray(infra_frame.get_data()) # Convert camera frame to np array image
    depth_img = np.asanyarray(depth_frame.get_data()) # Convert camera frame to np array image
    color_img = np.asanyarray(color_frame.get_data()) # Convert camera frame to np array image
    
    #########################
    ### Display 2D images ###
    #########################
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=255/1000), cv2.COLORMAP_JET)
    cv2.imshow('depth', depth_colormap)
    cv2.imshow('infra', infra_img)    
    cv2.imshow('color', color_img)

    # Get user input
    key = cv2.waitKey(1)
    if key==27: # Press esc to break the loop
        break

pipeline.stop() # Stop streaming
cv2.destroyAllWindows()