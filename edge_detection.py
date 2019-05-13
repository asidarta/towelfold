import numpy as np
import cv2
import imutils
#import scipy.ndimage as ndi

img = cv2.imread('C:/Users/ananda.sidarta/Pictures/coba.jpg')
#frame = imutils.resize(img, width=100)
frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#frame = cv2.GaussianBlur(gray, (3, 3), 0)

#smooth = ndi.filters.median_filter(gray, size=2)
#edges = gray#smooth > 180
edges = cv2.Canny(frame,50,100,apertureSize = 3)

lines = cv2.HoughLines(edges.astype(np.uint8), 0.5, np.pi/180, 120)

for rho,theta in lines[0]:
    print(rho, theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),8)

# Show the result
while True:
    cv2.imshow("Line Detection", img)
    # display the image and wait for a keypress
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop.
    if key == ord("q"):
        #print(traj)
        break
