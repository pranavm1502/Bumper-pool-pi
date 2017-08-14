import numpy as np
import cv2

# HSV values
cyanTable = (90,249,218)
whiteBall = (0,0,255)
redBall = (1,225,254)

#open image
frame = cv2.imread('foo4.jpg')

# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of Cyan color in HSV
lower_cyan = np.array([80,100,100])
upper_cyan = np.array([100,255,255])

# define range of white color in HSV
lower_white = np.array([10,0,220])
upper_white = np.array([170,30,255])

# define range of red color in HSV
lower_red1 = np.array([0,100,100])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([160,100,100])
upper_red2 = np.array([180,255,255])

# Set kernel for morphological operations
kernel = np.ones((5,5),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Threshold the HSV image to get only cyan colors
maskTable = cv2.inRange(hsv, lower_cyan, upper_cyan)
maskTable = cv2.dilate(maskTable, None, iterations=10)
maskTable = cv2.erode(maskTable, None, iterations=7)

# Threshold the HSV image for white color
maskWhite = cv2.inRange(hsv, lower_white, upper_white)
#maskWhite = cv2.erode(maskWhite, None, iterations=2)
#maskWhite = cv2.dilate(maskWhite, None, iterations=2)
#maskWhite = cv2.morphologyEx(maskWhite, cv2.MORPH_OPEN, kernel)
maskWhite = maskWhite & maskTable

# Threshold the HSV image for red color
maskRed1 = cv2.inRange(hsv, lower_red1, upper_red1)
maskRed2 = cv2.inRange(hsv, lower_red2, upper_red2)
maskRed = maskRed1 | maskRed2
#maskRed = cv2.dilate(maskRed, None, iterations=2)
#maskRed = cv2.erode(maskRed, None, iterations=3)
#maskRed = cv2.dilate(maskRed, None, iterations=1)
#maskRed = cv2.morphologyEx(maskRed, cv2.MORPH_CLOSE, kernel) # fill holes
#maskRed = cv2.morphologyEx(maskRed, cv2.MORPH_OPEN, kernel) # remove noise
maskRed = maskRed & maskTable

# Find table boundary (not needed)
#image, contours, hierarchy = cv2.findContours(maskTable.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#img = cv2.drawContours(frame, contours, -1, (0,255,0), 3)


# Bitwise-AND maskRed and original image
resR = cv2.bitwise_and(frame,frame, mask= maskRed)
# Bitwise-AND maskWhite and original image
resW = cv2.bitwise_and(frame,frame, mask= maskWhite)

# hough Red circles
resR = cv2.cvtColor(resR,cv2.COLOR_BGR2GRAY)
resR = cv2.medianBlur(resR,5)
circlesRed = cv2.HoughCircles(resR,cv2.HOUGH_GRADIENT,0.1,7,
                           param1=50,param2=4,minRadius=3,maxRadius=7)

circlesRed = np.uint16(np.around(circlesRed))
for i in circlesRed[0,:]:
     # draw the outer circle
    cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),1)
    # draw the center of the circle
    cv2.circle(frame,(i[0],i[1]),1,(0,0,255),-1)

# hough White circles
resW = cv2.cvtColor(resW,cv2.COLOR_BGR2GRAY)
resW = cv2.medianBlur(resW,5)
circlesWhite = cv2.HoughCircles(resW,cv2.HOUGH_GRADIENT,0.1,7,
                           param1=50,param2=5,minRadius=3,maxRadius=7)

circlesWhite = np.uint16(np.around(circlesWhite))
for i in circlesWhite[0,:]:
     # draw the outer circle
    cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),1)
    # draw the center of the circle
    cv2.circle(frame,(i[0],i[1]),1,(0,0,255),-1)

# find contours in the mask and initialize the current
# (x, y) center of the ball
#cntsRed = cv2.findContours(maskRed.copy(), cv2.RETR_EXTERNAL,
 #                       cv2.CHAIN_APPROX_SIMPLE)[-2]
#center = None
#for c in cntsRed:
#    ((x, y), radius) = cv2.minEnclosingCircle(c)
#    print(radius)
#    M = cv2.moments(c)
#    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#    if radius < 7 and radius > 4:
#        cv2.circle(frame, (int(x), int(y)), int(radius),
#                   (0, 255, 255), 2)
#        cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 0), -1)

#cntsWhite = cv2.findContours(maskWhite.copy(), cv2.RETR_EXTERNAL,
#                        cv2.CHAIN_APPROX_SIMPLE)[-2]
#center = None
#for c in cntsWhite:
#    ((x, y), radius) = cv2.minEnclosingCircle(c)
#    print(radius)
#    M = cv2.moments(c)
#    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#    if radius < 7 and radius > 4:
#        cv2.circle(frame, (int(x), int(y)), int(radius),
#                   (255, 0, 255), 2)
#        cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 0), -1)

cv2.imshow('frame',frame)
#cv2.imshow('mask',maskWhite)
#cv2.imshow('res',res)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
