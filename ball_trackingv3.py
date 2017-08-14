# import the necessary packages
#from picamera.array import PiRGBArray
#from picamera import PiCamera
#import time
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# initialize the camera and grab a reference to the raw camera capture
#camera = PiCamera()
#camera.resolution = (640, 480)
#camera.framerate = 32
#rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
#time.sleep(0.1)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=34,
    help="max buffer size")
args = vars(ap.parse_args())

# define range of Cyan color in HSV
lower_cyan = np.array([80, 100, 100])
upper_cyan = np.array([130, 255, 255])

# define range of white color in HSV
lower_white = np.array([0, 0, 230])
upper_white = np.array([180, 30, 255])

# define range of red color in HSV
lower_red1 = np.array([0, 50, 100])
upper_red1 = np.array([5, 255, 255])
lower_red2 = np.array([160, 30, 100])
upper_red2 = np.array([180, 255, 255])

# Set kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

ptsR = deque(maxlen=args["buffer"])
ptsW = deque(maxlen=args["buffer"])

### if a video path was not supplied, grab the reference
### to the webcam
##if not args.get("video", False):
##    camera = cv2.VideoCapture(0)
##
### otherwise, grab a reference to the video file
##else:
##    camera = cv2.VideoCapture(args["video"])
# temp hack
camera = cv2.VideoCapture(args["video"])
a=True
# keep looping
while True:

    # grab the current frame
  (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
  if args.get("video") and not grabbed:
    break
# capture frames from the camera
#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # resize the frame, blur it, and convert it to the HSV
    # color space
 # frame = imutils.resize(frame.array, width=600) # for picamera
  frame = imutils.resize(frame, width=600) # for video

    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only cyan colors
  maskTable = cv2.inRange(hsv, lower_cyan, upper_cyan)
  maskTable = cv2.dilate(maskTable, None, iterations=10)
  maskTable = cv2.erode(maskTable, None, iterations=10)

  # Threshold the HSV image for white color
  maskWhite = cv2.inRange(hsv, lower_white, upper_white)
#  maskWhite = cv2.dilate(maskWhite, None, iterations=2)
#  maskWhite = cv2.erode(maskWhite, None, iterations=2)
  #maskWhite = cv2.morphologyEx(maskWhite, cv2.MORPH_OPEN, kernel)
  maskWhite = maskWhite & maskTable

  # Threshold the HSV image for red color
  maskRed1 = cv2.inRange(hsv, lower_red1, upper_red1)
  maskRed2 = cv2.inRange(hsv, lower_red2, upper_red2)
  maskRed = maskRed1 | maskRed2
  maskRed = cv2.dilate(maskRed, None, iterations=1)
  maskRed = cv2.erode(maskRed, None, iterations=2)
  maskRed = cv2.dilate(maskRed, None, iterations=1)
#maskRed = cv2.morphologyEx(maskRed, cv2.MORPH_CLOSE, kernel) # fill holes
#  maskRed = cv2.morphologyEx(maskRed, cv2.MORPH_OPEN, kernel) # remove noise
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
                           param1=45,param2=4,minRadius=3,maxRadius=7)


  if circlesRed is not None:
      circlesRed= np.uint16(np.around(circlesRed))
      ptsR.appendleft(circlesRed[0,:,0:2])
#  for i in circlesRed[0,:]:
     # draw the outer circle
#    cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),1)
    # draw the center of the circle
#    cv2.circle(frame,(i[0],i[1]),1,(0,0,255),-1)

  # hough White circles
  resW = cv2.cvtColor(resW,cv2.COLOR_BGR2GRAY)
  resW = cv2.medianBlur(resW,5)
  circlesWhite = cv2.HoughCircles(resW,cv2.HOUGH_GRADIENT,0.1,7,
                           param1=50,param2=4,minRadius=3,maxRadius=6)
  if circlesWhite is not None:
      circlesWhite = np.uint16(np.around(circlesWhite))
      ptsW.appendleft(circlesWhite[0,:,0:2])
#  for i in circlesWhite[0,:]:
     # draw the outer circle
#    cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),1)
    # draw the center of the circle
#    cv2.circle(frame,(i[0],i[1]),1,(0,0,255),-1)


    # loop over the set of tracked points
  for i in xrange(1, len(ptsW)):
        # if either of the tracked points are None, ignore
        # them
    if ptsW[i - 1] is None or ptsW[i] is None:
          continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1)
    for pt in ptsW[i]:
      pt=tuple(pt)
      cv2.circle(frame,pt,2,(255,200,0),thickness)
    #ptsW.pop()
#        cv2.line(frame, ptsW[i - 1], ptsW[i], (0, 0, 255), thickness)
    # loop over the set of tracked points
  for i in xrange(1, len(ptsR)):
        # if either of the tracked points are None, ignore
        # them
    if ptsR[i - 1] is None or ptsR[i] is None:
          continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1)
    for pt in ptsR[i]:
      pt=tuple(pt)
      cv2.circle(frame,pt,2,(0,250,20),thickness)
    #ptsR.pop()
#        cv2.line(frame, ptsW[i - 1], ptsW[i], (0, 0, 255), thickness)

    # show the frame to our screen
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
  #rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
  if key == ord("q"):
        break
  a=False

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()


#cv2.imshow('frame',frame)
#cv2.imshow('mask',maskWhite)
#cv2.imshow('res',res)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
