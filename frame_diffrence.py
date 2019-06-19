import numpy as np
import cv2
import os

def nothing(x):
	pass

def get_frame_diff(pre_frame,curr_frame):

	diff1 = cv2.absdiff(curr_frame,pre_frame)
	

	return diff1


cap = cv2.VideoCapture(0)
cv2.namedWindow('trackbar')
cv2.createTrackbar('min_threshold','trackbar',0,255,nothing)
cv2.createTrackbar('max_threshold','trackbar',0,255,nothing)

while True:

	ret,pre_frame  = cap.read()
	ret,curr_frame = cap.read()
	
    # converting the bgr image to gray scale image
	pre_frame = cv2.cvtColor(pre_frame,cv2.COLOR_BGR2GRAY)
	curr_frame = cv2.cvtColor(curr_frame,cv2.COLOR_BGR2GRAY)
	
	background_image = get_frame_diff(pre_frame,curr_frame)
	min = cv2.getTrackbarPos('min_threshold','trackbar')
	max = cv2.getTrackbarPos('max_threshold','trackbar')

	a = np.array([0],np.uint8)
	b = np.array([255],np.uint8)

	image = np.where((background_image>min) &(background_image<max)  , b ,a)

	kernel = np.array((3,3),np.uint8)

	image = cv2.erode(image,kernel,iterations= 1)

	image = cv2.dilate(image,kernel,iterations =1)

	image = cv2.bitwise_and(curr_frame,curr_frame,mask= image)
	
	cv2.imshow('numpy_subtract',image)
	
	cv2.imshow('background_image',background_image)

	k = cv2.waitKey(5) & 0xFF

	if k==1:
		break

cap.release()
cv2.destroyAllWindows()
