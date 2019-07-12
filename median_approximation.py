import numpy as np
import cv2
import os

def nothing(x):
	pass

cap = cv2.VideoCapture(0)
images = []

#cv2.namedWindow('tracker')
#cv2.createTrackbar('val','tracker',10,255,nothing)
while True:
	
	ret,frame = cap.read()
	
	dim = (700,700)
	
	frame = cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)

	#cv2.imshow('image',frame)

	frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	        
	images.append(frame1)
  # considering the recent 50 images only to estimated background
	if len(images) ==50:
	        images.pop(0)
	

	image = np.array(images)
	
	image = np.median(images,axis=0)

	image = image.astype(np.uint8)
	
	background_index = np.where(frame1 >image)
	image[background_index] = image[background_index] +1
	background_index = np.where(frame1 < image)
	image[background_index] = image[background_index] -1
	cv2.imshow('background',image)
	#print(image.shape)
	forground = cv2.subtract(frame1,image)
	b = np.array([0],np.uint8)
	forground = np.where(forground>40,frame1,b)

	cv2.imshow('black_image',forground)
	
	
	
	
	k = cv2.waitKey(10) & 0xFF

	if k ==25:
		break


cap.release()

cv2.destroyAllWindows()		
