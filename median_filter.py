import numpy as np
import cv2


def nothing(x):
	pass


cap = cv2.VideoCapture('car.avi')
images = []

cv2.namedWindow('tracker')
cv2.createTrackbar('val','tracker',10,255,nothing)
while True:
	
	ret,frame = cap.read()

	cv2.imshow('image',frame)

	frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	        
	images.append(frame1)
	if len(images) ==10:
	        images.pop(0)
	

	image = np.array(images)
	
	image = np.median(images,axis=0)

	image = image.astype(np.uint8)
	
	cv2.imshow('BACKGROUND_IMAGE',image)
	#print(image.shape)
	background_image = cv2.absdiff(frame1,image)

	val =cv2.getTrackbarPos('val','tracker')

	a = np.array([255],np.uint8)
	b = np.array([0],np.uint8)

	forground = np.where(background_image>val,a,b)
	
	kernel = np.ones((3,3),np.uint8)
	
	forground_erode = cv2.erode(forground,kernel,iterations =1)
	
	forground_dilate = cv2.dilate(forground_erode,kernel,iterations =1)
	
	cv2.imshow('image',forground_dilate)

	forground_image =  cv2.bitwise_and(frame1,frame1,mask = forground_dilate)
	
	cv2.imshow('FOREGROUND_IMAGE',forground_image)

	k = cv2.waitKey(5) & 0xFF

	if k ==25:
		break


cap.release()

cv2.destroyAllWindows()		
