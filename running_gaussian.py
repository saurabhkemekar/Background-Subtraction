import cv2
import numpy as np


cap = cv2.VideoCapture(0)
ret,mean = cap.read()

# initializing the mean with intial value of first frame pixel value
mean =cv2.cvtColor(mean,cv2.COLOR_BGR2GRAY)
(row,col) = mean.shape[:2]

# define the variance with same random value
var = np.ones((row,col),np.uint8)
background = np.zeros((row,col),np.uint8)
forground = np.zeros((row,col),np.uint8)
var[:row,:col] = 400
m = np.zeros((row,col))
while True:

        ret,frame = cap.read()
        cv2.imshow('frame',frame)              
        # coverting the frame into the grayscale                   
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
        forground = np.zeros((row,col),np.uint8)
        a = np.uint8([255])
        b = np.uint8([0])
        mean = mean.astype(np.float64)
        # deciding the alpha value 
        alpha =0.001
        frame_gray = frame_gray.astype(np.float64)
        value  = cv2.absdiff(frame_gray,mean)
        value = value /np.sqrt(var)
        
        # updagting the value of mean and variance which change if particular pixel  is in  range of gaussian distribution
        back =np.where(value <= 2.5)
        fore = np.where(value>2.5)
        m[back] = m[back] -1
        m[fore] = m[fore] +1
        
        new_mean = (1-alpha)*mean + alpha*frame_gray       
        
        new_var = (alpha)*((frame_gray-mean)**2) + (1-alpha)*(var)
        # if object in video steady for more than 120 frame then it consider as background
        index = np.where(m>120)
        mean[index] = frame_gray[index]
        var[index] = 500
        m[index] =0
        
        
        mean = np.where(value < 2.5,new_mean,mean)
        var = np.where(value < 2.5,new_var,var)
        forground[fore] = frame_gray[fore]
        background[back] = frame_gray[back] 
        cv2.imshow('background',background)       
        kernel = np.ones((3,3),np.uint8)
        
        erode = cv2.erode(forground,kernel,iterations =1)
       # erode = cv2.absdiff(forground,background)
        dilate = cv2.dilate(erode,kernel,iterations = 1)     
        cv2.imshow('forground',dilate)
        k = cv2.waitKey(5) & 0xFF
        
        if k ==10:
                break
                
cap.release()                
cv2.destroyAllWindows()                
        
       
