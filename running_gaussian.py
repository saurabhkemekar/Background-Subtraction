import cv2
import numpy as np
from scipy.stats import norm
import math


cap = cv2.VideoCapture(0   )
ret,mean = cap.read()
mean =cv2.cvtColor(mean,cv2.COLOR_BGR2GRAY)
(col,row) = mean.shape[:2]

var = np.ones((col,row),np.uint8)
var[:row,:col] = 10

while True:

        ret,frame = cap.read()                                
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
        
        # alpha =0.00001  
        print(var[10,10])                
        new_mean = 0.12*mean + 0.88*frame_gray       
        new_mean = new_mean.astype(np.uint8)
        new_var = 0.88*(cv2.subtract(frame_gray,mean)**2) + 0.12*(var)
        new_var = new_var.astype(np.uint8)
        
        std = np.sqrt(new_var)*3
        std = std.astype(np.uint8
        
        mean = np.where((frame_gray>cv2.subtract(mean,std)) & (frame_gray<cv2.add(mean,std)),new_mean,mean)
        var = np.where((frame_gray>cv2.subtract(mean,std)) & (frame_gray<cv2.add(mean,std)),new_var,var)
        
        
        #value  = cv2.subtract(frame_gray,mean)
        
        #value = value.astype(np.uint8)
        
        #me an = np.where(value < 2.5,new_mean,mean)
        #var = np.where(value < 2.5,new_var,var)
        
        cv2.imshow('background',mean)
        a = np.uint8([0])
        forground =np.where((frame_gray>cv2.subtract(mean,std)) & (frame_gray<cv2.add(mean,std)),a,frame)
        #forground = cv2.GaussianBlur(forground,(5,5),1)
      
        black =np.array([0],np.uint8)
        white =np.array([255],np.uint8) 
        forground = np.where(forground>80,white,black)
       
        kernel = np.zeros((5,5),np.uint8)
        erode = cv2.erode(forground,kernel,iterations =2)
        #dilate = cv2.dilate(erode,kernel,iterations =1)
      
        forground =cv2.bitwise_and(frame,frame,mask =erode)

       
        cv2.imshow('forground',forground)            
        k = cv2.waitKey(5) & 0xFF
        
        if k ==10:
                break
                
cap.release()                
cv2.destroyAllWindows()                
        
       
