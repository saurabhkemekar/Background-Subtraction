import numpy as np
import cv2


def nothing(x):
     pass


cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
row, col = frame1.shape[:2]

# defining the trackbar 
cv2.namedWindow('track_bar')
cv2.createTrackbar('R','track_bar',0,255,nothing)
cv2.createTrackbar('G','track_bar',0,255,nothing)
cv2.createTrackbar('B','track_bar',0,255,nothing)
cv2.createTrackbar('ERASE','track_bar',0,1,nothing)
black_paper = np.ones((row, col, 3), np.uint8)

while(cap.isOpened()):

        ret,frame = cap.read()
        row, col = frame.shape[:2]
        
        frame = cv2.GaussianBlur(frame,(5,5,),1)
        
            
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        r = cv2.getTrackbarPos('R', 'track_bar')
        b = cv2.getTrackbarPos('B', 'track_bar')
        g = cv2.getTrackbarPos('G', 'track_bar')
        black_paper =cv2.rectangle(black_paper,(0,0),(col,50),(255,255,255),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(black_paper,'ERASE DRAWING',(80,40),font,1,(255,255,255),2,cv2.LINE_AA)


                
        lower_value = np.array([40,50,50])
        high_value = np.array([80,255,255])

        mask = cv2.inRange(hsv_frame, lower_value, high_value)

        # removing small noise present in image using morphological tools
        kernel = np.ones((5, 5), np.uint8)
        erode = cv2.erode(mask, kernel, iterations=1)

        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(erode, kernel, iterations=1)

        img = cv2.bitwise_and(frame, frame, mask=mask)
        try:

            M = cv2.moments(dilation)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            print(cx,cy)

            if  cy<50:
                 black_paper = np.ones((row,col,3),np.uint8)
                 black_paper =cv2.rectangle(black_paper,(0,0),(col,50),(255,255,255),3)
                 font = cv2.FONT_HERSHEY_SIMPLEX
                 cv2.putText(black_paper,'ERASE DRAWING',(80,40),font,1,(255,255,255),2,cv2.LINE_AA)
                 print('saurabh')

                
            else:    
                black_paper = cv2.circle(
                        black_paper, (cx, cy), 4, (b, g, r), -1)

        except ZeroDivisionError:
            pass
        cv2.imshow('mask2', black_paper)
        cv2.imshow('mask', frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 1:
            break

cap.release()
cv2.destroyAllWindows()
