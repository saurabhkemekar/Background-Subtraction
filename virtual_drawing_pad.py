import numpy as np
import cv2


def nothing(x):
     pass


cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
row, col = frame1.shape[:2]
cv2.namedWindow('track_bar')
cv2.createTrackbar('h_low','track_bar',0,180,nothing)
cv2.createTrackbar('h_high','track_bar',0,180,nothing)
black_paper = np.zeros((col, row, 3), np.uint8)
while(cap.isOpened()):

        ret,frame = cap.read()
  
    
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_h = cv2.getTrackbarPos('H_low', 'tracker')
        high_h = cv2.getTrackbarPos('H_high', 'tracker')

        lower_value = np.array([lower_h,50,50])
        high_value = np.array([high_h,255,255])

        mask = cv2.inRange(hsv_frame, lower_value, high_value)

        # removing small noise present in image using morphological tools
        kernel = np.ones((5, 5), np.uint8)
        erode = cv2.erode(mask, kernel, iterations=1)

        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(erode, kernel, iterations=1)

        img = cv2.bitwise_and(frame, frame, mask=mask)
        row, col = mask.shape[:2]
        try:

            M = cv2.moments(dilation)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            black_paper = cv2.circle(
                black_paper, (cx, cy), 4, (255, 255, 255), -1)

        except ZeroDivisionError:
            pass
        cv2.imshow('mask2', black_paper)
        cv2.imshow('mask', frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 1:
            break

cap.release()
cv2.destroyAllWindows()
