import numpy as np
import cv2
from scipy.stats import norm
# here omega indicate the weigth of particular gaussian
# 2 is most probable and 0 is least probable

cap = cv2.VideoCapture(0)
_,frame = cap.read()
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#reducing size of the frame
row,col = frame.shape
frame = cv2.resize(frame,(col//2,row//2),interpolation=cv2.INTER_CUBIC)
row = row//2
col = col//2
mean = np.zeros([3,row,col],np.float64)
mean[1,:,:] = frame
variance = np.zeros([3,row,col],np.float64)
variance[:,:,:] = 400

omega = np.zeros([3,row,col],np.float64)
omega[0,:,:],omega[1,:,:],omega[2,:,:] = 0,0,1

omega_by_sigma = np.zeros([3,row,col],np.float64)

foreground = np.zeros([row,col],np.uint8)
background = np.zeros([row,col],np.uint8)

gauss_fit_index = np.zeros([row,col])

alpha = 0.3
T = 0.7

while cap.isOpened():
    _,frame = cap.read()
    frame = cv2.resize(frame, (col,row), interpolation=cv2.INTER_CUBIC)
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray = frame_gray.astype(np.float64)

    
    sigma1 = np.sqrt(variance[0])
    sigma2 = np.sqrt(variance[1])
    sigma3 = np.sqrt(variance[2])
    
    sigma = [sigma1,sigma2,sigma3]

    compare_val_1 = cv2.absdiff(frame_gray,mean[0])
    compare_val_2 = cv2.absdiff(frame_gray,mean[1])
    compare_val_3 = cv2.absdiff(frame_gray,mean[2])

    value1 = 2.5 * sigma1
    value2 = 2.5 * sigma2
    value3 = 2.5 * sigma3

    back_index1 = np.where(omega[2]>T)
    back_index2 = np.where(((omega[2]+omega[1])>T) & (omega[2]<T))

    gauss_fit_index1 = np.where(compare_val_1 <= value1)
    gauss_not_fit_index1 = np.where(compare_val_1 > value1)

    gauss_fit_index2 = np.where(compare_val_2 <= value2)
    gauss_not_fit_index2 = np.where(compare_val_2 > value2)

    gauss_fit_index3 = np.where(compare_val_3 <= value3)
    gauss_not_fit_index3 = np.where(compare_val_3 > value3)
    
    gauss_fit_index = [gauss_fit_index1,gauss_fit_index2,gauss_fit_index3]
    gauss_not_fit_index = [gauss_not_fit_index1,gauss_not_fit_index2,gauss_not_fit_index3]

    temp = np.zeros([row, col])
    temp[back_index1] = 1
    temp[gauss_fit_index3] = temp[gauss_fit_index3] + 1
    index3 = np.where(temp == 2)

    temp = np.zeros([row,col])
    temp[back_index2] = 1
    index = np.where((compare_val_3<=value3)|(compare_val_2<=value2))
    temp[index] = temp[index]+1
    index2 = np.where(temp==2)

    match_index = np.zeros([row,col])
    match_index[gauss_fit_index1] = 1
    match_index[gauss_fit_index2] = 1
    match_index[gauss_fit_index3] = 1
    not_match_index = np.where(match_index == 0)

  # for updating least probalble gaussian  with new mean ,variance,omega
    mean[0][not_match_index] = frame_gray[not_match_index]
    variance[0][not_match_index] = 200
    omega[0][not_match_index] = 0.1
# updating the values of mean variance omega
    for i in range(3):
            rho = alpha * norm.pdf(frame_gray[gauss_fit_index[i]], mean[i][gauss_fit_index[i]], sigma[i][gauss_fit_index[i]])
            constant = rho * ((frame_gray[gauss_fit_index[i]] - mean[i][gauss_fit_index[i]]) ** 2)
            mean[i][gauss_fit_index[i]] = (1 - rho) * mean[i][gauss_fit_index[i]] + rho * frame_gray[gauss_fit_index[i]]
            variance[i][gauss_fit_index[i]] = (1 - rho) * variance[i][gauss_fit_index[i]] + constant
            omega[i][gauss_fit_index[i]] = (1 - alpha) * omega[i][gauss_fit_index[i]] + alpha
            omega[i][gauss_not_fit_index[i]] = (1 - alpha) * omega[i][gauss_not_fit_index[i]]

    # normalise omega
    sum = np.sum(omega,axis=0)
    omega = omega/sum

    omega_by_sigma[0] = omega[0] / sigma1
    omega_by_sigma[1] = omega[1] / sigma2
    omega_by_sigma[2] = omega[2] / sigma3

# sorting the mean,vairance,omeha according ot the omega_by_sigma
    index = np.argsort(omega_by_sigma,axis=0)
    omega_by_sigma = np.take_along_axis(omega_by_sigma,index,axis=0)

    mean = np.take_along_axis(mean,index,axis=0)
    variance = np.take_along_axis(variance,index,axis=0)
    omega = np.take_along_axis(omega,index,axis=0)

    frame_gray = frame_gray.astype(np.uint8)

    background[index2] = frame_gray[index2]
    background[index3] = frame_gray[index3]
    cv2.imshow('BACKGROUND',background)
    cv2.imshow('frame',foreground)
    cv2.imshow('frame_gray',frame_gray)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
