import numpy as np
import cv2
from scipy.stats import norm
def norm_pdf(x, mu, sigma):
    return (1.0 / (sigma * ((2.0 * np.pi)**(1/2)) )* np.exp(-1.0 * (x - mu)**2 / (2.0 * (sigma**2))))

def common_index(x,y):
        black1 = np.zeros((row,col))
        black1[x] = 1
        black1[y] = black1[y]+1
        common = np.where(black1==2)
        #print('comon',common)
        return common

# 1 is most probable and 3 is least probable

cap = cv2.VideoCapture(0)
_,mean_g1 = cap.read()
mean_g1 = cv2.cvtColor(mean_g1,cv2.COLOR_BGR2GRAY)
_,mean_g2 = cap.read()
mean_g2 = cv2.cvtColor(mean_g2,cv2.COLOR_BGR2GRAY)
_,mean_g3 = cap.read()
mean_g3 = cv2.cvtColor(mean_g3,cv2.COLOR_BGR2GRAY)
mean_g1 = mean_g1.astype(np.float64)
mean_g2 = mean_g2.astype(np.float64)
mean_g3 = mean_g3.astype(np.float64)

row,col= mean_g1.shape

variance_g1 = np.ones([row,col],np.float64)
variance_g2 = np.ones([row,col],np.float64)
variance_g3 = np.ones([row,col],np.float64)
variance_g1[:,:],variance_g2[:,:],variance_g3[:,:] = 800,400,200

omega_g1 = np.ones([row,col],np.float64)
omega_g2 = np.ones([row,col],np.float64)
omega_g3 = np.ones([row,col],np.float64)
omega_g1[:,:],omega_g2[:,:],omega_g3[:,:] = 0.8,0,0


foreground = np.zeros([row,col],np.uint8)
background = np.zeros([row,col],np.uint8)
T = np.zeros([row,col],np.uint8)


gauss_fit_index = np.zeros([row,col])

alpha = 0.3
T[:row,:col] = 0.6

while cap.isOpened():
    _,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray = frame_gray.astype(np.float64)
   # print(frame_gray.shape)

    sigma1 = np.sqrt(variance_g1)
    sigma2 = np.sqrt(variance_g2)
    sigma3 = np.sqrt(variance_g3)

    compare_val_1 = cv2.absdiff(frame_gray,mean_g1)
    compare_val_2 = cv2.absdiff(frame_gray,mean_g2)
    compare_val_3 = cv2.absdiff(frame_gray,mean_g3)

    value1 = compare_val_1 / sigma1
    value2 = compare_val_2/ sigma2
    value3 = compare_val_3/sigma3
    #print('value',value2)
    gauss_fit_index1 = np.where( value1<2.5)  
    gauss_not_fit_index1 = np.where(value1>2.5)
    print('gauss_fit_index',gauss_fit_index1)
    print(' noot gauss_fit_index',gauss_not_fit_index1)  

    gauss_fit_index2 = np.where(compare_val_2 <= value2)
    gauss_not_fit_index2 = np.where(compare_val_2 > value2)
    gauss_fit_index3 = np.where(compare_val_3 <= value3)
    gauss_not_fit_index3 = np.where(compare_val_3 > value3)

    match_index = np.zeros([row,col])
    match_index[gauss_fit_index1] = 1
    match_index[gauss_fit_index2] = 1
    match_index[gauss_fit_index3] = 1
    not_match_index = np.where(match_index == 0)    
    rho = alpha * norm.pdf(frame_gray, mean_g1, sigma1)
    mean_g1[gauss_fit_index1] = (1 - rho[gauss_fit_index1]) * mean_g1[gauss_fit_index1] + rho[gauss_fit_index1] * frame_gray[gauss_fit_index1]
    variance_g1[gauss_fit_index1] = (1 - rho[gauss_fit_index1]) * (sigma1[gauss_fit_index1] ** 2) + rho[gauss_fit_index1] * ((frame_gray[gauss_fit_index1] - mean_g1[gauss_fit_index1]) ** 2)
    omega_g1[gauss_fit_index1] = (1 - alpha) * omega_g1[gauss_fit_index1] + alpha
    omega_g1[gauss_not_fit_index1] = (1 - alpha) * omega_g1[gauss_not_fit_index1]

    rho = alpha * norm.pdf(frame_gray, mean_g2, sigma2)
    mean_g2[gauss_fit_index2] = (1 - rho[gauss_fit_index2]) * mean_g2[gauss_fit_index2] + rho[gauss_fit_index2] * frame_gray[gauss_fit_index2]
    variance_g2[gauss_fit_index2] = (1 - rho[gauss_fit_index2]) * (sigma2[gauss_fit_index2] ** 2) + rho[gauss_fit_index2] * ((frame_gray[gauss_fit_index2] - mean_g2[gauss_fit_index2]) ** 2)
    omega_g2[gauss_fit_index2] = (1 - alpha) * omega_g2[gauss_fit_index2] + alpha
    omega_g2[gauss_not_fit_index2] = (1 - alpha) * omega_g2[gauss_not_fit_index2]

    rho = alpha * norm.pdf(frame_gray, mean_g3, sigma3)
    mean_g3[gauss_fit_index3] = (1 - rho[gauss_fit_index3]) * mean_g3[gauss_fit_index3] + rho[gauss_fit_index3] * frame_gray[gauss_fit_index3]
    variance_g3[gauss_fit_index3] = (1 - rho[gauss_fit_index3]) * (sigma3[gauss_fit_index3] ** 2) + rho[gauss_fit_index3] * ((frame_gray[gauss_fit_index3] - mean_g3[gauss_fit_index3]) ** 2)
    omega_g3[gauss_fit_index3] = (1 - alpha) * omega_g1[gauss_fit_index3] + alpha
    omega_g3[gauss_not_fit_index3] = (1 - alpha) * omega_g1[gauss_not_fit_index3]
    

    mean_g3[not_match_index] = frame_gray[not_match_index]
    variance_g3[not_match_index] = 200
    omega_g3[not_match_index] = 0.01
                
    # normalise omega
    sumx = omega_g1 + omega_g2 + omega_g3
    omega_g1 = omega_g1/sumx
    omega_g2 = omega_g2/sumx
    omega_g3 = omega_g3/sumx
    omega_by_sigma_1 = omega_g1/sigma1
    omega_by_sigma_2 = omega_g2 / sigma2
    omega_by_sigma_3 = omega_g3 / sigma3 
    
    # chnaging the gaussian with less probability 
    # first_gauss,second_gauss,third_gauss,are those index of particular gaussian with least value
    
    first_gauss = np.where((omega_by_sigma_1 < omega_by_sigma_2) & (omega_by_sigma_1 < omega_by_sigma_3))        
    second_gauss = np.where((omega_by_sigma_2 < omega_by_sigma_1) & (omega_by_sigma_2 <omega_by_sigma_3)) 
    third_gauss = np.where((omega_by_sigma_3 < omega_by_sigma_1) & (omega_by_sigma_3 <omega_by_sigma_2))
                
    changing_index_1 = np.where(first_gauss==not_match_index) 
    if changing_index_1[0]:                       
        changing_index_1 = first_gauss[changing_index_1]
            # the changing_Index_1 is index which does not match with any index and has first_gaussian as least value w/sigma              
        mean_g1[changing_index_1]= frame_gray
        variance_g1[changing_index_1] = 200 
        omega_g1[changing_index_1] = 0.001      

    changing_index_2 = np.where(first_gauss==not_match_index)
    if changing_index_2[0]:
            changing_index_2 = second_gauss[changing_index_2]
            # the changing_Index_2 is index which does not match with any index and has first_gaussian as least value w/sigma              
            mean_g2[changing_index_2]= frame_gray
            variance_g2[changing_index_2] = 200 
            omega_g2[changing_index_2] = 0.001      

    changing_index_3 = np.where(first_gauss==not_match_index)
    if changing_index_3[0]:
        changing_index_3 = third_gauss[changing_index_3]
        # the changing_Index_3 is index which does not match with any index and has first_gaussian as least value w/sigma              

        mean_g3[changing_index_3]= frame_gray
        variance_g3[changing_index_3] = 200 
        omega_g3[changing_index_3] = 0.001      
        
# sorting the gaussian according to the omega/sigma  and assign the  background and forground
    order_omega = np.array([omega_by_sigma_1,omega_by_sigma_2,omega_by_sigma_3])
    omega = np.array([omega_g1,omega_g2,omega_g3])
    mean =np.array( [ mean_g1,mean_g2,mean_g3])
    variance = np.array([variance_g1,variance_g2,variance_g3])
    index_order = np.argsort(order_omega,axis = 0)
    sorted_omega= np.take_along_axis(order_omega,index_order,axis =0)
    sorted_omega= np.take_along_axis(omega,index_order,axis =0)
    sorted_mean = np.take_along_axis(mean,index_order,axis =0)
    sorted_variance = np.take_along_axis(variance,index_order,axis =0)
    
    value1 = cv2.absdiff(frame_gray,sorted_mean[0])
    value1 = value1/np.sqrt(sorted_variance[0])    
    value2 = cv2.absdiff(frame_gray,sorted_mean[1])
    value2 = value2/np.sqrt(sorted_variance[1])    
    value3 = cv2.absdiff(frame_gray,sorted_mean[2])
    value3 = value3/np.sqrt(sorted_variance[2]) 
    #print('value',value3)   
   # background = np.where(value3<2.5,frame_gray,np.uint8([0]))
    #print(in_3)
    in_3 = np.where(value3<2.5)

    cv2.imshow('back',background)
    in_1 = np.where(value1<2.5)
    in_1_2 = np.where((value1<2.5) | (value2<2.5))
    fit_3 = np.where(sorted_omega[2]>T)
    in_3_2 = np.where((value3<2.5) | (value2<2.5))    
    common_3 = common_index(fit_3,in_3)
    common_1_2 = common_index(fit_3,in_1_2)
    background[common_3] =frame_gray[common_3]
    foreground[common_1_2]= frame_gray[common_1_2]    

    fit_2_1 =np.where((sorted_omega[2]+sorted_omega[1] >T) &(sorted_omega[2] < T))
    common2_3 = common_index(fit_2_1,in_3_2)
    common2_1 = common_index(fit_2_1,in_1)   
    background[common2_3] =frame_gray[common2_3]
    foreground[common2_1] = frame_gray[common2_1]

    frame_gray = frame_gray.astype(np.uint8)
    foreground = foreground.astype(np.uint8)
    cv2.imshow('s',foreground)    

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
