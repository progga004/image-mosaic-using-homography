import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# 1) gaussian smoothing
def gaussian_smoothing(img,sigma):
    kernel_size=6*sigma+1
    if(kernel_size%2==0):
      kernel_size=kernel_size+1
    else:
       kernel_size=kernel_size
    return cv.GaussianBlur(img,(kernel_size,kernel_size),sigma)

#2)Gradient Gx,Gy
def gradient(img):
   kernel_x = np.array([[-1, 0, 1]])
   kernel_y = np.array([[-1], [0], [1]])

    # Applying 1D convolution for Gx along the x-direction
   Gx = cv.filter2D(img, -1, kernel_x)

    # Applying 1D convolution for Gy along the y-direction
   Gy = cv.filter2D(img, -1, kernel_y)
   return Gx,Gy

#3)a) Compute the r score
def r_score(Gx,Gy,N,k):
   gx2 = Gx * Gx
   gxy = Gx * Gy
   gy2 = Gy * Gy
   sx2 = cv.boxFilter(gx2, -1, (N,N))
   sxy = cv.boxFilter(gxy, -1, (N,N))
   sy2 = cv.boxFilter(gy2, -1, (N,N))
   # Compute Harris response
   detH = sx2 * sy2 - sxy * sxy
   traceH = sx2 + sy2
   R = detH - k * traceH * traceH
   return R

#4)b) Non maximum suppression

def extract_corners(R, threshold, max_corners):
    corners = []
    for i in range(max_corners):
        max_value = np.max(R)
        if max_value < threshold:
            break
        max_index = np.unravel_index(np.argmax(R), R.shape)
        corners.append(max_index)
        R[max_index] = 0
    return corners
    



