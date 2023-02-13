import numpy as np
import scipy
from scipy import signal
import cv2

# Harris corner detector
def extract_harris(img, sigma = 0.5, k = 0.05, thresh = 1e-4):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # Compute image gradients
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    
    # compute derivatives in x and y direction, respectively
    gx = np.array([[0,0,0],[-1,0,1],[0,0,0]])
    gy = np.array([[0,-1,0],[0,0,0], [0,1,0]])
    Ix = signal.convolve2d(img, gx, mode="same")
    Iy = signal.convolve2d(img, gy, mode="same")

    # preparation for auto-correlation matrix
    Ixx = Ix*Ix
    Iyy = Iy*Iy
    Ixy = Ix*Iy

    # Compute local auto-correlation matrix
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    
    wIxx = cv2.GaussianBlur(Ixx,(0,0),sigma,cv2.BORDER_REPLICATE)
    wIyy = cv2.GaussianBlur(Iyy,(0,0),sigma,cv2.BORDER_REPLICATE)
    wIxy = cv2.GaussianBlur(Ixy,(0,0),sigma,cv2.BORDER_REPLICATE)

    # Compute Harris response function
    # TODO: compute the Harris response function C here

    # closed form matrix notation of Harris response function C
    traceM = wIxx + wIyy
    detM = wIxx * wIyy - wIxy * wIxy
    C = detM - k * traceM**2

    # Detection with threshold
    # TODO: detection and find the corners here
    # For the local maximum check, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    
    # slides 3x3 window over response img C and
    # saves maximum pixels and overwrites non maximum pixel with max in CMax
    CMax = scipy.ndimage.maximum_filter(C, size=(3,3))

    corners = np.array([[]])
    x = np.array([])
    y = np.array([])
    for rows in range(len(C)):
        for cols in range(len(C)):
            # check if response is greater than thresh and if it 
            # corresponds to the max value in 3x3 window
            if C[rows,cols] > thresh and C[rows,cols] == CMax[rows,cols]:
                x = np.append(x,cols)
                y = np.append(y,rows)
    x = x.reshape(len(x),1)
    y = y.reshape(len(y),1)
    corners = np.hstack((x,y))
    corners = corners.astype(int)
    return corners, C
