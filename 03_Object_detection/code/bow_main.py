import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm



def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    vPoints = None  # numpy array, [nPointsX*nPointsY, 2]

    # todo

    # create the nPointsX*nPointsY gridpoints over the image with *border pixels margin
    width = img.shape[1] # width = x
    height = img.shape[0] # height = y

    # make gridpoint at every 10th pixel, starting from border to width-border, height-border respectively
    x = np.linspace(border, width-border, num = nPointsX, endpoint=True, dtype=int)
    y = np.linspace(border, height-border, num = nPointsY, endpoint=True, dtype=int)
    xv,yv = np.meshgrid(x,y)

    # get coordinates [nPointsX*nPointsY, 2]
    vPoints = np.array([])
    grid_length = int(xv.shape[0]*xv.shape[1])
    for rows in range(xv.shape[0]):
        vPoints = np.append(vPoints, np.vstack(np.array(list(zip(xv[rows],yv[rows])))))
    vPoints = vPoints.reshape((grid_length,2))

    return vPoints



def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    nBins = 8
    w = cellWidth
    h = cellHeight

    grad_x = cv2.Sobel(img, cv2.CV_32F, dx=1, dy=0, ksize=1) # changed from cv2.CV_16S to CV_32F for cartToPolar 
    grad_y = cv2.Sobel(img, cv2.CV_32F, dx=0, dy=1, ksize=1) # changed from cv2.CV_16S to CV_32F for cartToPolar

    # compute the magnitude and angle
    mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True) # returns angle from 0 to 360 degrees

    descriptors = [] # list of descriptors for the current image, each entry is one 128-d vector for a grid point
    # for every grid point (in vPoints) compute 4x4 cells, each cell contains 16 pixel
    for i in range(len(vPoints)):
        center_x = round(vPoints[i, 0])
        center_y = round(vPoints[i, 1])

        desc = np.array([]) 
    
        for cell_y in range(-2, 2):
            for cell_x in range(-2, 2):
                start_y = center_y + (cell_y) * h
                end_y = center_y + (cell_y + 1) * h

                start_x = center_x + (cell_x) * w
                end_x = center_x + (cell_x + 1) * w

                # todo
                
                # compute the histogram

                # size of each bin 360/8 = 45
                # last bin is between 360 and 0 because here 0 = 360
                bins = [0,45,90,135,180,225,270,315,0]
                hist = [0,0,0,0,0,0,0,0]
                for i in range(1,nBins+1):
                    for y in range(start_y,end_y):
                        for x in range(start_x,end_x):
                            # first, angle of each gradient gets assigned to corresponding bin in the histogram
                            # second, magnitude of each gradient is split proportional between lower and upper bin and added to the bins value
                            if bins[i-1] <= abs(angle[y,x]) <= bins[i]:
                                prop = (angle[y,x] - bins[i-1]) / 45
                                hist[i-1] += mag[y,x]*prop
                                hist[i] += mag[y,x]*(1-prop)
                # concatenate 16 histograms, each has length 8            
                desc = np.append(desc,hist)
                # flatten array
                desc = np.ravel(desc) #[128]
        descriptors.append(desc) # [nPointsX*nPointsY, 128], descriptor for the current image (100 grid points)

    descriptors = np.asarray(descriptors)
    return descriptors #[100,128]


def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    vFeatures = []  # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    # Extract features for all image
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # Collect local feature points for each image, and compute a descriptor for each local feature point
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        descriptors = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        vFeatures.append(descriptors)

    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])  # [n_imgs*n_vPoints, 128]
    print('number of extracted features: ', len(vFeatures))


    # Cluster the features using K-Means
    print('clustering ...')
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter).fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_  # [k, 128]
    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    histo = None
    # assign descriptors to cluster center
    idx, dist = findnn(vFeatures,vCenters) # vFeatures = [100,128]
    # create array of N bins
    idx_centers = np.arange(len(vCenters))
    # count of assigned descriptors per visual word / cluster center in histogram
    histo, bin_edges = np.histogram(idx, bins=idx_centers, density=False)
    # one histogram per image containing N entries
    return histo


def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    vBoW = []
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # extract 100 features per image, create bow histogram for each img over the features
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        descriptors = descriptors_hog(img, vPoints, cellWidth, cellHeight) # descriptors = [100,128]
        hist = bow_histogram(descriptors,vCenters) # hist = [1,k]
        vBoW.append(hist)
    vBoW = np.asarray(vBoW)  # [n_imgs, k]
    return vBoW



def bow_recognition_nearest(histogram,vBoWPos,vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    DistPos, DistNeg = None, None

    # Find the nearest neighbor in the positive and negative sets and decide based on this neighbor
    # idx and distance of closest feature from test histogram to pos/neg training histograms
    idx, DistPos = findnn(histogram,vBoWPos)
    idx, DistNeg = findnn(histogram,vBoWNeg)

    if (DistPos < DistNeg):
        sLabel = 1
    else:
        sLabel = 0
    return sLabel





if __name__ == '__main__':
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'


    k = 30  # found empirically
    numiter = 60  # found empirically

    print('creating codebook ...')
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

    print('creating bow histograms (pos) ...')
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print('creating bow histograms (neg) ...')
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print('creating bow histograms for test set (pos) ...')
    vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print('testing pos samples ...')
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWPos_test[i:(i+1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print('test pos sample accuracy:', acc_pos)

    # test neg samples
    print('creating bow histograms for test set (neg) ...')
    vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print('testing neg samples ...')
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print('test neg sample accuracy:', acc_neg)
