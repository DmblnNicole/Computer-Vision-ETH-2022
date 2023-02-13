import numpy as np
import scipy

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1] 
    # TODO: implement this function please

    # compare the i-th feature of desc1 with the j-th feature of desc2
    # store each distance in the ij-th entry of q1 x q2 matrix 'distances'
    distances = scipy.spatial.distance.cdist(desc1, desc2, 'sqeuclidean')

    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here

        # get idx of minimum distance of i-th feature in each row
        Imin = np.argmin(distances, axis=1)
        # create index array with len = q1
        i = np.arange((q1))
        # matches = [[0,Imin[0]], [1, Imin[1]], ... ,[q1,Imin[q1]]
        Imin = Imin.reshape(len(Imin),1)
        i = i.reshape(len(i),1)
        matches = np.hstack((i,Imin))

    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        
        # minimum of distance btw (desc1,desc2) saved in index array matches_ow
        min1 = np.min(distances, axis=1)
        Imin1 = np.argmin(distances, axis=1)
        i = np.arange((q1))
        Imin1 = Imin1.reshape(len(Imin1),1)
        i = i.reshape(len(i),1)
        matches_ow = np.hstack((i,Imin1))

        # minimum of distance btw (desc2,desc1) saved in index array matches_wo
        min2 = np.min(distances, axis=0)
        Imin2 = np.argmin(distances, axis=0)
        j = np.arange((q2))
        Imin2 = Imin2.reshape(len(Imin2),1)
        j = j.reshape(len(j),1)
        matches_wo = np.hstack((j,Imin2))

        # filter out mutual matches
        matches = np.array([])
        for i in range(q1):
            for j in range(q2):
                # find for every ow-match [0,1] the match [1,0]
                if np.sum(matches_ow[i] - matches_wo[j]) == 0: 
                    # check if for this matches min(desc1,desc2) == min(desc2,desc1)
                    if min1[i] == min2[j]:
                        matches = np.append(matches, matches_ow[i])
        matches = matches.reshape((int(0.5*len(matches)),2))
        matches = matches.astype(int)

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # ow match valid if the ratio between the first and the second 
        # nearest neighbor is lower than a given threshold

        # minimum of distance btw (desc1,desc2) saved in index array matches_ow
        Imin = np.argmin(distances, axis=1)
        i = np.arange((q1))
        Imin = Imin.reshape(len(Imin),1)
        i = i.reshape(len(i),1)
        matches_ow = np.hstack((i,Imin))

        # find first and second minimum of each ow-match
        part = np.partition(distances,2,axis=1)
        # first element from partially sorted array 'distances is 1st minimum
        firstMin = part[:,0]
        # second element from partially sorted array 'distances' is 2nd minimum
        secondMin = part[:,1]
        # compute ratio between first and second minimum
        ratio = firstMin / secondMin
        # retain only ow-matches that have a ratio smaller than ratio_thresh
        mask = ratio < ratio_thresh
        matches = matches_ow[mask]
    else:
        raise NotImplementedError
    return matches

