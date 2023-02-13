import numpy as np

def filter_keypoints(img, keypoints, patch_size = 9):
    # TODO: Filter out keypoints that are too close to the edges

    # filter out keypoints that lie within a 5 pixel margin at the image edge

    h = img.shape[0] - 5 # (.,y) l1 and l2. Height corresponds to y position in keypoints
    w = img.shape[1] - 5 # (x,.) l1 and l2. Width corresponds to x position in keypoints

    # delete all [x,y] where x < w
    mask0 = keypoints[:,0] < w
    keypoints = np.compress(mask0, keypoints, axis=0)
    # delete all [x,y] where x < 5
    mask1 = keypoints[:,0] > 5
    keypoints = np.compress(mask1, keypoints, axis=0)
    # delete all [x,y] where y < w
    mask2 = keypoints[:,1] < h
    keypoints = np.compress(mask2, keypoints, axis=0)
    # delete all [x,y] where y < 5
    mask3 = keypoints[:,1] > 5
    keypoints = np.compress(mask3, keypoints, axis=0)

    return keypoints

# The implementation of the patch extraction is already provided here
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc


