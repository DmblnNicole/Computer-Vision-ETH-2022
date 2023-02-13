import time
import os
import random
import math
import torch
import numpy as np
#from tqdm import tqdm
# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    """ compute euclidian distance between x and all other points X
    :param: x: single data point [3]
    :param: X: data set [3675,3]
    :returns: 1-d np-array of distances of x to each point in X of size len(X) """
    dist = np.sqrt(((x[0]-X[:,0])**2 +(x[1]-X[:,1])**2 + (x[2]-X[:,2])**2))
    return dist

def distance_batch(x, X):
    dist = torch.cdist(x,X,p=2)
    return dist

def gaussian(dist, bandwidth):
    """ compute weights of each point according to distance 
    :param: dist: distance btw x and X
    :param: bandwidth: std deviation of the gaussian
    :returns: 1-d np-array of weights of size len(X) """
    # weight decreases when point is far away from mean
    weight = np.exp(-(dist*dist)/(2* bandwidth**2))
    return weight

def update_point(weight, X):
    """ compute weighted mean of all data points in training set X
    :param: weight: gaussian weight
    :param: X: training set """
    denom = torch.sum(weight)
    r = torch.sum((weight/denom)*X[:,0])
    g = torch.sum((weight/denom)*X[:,1])
    b = torch.sum((weight/denom)*X[:,2])
    mean = torch.tensor([r.item(),g.item(),b.item()])
    return mean

def update_point_batch(weight, X):
    denom = torch.sum(weight,dim=0)
    mean = torch.matmul(weight,X)/denom.unsqueeze(1)
    return mean

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    print(X_)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    dist = distance_batch(X, X)
    weight = gaussian(dist, bandwidth)
    X_ = update_point_batch(weight, X)
    return X_

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        print(_)
        #X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster 0.25

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)
result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)