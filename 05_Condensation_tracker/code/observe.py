import numpy as np
from chi2_cost import chi2_cost
from color_histogram import color_histogram


def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    """ computes color histogram for each particle's bounding box and updates particle weights according to distance between 
        computed histogram and target histogram 
        particles: center of bounding boxes
        frame: RGB image
        bb_height: height of particles bounding box
        bb_width: widht of particles bounding box
        hist_bin: num of bins
        hist: target histogram
        sigma_observe: noise
        returns: particle_w weights """

    # compute color histogram for each particle
    particles_w = []
    gaussian_normalizer = 1/(np.sqrt(2*np.pi)*sigma_observe)
    for i in range(particles.shape[0]):
        xmin = particles[i,0] - bbox_width/2
        xmax = particles[i,0] + bbox_width/2
        ymin = particles[i,1] - bbox_height/2
        ymax = particles[i,1] + bbox_height/2
        histo_bb = color_histogram(xmin=int(xmin), ymin=int(ymin), xmax=int(xmax), ymax=int(ymax), frame=frame, hist_bin=hist_bin)
        # compute distance from each bounding box histogram to the target histogram
        dist = chi2_cost(hist, histo_bb)
        # compute weights with gaussian function
        weights = gaussian_normalizer * np.exp(-(dist*dist)/(2*(sigma_observe*sigma_observe)))
        particles_w.append(weights)
    # return normalized weights
    particles_w = (particles_w/np.sum(particles_w)).reshape(particles.shape[0],1)
    return particles_w
    