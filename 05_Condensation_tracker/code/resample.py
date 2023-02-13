import numpy as np


def resample(particles, particles_w):
    """ resample particles based on their weights with multinomial resampling
        returns resampled particles,resampled particles_w"""

    # associate each weight with its "importance"
    N = particles.shape[0]
    cumulative_sum = np.cumsum(particles_w)
    cumulative_sum[-1] = 1.0 # avoid rounding errors
    # generate N random numbers in [0,1] for "picking" N times
    random_numbers = np.random.random(N)
    # identify weight that is closest to the random pick
    idx = np.searchsorted(cumulative_sum, random_numbers)
 
    # resample particles and weights according to idx
    particles_new = particles[idx]
    particles_w_new = particles_w[idx]
    # normalize resampled weights
    particles_w_new = particles_w_new/np.sum(particles_w_new)

    return particles_new,particles_w_new