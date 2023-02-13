import numpy as np

def estimate(particles, particles_w):
    """ estimate mean of the state vectors according to their weights """
    #model = 0
    if particles.shape[1] == 2:
        product = particles*particles_w
        x = np.sum(product[:,0])
        y = np.sum(product[:,1])
        mean_state = [x,y]
        return mean_state

    #model = 1
    if particles.shape[1] == 4:
        product = particles*particles_w
        x = np.sum(product[:,0])
        y = np.sum(product[:,1])
        xv = np.sum(product[:,2])
        yv = np.sum(product[:,3])
        mean_state = [x,y,xv,yv]
        return mean_state
