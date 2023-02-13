import numpy as np


def propagate(particles, frame_height, frame_width, params):
    """ if model = 0:
        particles: array containing the center of the bounding box with shape (num_particles,2)
        frame_height: height of bounding box
        frame_width: width of bounding box
        params: {model=0,sigma_position}
        returns: s_new: array with propagated particles with shape (num_particles,2)
        if model = 1:
        particles: [x,y,x',y'] where x,y center of bounding box location and x',y' constant velocities, shape (num_particles,4)
        frame_height: height of bounding box (here y = 120)
        frame_width: width of bounding box (here x = 160)
        params: {model=1,sigma_position,sigma_velocity,initial_velocity}
        returns: s_new: array with propagated particles with shape (num_particles,4)"""
    
    model = params['model']
    #(0) no motion at all i.e. just noise model for x'= y'= 0, hence s = [x,y]
    if model == 0:
        sigma_position = params['sigma_position']
        noise = sigma_position * np.random.randn(particles.shape[0], particles.shape[1])
        A = np.array([[0,1],[1,0]])
        x = particles[:,0]
        y = particles[:,1]
        s_old = np.array([x,y])
        # propagate each sample with model given by s_t = A*s_t-1 + noise
        s_new = np.matmul(A,s_old)
        s_x = s_new[0,:].reshape((particles.shape[0],1))
        s_y = s_new[1,:].reshape((particles.shape[0],1))
        s_new = np.hstack((s_y,s_x))
        s_new += noise

        # make sure new bounding box center lies within the image frame
        for i in range(particles.shape[0]):
            if s_new[i,0] > frame_width:
                s_new[i,0] = frame_width
            if s_new[i,0] < 0:
                s_new[i,0] = 0
            if s_new[i,1] > frame_height:
                s_new[i,1] = frame_height
            if s_new[i,1] < 0:
                s_new[i,1] = 0
        return s_new

    # (1) constant velocity motion model for s = [x,y,x',y'] where x,y center of bounding box location and x',y' constant velocities                     
    else:
        sigma_position, sigma_velocity = params['sigma_position'], params['sigma_velocity']
        noise = [sigma_position,sigma_position,sigma_velocity, sigma_velocity] * np.random.randn(particles.shape[0], particles.shape[1])
        initial_velocity = params['initial_velocity']
        x = particles[:,0]
        y = particles[:,1]
        #delta_x = particles[:,2]
        delta_x = initial_velocity[0]
        #delta_y = particles[:,3]
        delta_y = initial_velocity[1]
        s_old = np.array([x,y,delta_x,delta_y])

        A = np.array([[1,0,1,0],
                    [0,1,0,1],
                    [0,0,1,0],
                    [0,0,0,1]])          
        # propagate each sample with model given by s_t = A*s_t-1 + noise
        s_new = np.matmul(A,s_old)
        s_x = s_new[0].reshape((particles.shape[0],1))
        s_y = s_new[1].reshape((particles.shape[0],1))
        s_xv = s_new[2].reshape((particles.shape[0],1))
        s_yv = s_new[3].reshape((particles.shape[0],1))
        s_new = np.hstack((s_x,s_y,s_xv,s_yv))
        s_new += noise

        # make sure new bounding box center lies within the image frame
        for i in range(particles.shape[0]):
            if s_new[i,0] > frame_width:
                s_new[i,0] = frame_width
            if s_new[i,0] < 0:
                s_new[i,0] = 0
            if s_new[i,1] > frame_height:
                s_new[i,1] = frame_height
            if s_new[i,1] < 0:
                s_new[i,1] = 0
        return s_new