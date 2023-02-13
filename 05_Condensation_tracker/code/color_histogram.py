import numpy as np


def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):

    # crop out bounding box    
    cropped_image = frame[ymin:ymax,xmin:xmax]
    R = cropped_image[:,:,0]
    G = cropped_image[:,:,1]
    B = cropped_image[:,:,2]

    # R histogram
    histo_R, bin_edges_R = np.histogram(R, bins=hist_bin, density=False)
    # G histogram
    histo_G, bin_edges_G = np.histogram(G, bins=hist_bin, density=False)
    # B histogram
    histo_B, bin_edges_B = np.histogram(B, bins=hist_bin, density=False)

    # create RGB histogram
    histo = [histo_R,histo_G,histo_B]
    # normalize histogram counts by the total amount of pixel contained in the bounding box
    denom = np.sum(histo)
    if denom != 0:
        return histo/denom
    if denom == 0:
        return histo