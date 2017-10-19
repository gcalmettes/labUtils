import numpy as np
import sys

def varmap(image, winsize):
    """
    Compute the variance map of an image.
    INPUTS:
        - image = image to analyze
        - winsize = size of the sliding square window. Must be an odd number.
    """
    # ensure that window size is an odd number
    if np.mod(winsize, 2)==0:
        raise ValueError("The size of the sliding window must be an odd number")
        

    # size of the image
    nr, nc = image.shape

    # create empty images of means and variance
    #im_mean = np.zeros((nr, nc))
    im_var = np.zeros((nr, nc))

    # number of pixels around center of the window
    r = (winsize-1)/2

    # compute the map of mean values
    #for i in range(r,nr-r):
    #    for j in range(r,nc-r):
    #        im_mean[i, j] = np.mean(image[i-r:i+r+1, j-r:j+r+1])
    #        #s = 0.
    #        #for a in range(-r,r+1):
    #        #    for b in range(-r, r+1):
    #        #        s = s+image[i+a,j+b]
    #        #im_mean[i, j] = s/(winsize*winsize)

    # compute the map of variance values
    for i in range(r,nr-r):
        for j in range(r,nc-r):
            im_var[i, j] = np.var(image[i-r:i+r+1, j-r:j+r+1])
            #s = 0.
            #for a in range(-r,r+1):
            #    for b in range(-r, r+1):
            #        s = s+(image[i+a,j+b]-im_mean[i, j])*2
            #im_var[i, j] = s/((winsize*winsize)-1)
            #var = s/((winsize*winsize)-1)
            #if (var>alpha):
            #    im_var[i, j] = 255
            #else:
            #    im_var[i, j] = 0

    return im_var
