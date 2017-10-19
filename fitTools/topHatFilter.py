import numpy as np
from scipy import ndimage#used for tophat filter

def tophat(data, factor):
    '''
    Remove the baseline noise of a spectrum. 
    Note: the tophat filter is a method borrowed from the image processing
    community that treats a 1D graph as a 2D black/white image.

    Usage: tophat(data, factor) with
        - data = array of the data
        - factor = the less the number, the more noise will be removed
    '''
    pntFactor = factor
    struct_pts = int(round(data.size*pntFactor))
    str_el = np.repeat([1], struct_pts)
    tFil = ndimage.white_tophat(data, None, str_el)
    return tFil
