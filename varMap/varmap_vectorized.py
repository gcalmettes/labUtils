import numpy as np
import scipy as sp

def variancemap(im, k):
    """
    Compute the variance on the neighborhood of the image pixels. The
    neighborhood is given by the binary matrix (k).

    OUTPUT:
        The image of the neighborhood variance for each image pixel
    INPUT:
        im: image
        k: matrix (kernel). Binary matrix where 1 values denotes a neighbor of
        the pixel at central element. ex: np.ones((3, 3))
    """
    # Ensure that inputs are arrays
    im = np.asarray(im).astype(np.float64)
    k = np.asarray(k).astype(bool)[::-1, ::-1]
    
    # Calcul of the neighborhood variance based on the crafty formula for variance:
    # variance = (sum of the square)/n - (square of the sums)/n*n

    n = np.float(np.sum(k))

    # (sum of the square)/n of each matrix neighborhood
    soa = sp.signal.convolve2d(im**2, k/n, mode='same', boundary='wrap')

    # square of the (sum/n)
    aos = sp.signal.convolve2d(im, k/n, mode='same', boundary='wrap')**2

    imvar = soa-aos

    return imvar
