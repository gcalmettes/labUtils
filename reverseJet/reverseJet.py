from skimage import io as sio
import skimage

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def reverse_jet(img):
    img = skimage.img_as_float(sio.imread(img))
    jet = plt.cm.jet

    jet._init()
    lut = jet._lut[..., :3]

    z = img - lut[:, None, None, :]
    z *= z
    d = z.sum(axis=-1)

    out = d.argmin(axis=0)
    return out

if __name__ == '__main__':

    out = reverse_jet('img2.tif')

    f, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(img, cmap=plt.cm.gray)
    ax1.imshow(out, cmap=plt.cm.gray)
    ax2.imshow(out, cmap=plt.cm.jet)

    plt.show()
