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

    url = 'im2.tif'
    img = plt.imread(url)
    out = reverse_jet(url)

    f, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(img)
    ax1.imshow(skimage.color.rgb2grey(img), cmap=plt.cm.gray)
    ax2.imshow(out, cmap=plt.cm.gray)
    ax0.set_title("original")
    ax1.set_title("grayscale")
    ax2.set_title("reversed jet")
    for ax in [ax0, ax1, ax2]:
      ax.axis('off')

    plt.show()
