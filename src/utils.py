import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import color, transform, restoration

DATA_PATH = "../data"


def resize(img, factor):
    num = int(-np.log2(factor))
    for i in range(num):
        img = 0.25 * (
            img[::2, ::2, ...]
            + img[1::2, ::2, ...]
            + img[::2, 1::2, ...]
            + img[1::2, 1::2, ...]
        )
    return img


def crop(X):
    return X[starti:endi, startj:endj]


def crop_array(X, nx, ny, fill_factor):
    import math

    w, h = X.shape
    X_out = np.zeros(X.shape)
    for i in range(w // nx):
        start_i, end_i = w // nx * i, math.floor(w // nx * (i + fill_factor))
        for j in range(h // ny):
            start_j, end_j = h // ny * j, math.floor(h // ny * (j + fill_factor))
            X_out[start_i:end_i, start_j:end_j] = X[start_i:end_i, start_j:end_j]
    return X_out


def load_psf():
    return np.load(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_PATH, "psf.npy")
    )


def load_diffuser_image(n):
    return np.load(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            DATA_PATH,
            "diffuser",
            f"im{n}.npy",
        )
    )


def load_lensed_image(n):
    return np.load(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            DATA_PATH,
            "lensed",
            f"im{n}.npy",
        )
    )


def display_array(arr):
    plt.figure()
    plt.gray()
    plt.imshow(arr)
    plt.show()


def deconvolve(psf, measured_data):
    deconvolved, _ = restoration.unsupervised_wiener(
        color.rgb2gray(measured_data), color.rgb2gray(psf)
    )
    return deconvolved


if __name__ == "__main__":
    for i in range(1, 20):
        scale_factor = 0.012 * i
        psf = transform.resize(load_psf(), (scale_factor * 1080, scale_factor * 1920))
        measured_data = load_diffuser_image(10)

        result = deconvolve(psf, measured_data)
        display_array(result)
