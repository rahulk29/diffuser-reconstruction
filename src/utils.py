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


def next_pow2(n):
    return int(2 ** np.ceil(np.log2(n)))


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


def display_array(arr, title):
    plt.figure()
    plt.imshow(arr, cmap="gray")
    plt.title(title)
