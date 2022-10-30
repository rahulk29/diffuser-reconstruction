import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import color, transform, restoration

DATA_PATH = "../data"

def load_psf():
    return np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_PATH, 'psf.npy'))

def load_diffuser_image(n):
    return np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_PATH, 'diffuser', f'im{n}.npy'))

def load_lensed_image(n):
    return np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_PATH, 'lensed', f'im{n}.npy'))

def display_array(arr):
    plt.figure()
    plt.gray()
    print(arr.shape)
    plt.imshow(arr)
    plt.show()

def deconvolve(psf, measured_data):
    print(measured_data.shape, psf.shape)
    deconvolved, _ = restoration.unsupervised_wiener(color.rgb2gray(measured_data), color.rgb2gray(psf))
    return deconvolved

if __name__ == "__main__":
    for i in range(1, 20):
        scale_factor = 0.012*i
        psf = transform.resize(load_psf(), (scale_factor*1080, scale_factor*1920))
        measured_data = load_diffuser_image(10)

        result = deconvolve(psf, measured_data)
        display_array(result)
