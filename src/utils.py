import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import restoration

DATA_PATH = "../data"

def load_psf():
    return np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_PATH, 'psf.npy'))

def load_diffuser_image(n):
    return np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_PATH, 'diffuser', f'im{n}.npy'))

def load_lensed_image(n):
    return np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_PATH, 'lensed', f'im{n}.npy'))

def display_array(arr):
    plt.figure()
    plt.imshow(arr)
    plt.show()

def deconvolve(psf, measured_data):
    return restoration.richardson_lucy(measured_data, psf)

if __name__ == "__main__":
    psf = load_psf()
    measured_data = load_diffuser_image(10)

    result = deconvolve(psf, measured_data)
    display_array(result)
