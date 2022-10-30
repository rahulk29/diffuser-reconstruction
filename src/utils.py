import numpy as np
import os
import matplotlib.pyplot as plt

DATA_PATH = "../data"

def load_psf():
    return np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_PATH, 'psf.npy'))

def load_diffuser_image(n):
    return np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_PATH, 'diffuser', f'im{n}.npy'))

def load_diffuser_image(n):
    return np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), DATA_PATH, 'lensed', f'im{n}.npy'))

def display_array(arr):
    plt.figure()
    plt.imshow(arr)
    plt.show()
