import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import os
from PIL import Image

import utils


class GDSolver:
    def __init__(
        self,
        f=0.125,
        nx=3,
        ny=4,
        fill_factor=0.8,
        iters=100,
        psf_file="../data/tutorial/psf_sample.tif",
        data_file="../data/tutorial/rawdata_hand_sample.tif",
    ):
        self.f = f # Downsampling factor
        self.nx = nx # Number of sensor rows
        self.ny = ny # Number of sensor columns
        self.fill_factor = fill_factor
        self.iters = iters
        self.psf_file = psf_file
        self.data_file = data_file

    def load_data(self):
        psf_img = Image.open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), self.psf_file)
        )
        self.psf = np.array(psf_img, dtype="float32")
        data_img = Image.open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), self.data_file)
        )
        self.data = np.array(data_img, dtype="float32")

        # Subtract non-trivial background
        bg = np.mean(self.psf[5:15, 5:15])
        self.psf -= bg
        self.data -= bg

        self.psf = utils.resize(self.psf, self.f)
        self.data = utils.resize(self.data, self.f)

        # Normalize PSF and measured data
        self.psf /= np.linalg.norm(self.psf.ravel())
        self.data /= np.linalg.norm(self.data.ravel())

    def init_matrices(self):

