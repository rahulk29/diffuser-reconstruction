import numpy as np
import math
import numpy.fft as fft
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps

import utils


class GDSolver:
    def __init__(
        self,
        f=0.5,
        nx=3,
        ny=4,
        fill_factor=0.8,
        iters=500,
        psf_file="../data/psf.npy",
        data_file="../data/diffuser/im10.npy",
        method="fista",
        proj=True,
        channel=0,
    ):
        self.f = f  # Downsampling factor
        self.nx = nx  # Number of sensor rows
        self.ny = ny  # Number of sensor columns
        self.fill_factor = fill_factor  # Sensor fill factor
        self.iters = iters  # Number of iterations to run with gradient descent
        self.psf_file = psf_file  # TIF file with PSF
        self.data_file = data_file  # TIF file with raw data
        self.method = method  # Gradient descent method
        self.channel = channel  # Color channel (usually 0, 1, 2)

        # Function for projecting final image
        self.proj_fn = lambda x: np.maximum(0, x) if proj else lambda x: x

        self.load_data()
        self.init_matrices()

    def run(self):
        return self.grad_descent()

    def crop(self, X):
        start_i = (self.padded_shape[0] - self.init_shape[0]) // 2
        end_i = start_i + self.init_shape[0]
        start_j = (self.padded_shape[1] // 2) - (self.init_shape[1] // 2)
        end_j = start_j + self.init_shape[1]
        return X[start_i:end_i, start_j:end_j]

    def pad(self, v):
        start_i = (self.padded_shape[0] - self.init_shape[0]) // 2
        end_i = start_i + self.init_shape[0]
        start_j = (self.padded_shape[1] // 2) - (self.init_shape[1] // 2)
        end_j = start_j + self.init_shape[1]
        vpad = np.zeros(self.padded_shape).astype(np.complex64)
        vpad[start_i:end_i, start_j:end_j] = v
        return vpad

    def crop_array(self, X):
        w, h = X.shape
        X_out = np.zeros(X.shape)
        for i in range(w // self.nx):
            start_i, end_i = w // self.nx * i, math.floor(
                w // self.nx * (i + self.fill_factor)
            )
            for j in range(h // self.ny):
                start_j, end_j = h // self.ny * j, math.floor(
                    h // self.ny * (j + self.fill_factor)
                )
                X_out[start_i:end_i, start_j:end_j] = X[start_i:end_i, start_j:end_j]
        return X_out

    def load_data(self):
        self.psf = np.load(self.psf_file).astype("float32")[:, :, self.channel]
        self.data = np.load(self.data_file).astype("float32")[:, :, self.channel]

        # Subtract non-trivial background
        bg = np.mean(self.psf[5:15, 5:15])
        self.psf -= bg
        self.data -= bg

        # Downsample PSF and data
        if self.psf.shape != self.data.shape:
            if (
                self.psf.shape[0] * self.data.shape[1]
                != self.psf.shape[1] * self.data.shape[0]
            ):
                raise Exception("PSF does not have the same aspect ratio as raw data")
            factor = self.psf.shape[0] / self.data.shape[0]
            if not np.isclose(utils.next_pow2(factor), factor):
                raise Exception(
                    "PSF and data cannot be resized as they do not differ in size by a power of 2"
                )
            if factor < 1:
                self.data = utils.resize(self.data, factor)
            else:
                self.psf = utils.resize(self.psf, 1 / factor)

        self.psf = utils.resize(self.psf, self.f)
        self.data = utils.resize(self.data, self.f)

        # Normalize PSF and measured data
        self.psf /= np.linalg.norm(self.psf.ravel())
        self.data /= np.linalg.norm(self.data.ravel())

    def init_matrices(self):
        pixel_start = (np.max(self.psf) + np.min(self.psf)) / 2
        x = np.ones(self.psf.shape) * pixel_start

        self.init_shape = self.psf.shape
        self.padded_shape = [utils.next_pow2(2 * n - 1) for n in self.init_shape]

        self.H = fft.fft2(fft.ifftshift(self.pad(self.psf)), norm="ortho")
        self.Hadj = np.conj(self.H)

        self.v = np.real(self.pad(x))

        self.alpha = np.real(1.8 / (np.max(self.Hadj * self.H)))

    def A(self, vk):
        Vk = fft.fft2(fft.ifftshift(vk))
        return self.crop_array(self.crop(fft.fftshift(fft.ifft2(self.H * Vk))))

    def A_herm(self, diff):
        xpad = self.pad(self.crop_array(diff))
        X = fft.fft2(fft.ifftshift(xpad))
        return fft.fftshift(fft.ifft2(self.Hadj * X))

    def grad(self, vk, b):
        Av = self.A(vk)
        diff = Av - b
        return np.real(self.A_herm(diff))

    def gd_update(self, vk, b):
        gradient = self.grad(vk, b)
        vk -= self.alpha * gradient
        vk = self.proj_fn(vk)

        return vk

    def nesterov_update(self, vk, b, p, mu):
        p_prev = p
        gradient = self.grad(vk, b)
        p = mu * p - self.alpha * gradient
        vk += -mu * p_prev + (1 + mu) * p
        vk = self.proj_fn(vk)

        return vk, p

    def fista_update(self, vk, b, tk, xk):
        x_k1 = xk
        gradient = self.grad(vk, b)
        vk -= self.alpha * gradient
        xk = self.proj_fn(vk)
        t_k1 = (1 + np.sqrt(1 + 4 * tk ** 2)) / 2
        vk = xk + (tk - 1) / t_k1 * (xk - x_k1)
        tk = t_k1

        return vk, tk, xk

    def grad_descent(self):
        vk = self.v
        b = self.data

        if self.method == "nesterov":
            p = 0
            mu = 0.9
        elif self.method == "fista":
            tk = 1
            xk = self.v

        for _ in range(self.iters):
            if self.method == "regular":
                vk = self.gd_update(vk, b)
            elif self.method == "nesterov":
                vk, p = self.nesterov_update(vk, b, p, mu)
            elif self.method == "fista":
                vk, tk, xk = self.fista_update(vk, b, tk, xk)

        return self.proj_fn(self.crop(vk))


if __name__ == "__main__":
    images = []
    for channel in range(3):
        print(f"channel = {channel}")
        gd_solver = GDSolver(fill_factor=0.8, channel=channel)
        images.append(gd_solver.run())

    image = np.stack(images, axis=-1)

    utils.display_array(image, "Final reconstruction", cmap="gray")

    plt.show()
