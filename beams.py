import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

class laser_beam:
    def __init__(self, grid, spot_size, wavelength):
        self.grid = grid
        self.wavelength = wavelength

        self.x_field = np.exp(
            -np.square(self.grid.r_matrix) /
            np.square(spot_size))

    def propagate(self, distance):
        fresnel_phase = np.exp(
            -0.5j *
            np.square(self.grid.k_matrix) *
            distance /
            self.get_wavenumber())

        k_field = fftshift(fft2(self.x_field))
        k_field *= fresnel_phase
        self.x_field = ifft2(ifftshift(k_field))

    def get_intensity(self):
        return np.square(np.abs(self.x_field))

    def get_wavenumber(self):
        return 2.0 * np.pi / self.wavelength